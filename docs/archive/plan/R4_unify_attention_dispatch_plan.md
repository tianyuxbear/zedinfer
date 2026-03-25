# R4: Unify Attention Dispatch - Detailed Refactoring Plan

**Goal:** Consolidate 4 separate attention functions (with 10-15 positional parameters each) into a single parameterized `attention()` interface with a typed params struct. Simplify the dispatch layer and prepare for future attention variants (flash-attn varlen, speculative verification).

---

## 1. Current State Analysis (Post R1+R2+R3)

### 1.1 Four Attention Functions in ops.hpp

```cpp
// 1. Contiguous (warmup/profile only)
void self_attention(tensor_t out, tensor_t q, tensor_t k, tensor_t v, float scale);

// 2. Paged decode, single request — 13 parameters
void paged_attention_decode(tensor_t out, tensor_t q,
    const void *pool_base, const int *k_bt, const int *v_bt,
    int seq_len, float scale, dtype, device_type, device_id,
    int nhead, int nkvhead, int head_dim, int block_size);

// 3. Paged prefill, single request — 14 parameters
void paged_attention_prefill(tensor_t out, tensor_t q,
    const void *pool_base, const int *k_bt, const int *v_bt,
    int seqlen_q, int past_len, float scale, dtype, device_type, device_id,
    int nhead, int nkvhead, int head_dim, int block_size);

// 4. Paged decode, batched — 15 parameters
void paged_attention_decode_batched(tensor_t out, tensor_t q,
    const void *pool_base, const int *k_bts, const int *v_bts, const int *seq_lens,
    int num_reqs, int max_blocks, float scale, dtype, device_type, device_id,
    int nhead, int nkvhead, int head_dim, int block_size);
```

### 1.2 Where They're Called

After R1-R3, the call sites are well-isolated:

| Function | Called from | Frequency |
|----------|-----------|-----------|
| `self_attention` | `ContiguousForwardContext::attend()` | warmup/profile only |
| `paged_attention_decode` | `PagedForwardContext::attend()` | single-request decode |
| `paged_attention_decode_batched` | `PagedForwardContext::attend()` | multi-request decode |
| `paged_attention_prefill` | `PagedForwardContext::attend()` | per-request prefill |

**Observation:** The model code (`transformer_forward.cpp`) never sees these functions — it only calls `ctx.attend()`. The complexity is encapsulated in `PagedForwardContext::attend()`, which is ~80 lines of block table assembly + dispatch logic.

### 1.3 Pain Points

1. **Parameter explosion:** 13-15 positional raw parameters per function. Easy to misorder. No type safety.
2. **Repeated parameters:** `dtype`, `device_type`, `device_id`, `nhead`, `nkvhead`, `head_dim`, `block_size` are passed through every call but never change within a session.
3. **Future variants will add more functions:** flash-attn varlen (needs `cu_seqlens`), speculative verification (needs draft/verify split), prefix-shared attention.
4. **Dispatch in op.cpp is duplicated:** Each function has the same CPU/GPU switch pattern.

---

## 2. Target Design

### 2.1 AttentionParams Struct

Bundle all attention parameters into a typed struct:

```cpp
// include/backend/ops/attention_params.hpp

struct AttentionParams {
    // Output and query
    tensor_t out;
    tensor_t q;

    // Scale
    float scale;

    // Model dimensions (constant per session)
    int nhead;
    int nkvhead;
    int head_dim;

    // Device info
    zedinferDataType_t dtype;
    zedinferDeviceType_t device_type;
    int device_id;

    // --- Mode-specific fields ---

    // Contiguous mode (warmup/profile): k and v are contiguous tensors
    tensor_t k_contiguous = nullptr;  // non-null = contiguous mode
    tensor_t v_contiguous = nullptr;

    // Paged mode: block pool + block tables
    const void *pool_base = nullptr;  // non-null = paged mode
    int block_size = 0;

    // Single-request paged
    const int *k_block_table = nullptr;
    const int *v_block_table = nullptr;
    int seq_len = 0;             // KV length for decode
    int seqlen_q = 0;            // query length (1=decode, >1=prefill)
    int past_len = 0;            // for prefill causal mask

    // Batched paged decode
    const int *k_block_tables_batched = nullptr;  // non-null = batched mode
    const int *v_block_tables_batched = nullptr;
    const int *seq_lens_batched = nullptr;
    int num_requests = 0;
    int max_blocks_per_seq = 0;
};
```

### 2.2 Single Dispatch Function

```cpp
// include/backend/ops/ops.hpp (simplified)
void attention(const AttentionParams &params);
```

Internal dispatch logic (in `op.cpp`):

```cpp
void attention(const AttentionParams &params) {
    if (params.k_contiguous) {
        // Contiguous mode
        return self_attention_impl(params);
    }
    if (params.k_block_tables_batched) {
        // Batched paged decode
        return paged_attention_decode_batched_impl(params);
    }
    if (params.seqlen_q == 1) {
        // Single paged decode
        return paged_attention_decode_impl(params);
    }
    // Single paged prefill
    return paged_attention_prefill_impl(params);
}
```

### 2.3 AttentionConfig for Session-Constant Parameters

Parameters that don't change within a session (`nhead`, `nkvhead`, `head_dim`, `dtype`, `device_type`, `device_id`, `block_size`) can be factored into a config struct created once:

```cpp
struct AttentionConfig {
    int nhead;
    int nkvhead;
    int head_dim;
    float scale;
    int block_size;
    zedinferDataType_t dtype;
    zedinferDeviceType_t device_type;
    int device_id;
};

// AttentionParams then becomes lighter:
struct AttentionParams {
    const AttentionConfig &config;
    tensor_t out;
    tensor_t q;
    // ... mode-specific fields only
};
```

`AttentionConfig` is created once per `PagedForwardContext` / `ContiguousForwardContext` and reused across all layers.

---

## 3. Simplification of PagedForwardContext::attend()

Current `attend()` is ~80 lines with block table assembly, GPU tensor uploads, and dispatch. With the unified params struct:

```cpp
tensor_t PagedForwardContext::attend(int layer, tensor_t q_rope, ...) {
    auto attn = make({N, nhead, head_dim});

    // Decode slots
    if (num_decode > 0) {
        AttentionParams params{attn_config_};
        params.out = decode_out;
        params.q = decode_q;
        params.pool_base = pool_.block_data(0);
        if (num_decode == 1) {
            params.k_block_table = ...;
            params.v_block_table = ...;
            params.seq_len = ...;
            params.seqlen_q = 1;
        } else {
            params.k_block_tables_batched = ...;
            params.v_block_tables_batched = ...;
            params.seq_lens_batched = ...;
            params.num_requests = num_decode;
            params.max_blocks_per_seq = max_blocks;
        }
        ops::attention(params);
    }

    // Prefill slots
    for (slot : prefill_slots) {
        AttentionParams params{attn_config_};
        params.out = pf_out;
        params.q = pf_q;
        params.pool_base = pool_.block_data(0);
        params.k_block_table = ...;
        params.v_block_table = ...;
        params.seqlen_q = chunk;
        params.past_len = past;
        ops::attention(params);
    }

    return attn;
}
```

No more 15-parameter function calls. Parameters are named and typed.

---

## 4. Files Affected

### New Files

| File | Purpose |
|------|---------|
| `include/backend/ops/attention_params.hpp` | `AttentionConfig`, `AttentionParams` structs |

### Modified Files

| File | Change |
|------|--------|
| `include/backend/ops/ops.hpp` | Add `attention(const AttentionParams &)`. Keep old functions temporarily for transition. |
| `src/backend/ops/self_attention/op.cpp` | Add unified `attention()` dispatch. Old functions delegate to it. |
| `src/frontend/models/paged_forward_context.cpp` | Use `AttentionConfig` + `AttentionParams` instead of raw parameters |
| `src/frontend/models/forward_context.cpp` | ContiguousForwardContext uses `AttentionParams` with contiguous mode |

### Later Cleanup (after verified)

| File | Change |
|------|--------|
| `include/backend/ops/ops.hpp` | Remove old `paged_attention_decode/prefill/batched` functions |
| `src/backend/ops/self_attention/op.cpp` | Remove old dispatch functions |

---

## 5. Implementation Tasks

### Task 1: Define AttentionConfig and AttentionParams

- [ ] Create `include/backend/ops/attention_params.hpp`
- [ ] Define `AttentionConfig` (session-constant params)
- [ ] Define `AttentionParams` (per-call params with mode fields)
- [ ] Build to verify (header only, no callers yet)

### Task 2: Add unified attention() dispatch

- [ ] Add `void attention(const AttentionParams &params)` to `ops.hpp`
- [ ] Implement in `op.cpp`: dispatch by params fields (contiguous / paged decode / paged prefill / batched)
- [ ] Internally call existing backend functions (nvidia/cpu kernels unchanged)
- [ ] Build to verify

### Task 3: Update ForwardContext callers

- [ ] `PagedForwardContext`: create `AttentionConfig` in constructor, use `AttentionParams` in `attend()`
- [ ] `ContiguousForwardContext`: use `AttentionParams` with contiguous mode in `attend()`
- [ ] Build and test — verify identical output

### Task 4: Remove old functions

- [ ] Remove `paged_attention_decode`, `paged_attention_prefill`, `paged_attention_decode_batched`, `self_attention` from `ops.hpp`
- [ ] Remove old dispatch functions from `op.cpp`
- [ ] Build and test

---

## 6. Impact on Future Work

| Feature | Before (4 functions) | After (unified params) |
|---------|---------------------|----------------------|
| **flash-attn varlen** | Add 5th function with `cu_seqlens` | Add `cu_seqlens` field to `AttentionParams`, new dispatch branch |
| **Speculative verification** | Add 6th function | Add `is_verification` flag, reuse existing kernel with different Q length |
| **Prefix-shared attention** | Unclear where to add | Add `shared_prefix_len` field |
| **New backend (e.g., ROCm)** | Add to 4 switch statements | Add to 1 switch statement |
| **Parameter tuning (tile size, etc.)** | Can't pass without changing 4 signatures | Add to `AttentionConfig` |

---

## 7. Priority Assessment

R4 is **medium priority**. After R1-R3, the attention dispatch is already encapsulated in `ForwardContext` — the model code doesn't see it. The main beneficiaries are:

1. **Code readability** — named params vs 15 positional args
2. **Future attention variants** — one struct to extend vs new functions
3. **Backend maintainability** — one dispatch vs four

R4 does **not** block any feature (HTTP API, CUDA graphs, flash-attn). It's a quality-of-life improvement that pays off when adding new attention backends. Can be deferred if feature work is more urgent.
