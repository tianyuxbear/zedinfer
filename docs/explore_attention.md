# Attention Implementation Deep Dive

> Exploration checkpoint. Covers all attention paths: CPU/CUDA, paged/contiguous, decode/prefill, warmup/serving, and cuDNN integration.

---

## 1. Unified Dispatch Interface

All attention calls go through a single entry point:

```cpp
// include/backend/ops/ops.hpp:19
void attention(const AttentionParams &params);
```

Dispatch logic in `src/backend/ops/self_attention/op.cpp:154-165`:

```cpp
void attention(const AttentionParams &params) {
    if (params.is_contiguous())   return dispatch_contiguous(params);
    if (params.is_batched())      return dispatch_paged_decode_batched(params);
    if (params.is_decode())       return dispatch_paged_decode(params);
    return dispatch_paged_prefill(params);
}
```

Mode detection (`include/backend/ops/attention_params.hpp:60-62`):

| Method | Condition | Meaning |
|--------|-----------|---------|
| `is_contiguous()` | `k_contiguous != nullptr` | Warmup/profile path (DynamicKVCache) |
| `is_batched()` | `batched_k_block_tables != nullptr` | Multi-request paged decode |
| `is_decode()` | `seq_len > 0` and not contiguous/batched | Single-request paged decode |
| (else) | none of the above | Paged prefill |

---

## 2. File Inventory

### Headers

| File | Lines | Purpose |
|------|-------|---------|
| `include/backend/ops/ops.hpp` | 22 | `void attention(const AttentionParams &)` declaration |
| `include/backend/ops/attention_params.hpp` | 66 | `AttentionConfig` + `AttentionParams` structs |

### Dispatch

| File | Lines | Purpose |
|------|-------|---------|
| `src/backend/ops/self_attention/op.cpp` | 167 | Unified dispatch: CPU/GPU x contiguous/paged/batched |

### CPU Kernels

| File | Lines | Purpose |
|------|-------|---------|
| `src/backend/ops/self_attention/cpu/self_attention_cpu.cpp` | 112 | Contiguous attention (warmup/profile) |
| `src/backend/ops/self_attention/cpu/paged_attention_cpu.cpp` | 244 | Paged decode + prefill + batched decode |
| `include/backend/ops/self_attention/cpu/self_attention_cpu.hpp` | 9 | Header |
| `include/backend/ops/self_attention/cpu/paged_attention_cpu.hpp` | 38 | Header |

### CUDA Kernels

| File | Lines | Purpose |
|------|-------|---------|
| `src/backend/ops/self_attention/nvidia/self_attention_nvidia.cu` | 350 | Contiguous decode + prefill kernels + cuDNN dispatch |
| `src/backend/ops/self_attention/nvidia/paged_attention_nvidia.cu` | 538 | Paged decode + prefill + batched decode kernels |
| `src/backend/ops/self_attention/nvidia/flash_attention_cudnn.cu` | 349 | cuDNN FlashAttention wrapper (conditional: `USE_CUDNN_FLASH`) |
| `include/backend/ops/self_attention/nvidia/self_attention_nvidia.cuh` | 9 | Header |
| `include/backend/ops/self_attention/nvidia/paged_attention_nvidia.cuh` | 53 | Header |
| `include/backend/ops/self_attention/nvidia/flash_attention_cudnn.cuh` | 22 | Header |

### Callers (ForwardContext)

| File | Lines | Purpose |
|------|-------|---------|
| `src/frontend/models/forward_context.cpp` | 85 | `ContiguousForwardContext::attend()` — warmup/profile |
| `src/frontend/models/paged_forward_context.cpp` | 319 | `PagedForwardContext::attend()` — serving |

---

## 3. Execution Paths Overview

```
                            ops::attention(params)
                                    |
                 +------------------+------------------+
                 |                  |                  |
          is_contiguous?      is_batched?        is_decode?
           (warmup/profile)   (multi-req decode)  (single-req decode)
                 |                  |                  |
        dispatch_contiguous  dispatch_paged_     dispatch_paged_
                 |           decode_batched       decode
                 |                  |                  |
           +-----+-----+          |                  |          (else)
           |           |          |                  |     dispatch_paged_
         decode     prefill       |                  |       prefill
        (sl==1)    (sl>1)         |                  |
           |           |          |                  |
      custom      try cuDNN   batched kernel    paged decode   paged prefill
      kernel    FlashAttn*       (CUDA)         kernel          kernel
                fallback to
              custom kernel

* cuDNN FlashAttention: only for contiguous prefill, FP16/BF16, USE_CUDNN_FLASH
```

---

## 4. Warmup / Profile Path (Contiguous Mode)

### Call Chain

```
Profiler::warmup() / profile()
  -> transformer_forward(..., ContiguousForwardContext)
    -> ContiguousForwardContext::attend()    [forward_context.cpp:56-78]
      -> ops::attention(params)             params.k_contiguous != nullptr
        -> dispatch_contiguous()            [op.cpp:18-44]
```

### How It Works

`ContiguousForwardContext` uses `DynamicKVCache` (contiguous memory, no block table). At `forward_context.cpp:73-74`:

```cpp
params.k_contiguous = kvcache_.get_k_cache_slice(layer, past_len_ + sl_);
params.v_contiguous = kvcache_.get_v_cache_slice(layer, past_len_ + sl_);
```

These return contiguous `[total_len, nkvhead, head_dim]` tensors. `AttentionConfig.block_size` is set to 0.

### CUDA Contiguous Dispatch

In `self_attention_nvidia.cu:267-347`:

**Decode (seqlen == 1):**
- `self_attention_decode_kernel`: Grid=(nhead), Block=256
- Online softmax, TILE_KV=256 tiled K/V traversal
- Shared memory: Q vector + score tile + warp-level reduce buffer
- **Does NOT use cuDNN** (memory-bound single-query, FlashAttention provides no benefit)

**Prefill (seqlen > 1):**
1. If `USE_CUDNN_FLASH` defined and dtype is FP16 or BF16:
   - Try `cudnn_flash::flash_attention_prefill()` — returns true/false
   - On success: done. On failure: fallback.
2. Fallback: `self_attention_prefill_kernel`: Grid=(seqlen, nhead), Block=256
   - Each block handles one (query_position, head) pair
   - Causal mask applied: only attend to positions 0..(past_len + query_pos)
   - Online softmax, same tile structure as decode kernel

**This is the ONLY place cuDNN FlashAttention is used.**

### CPU Contiguous Dispatch

`self_attention_cpu.cpp:12-111`:
- OMP parallel for collapse(2) over (query_pos, head)
- Per-thread `std::vector<float>` score buffer
- Naive GQA with causal mask
- FP32 accumulation, cast for BF16/FP16

---

## 5. Serving Path: Paged Decode (Single Request)

### When Triggered

Only 1 decode request in the batch. `PagedForwardContext::attend()` at `paged_forward_context.cpp:248-266` sets `params.seq_len` and single-request block table pointers.

### CUDA Kernel

`paged_attention_decode_kernel` at `paged_attention_nvidia.cu:72-184`:
- Grid=(nhead), Block=256
- Same algorithmic structure as contiguous decode, but K/V access goes through block table:

```cpp
int block_idx = j_global / block_size;
int block_offset = j_global % block_size;
int k_physical = k_block_table[block_idx] * block_size + block_offset;
const T *k_ptr = pool_base + k_physical * nkvhead * d + kvh * d;
```

- Online softmax with tile-based K/V processing
- All K/V tokens are valid (no causal mask needed for decode)

### CPU Kernel

`paged_attention_decode` at `paged_attention_cpu.cpp:16-110`:
- OMP parallel over heads
- Same block-table-indexed access pattern
- Per-thread score buffer

---

## 6. Serving Path: Paged Batched Decode (Multiple Requests)

### When Triggered

Multiple decode requests in the batch. `PagedForwardContext::attend()` at `paged_forward_context.cpp:267-286` flattens all request block tables into GPU tensors and sets `params.batched_k_block_tables`.

### GPU Block Table Caching

At `paged_forward_context.cpp:156-224`, block tables are uploaded to GPU **once** on the first `attend()` call, then reused for all subsequent layers within the same forward pass. This avoids 28x (one per layer) redundant CPU->GPU uploads.

### CUDA Kernel

`paged_attention_decode_batched_kernel` at `paged_attention_nvidia.cu:194-306`:
- **Grid=(num_requests, nhead)**: each block handles one (request, head) pair
- Each block reads its own block table: `k_bt = k_block_tables + req_idx * max_blocks_per_seq`
- Each block reads its own seq_len: `seq_lens[req_idx]`
- Requests are completely isolated — no cross-request attention leakage
- Same online softmax + tile structure

### CPU Kernel

`paged_attention_decode_batched` at `paged_attention_cpu.cpp:215-241`:
- Simple loop over requests, calls `paged_attention_decode()` for each

---

## 7. Serving Path: Paged Prefill

### When Triggered

Prefill slots in the batch. `PagedForwardContext::attend()` at `paged_forward_context.cpp:289-305` iterates over non-decode slots, launching the prefill kernel **per request sequentially**.

### CUDA Kernel

`paged_attention_prefill_kernel` at `paged_attention_nvidia.cu:360-454`:
- Grid=(seqlen_q, nhead): one block per (query_position, head)
- Causal mask: `causal_end = past_len + query_pos + 1`
- Block-table-indexed K/V access (same as paged decode)
- **Naive implementation**: no IO-aware tiling, no Q-Q reuse across query positions, no shared memory for V

### CPU Kernel

`paged_attention_prefill` at `paged_attention_cpu.cpp:116-209`:
- OMP parallel collapse(2) over (query_pos, head)
- Causal mask applied per query position
- Block-table-indexed K/V access

### Known Weakness

This is the kernel flagged in `docs/system_analysis.md` as **2-4x slower than FlashAttention-2** for long prompts (2K+). Each block independently processes its query position against all valid KV tokens with no IO-aware tiling. The docs recommend integrating flash-attn C++ API with `block_table` + `cu_seqlens` support as the next optimization.

---

## 8. cuDNN FlashAttention — Detailed Analysis

### File

`src/backend/ops/self_attention/nvidia/flash_attention_cudnn.cu` (349 lines)

### Scope of Usage

**Extremely narrow:**
- Only used in **contiguous prefill** path (warmup/profile)
- Only supports **FP16** and **BF16** (not FP32)
- Requires compile-time **`USE_CUDNN_FLASH`** macro
- **NOT used for paged attention** — the actual serving path never calls cuDNN

### Why It Doesn't Apply to Paged Prefill

cuDNN's `SDPA` (Scaled Dot-Product Attention) expects **contiguous** K/V tensors with known strides. The paged KV cache stores K/V in non-contiguous blocks indexed by a block table. To use cuDNN for paged attention, you would need to either:
1. Gather blocks into contiguous memory first (defeats the purpose of paging)
2. Use a library that natively supports block-table-indexed K/V (flash-attn C++ API)

### Implementation Details

**1. Graph Caching Mechanism**

Cache key: `(seqlen_q, seqlen_kv, nhead, nkvhead, head_dim, dtype, kv_seq_major)`

```cpp
static std::unordered_map<FlashAttnKey, std::shared_ptr<CachedGraph>, FlashAttnKeyHash> graph_cache;
```

**2. K/V Layout Adaptation Strategy**

zedinfer stores K/V as `[seqlen, nkvhead, head_dim]` (seq-major). cuDNN expects `[1, nkvhead, seqlen, head_dim]` (head-major by default). The code tries two approaches:

1. **First try: seq-major strides** — Tell cuDNN that K/V has custom strides `[skv*nkvh*hd, hd, nkvh*hd, 1]`. If cuDNN accepts this, no K/V transpose is needed.
2. **Fallback: head-major** — Transpose K/V to `[nkvhead, seqlen, head_dim]`. Works everywhere but adds transpose overhead.
3. **If neither works:** return `false`, caller uses custom kernel.

**3. Transpose Overhead**

| Tensor | Always transposed? | Size |
|--------|--------------------|------|
| Q | Yes (small: seqlen * nhead * head_dim) | Negligible |
| O | Yes (same as Q) | Negligible |
| K | Only if cuDNN rejects seq-major strides | Could be large: total_len * nkvhead * head_dim |
| V | Only if cuDNN rejects seq-major strides | Same as K |

**4. cuDNN Frontend API Usage**

```cpp
auto sdpa_opts = fe::graph::SDPA_attributes()
    .set_name("flash_attn")
    .set_generate_stats(true)    // needed for backward, but required by API
    .set_causal_mask(true)
    .set_attn_scale(scale);

auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);
```

The graph is built once, then executed with different data pointers via `variant_pack`.

**5. Buffer Management**

Pre-allocated static buffers that grow on demand:

```cpp
struct TransposeBuffers {
    std::byte *q, *o;           // always used (Q/O transpose)
    std::byte *k, *v;           // only when kv_seq_major=false
    void *stats;                // cuDNN requires stats output
    size_t q_cap, kv_cap, stats_cap;
};
```

**6. Public API**

```cpp
bool flash_attention_prefill(
    std::byte *output, const std::byte *q, const std::byte *k, const std::byte *v,
    float scale, zedinferDataType_t type,
    int seqlen, int nhead, int head_dim, int total_len, int nkvhead);
```

Returns `true` on success, `false` if the configuration is unsupported by cuDNN.

---

## 9. Summary: Which Kernel Runs When

| Scenario | ForwardContext | KV Storage | CPU Kernel | CUDA Kernel | cuDNN? |
|----------|---------------|-----------|------------|-------------|--------|
| Warmup/profile decode | Contiguous | DynamicKVCache (contiguous) | `self_attention` (naive GQA, OMP) | `self_attention_decode_kernel` (online softmax, tiled) | No |
| Warmup/profile prefill | Contiguous | DynamicKVCache (contiguous) | `self_attention` (naive GQA, OMP) | Try cuDNN FlashAttn (FP16/BF16) -> fallback `self_attention_prefill_kernel` | **Yes, only here** |
| Serving: 1 decode request | Paged | BlockPool (block table) | `paged_attention_decode` | `paged_attention_decode_kernel` | No |
| Serving: N decode requests | Paged | BlockPool (block table) | loops over single-request decode | `paged_attention_decode_batched_kernel` Grid=(N, nhead) | No |
| Serving: prefill | Paged | BlockPool (block table) | `paged_attention_prefill` | `paged_attention_prefill_kernel` (naive, no IO-aware tiling) | No |

---

## 10. Performance Gaps and Next Steps

### Paged Prefill Is the Main Bottleneck

The current `paged_attention_prefill_kernel` (`paged_attention_nvidia.cu:360-454`) is a naive implementation:

- **No IO-aware tiling** (FlashAttention-2 reads K/V from HBM once, reuses in SRAM)
- **No Q-Q reuse** (each block re-reads all K tokens for its one query position)
- **Grid=(seqlen_q, nhead)** means 65K blocks for seqlen=2048, nhead=32
- **No shared memory for V** — V read from global memory in inner loop

For long prompts (2K+), this is **2-4x slower than FlashAttention-2**.

### Prefill Requests Not Merged

Each prefill request launches a separate kernel per layer. 10 concurrent prefills = 10 kernel launches per layer. A varlen attention API (`cu_seqlens`) would batch them into one.

### Decode V Aggregation Waste

In the decode kernels, block_size=256 threads but head_dim=128 — only 128 threads participate in V aggregation, the other 128 idle (50% waste).

### Recommended Next Step

Integrate **flash-attn C++ API** (Dao-AILab/flash-attention) for paged prefill:
- Supports `block_table` parameter natively
- Supports `cu_seqlens` for variable-length batched prefill
- IO-aware tiling, Q-Q reuse, optimized for Hopper/Blackwell
- Would replace `paged_attention_prefill_kernel` on GPU
- cuDNN FlashAttention wrapper could then be removed (it only covers contiguous prefill which is warmup-only)

CPU paged prefill is adequate for now (not the primary serving target).
