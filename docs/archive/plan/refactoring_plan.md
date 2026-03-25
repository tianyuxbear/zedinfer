# ZedInfer Refactoring Plan

> Post PR-9 codebase cleanup. Ordered by impact on development velocity.

---

## Overview

After PR-1 through PR-9, the framework has all core serving features (paged KV cache, paged attention, continuous batching). But rapid feature addition has left structural debt that slows future work:

- **4 near-identical forward files** — adding a model or changing an op means editing 4 places
- **2 KV management paths** — Session mode (PagedKVCache) and batch mode (Request block table) coexist with duplicated scatter logic
- **KVCache base class polluted** with paged-specific methods that non-paged subclasses stub out
- **4 attention op functions** with model-side if/else dispatch
- **Engine class** accumulating unrelated responsibilities

The refactoring order is chosen to maximize developer velocity for upcoming work (HTTP API, FlashAttention integration, CUDA graphs, quantization, new model families).

---

## R1: Unify Model Forward (High Priority)

**Problem:**

```
qwen2.cpp              → forward()        ~150 lines
qwen3.cpp              → forward()        ~160 lines (diff: +Q/K norm)
qwen2_forward_batch.cpp → forward_batch() ~200 lines
qwen3_forward_batch.cpp → forward_batch() ~210 lines (diff: +Q/K norm)
```

~720 lines total, ~90% identical. The shared logic is:

```
embedding → for each layer { norm → Q/K/V proj → [Q/K norm] → RoPE →
KV write → attention → O proj → residual → norm → gate/up → swiglu →
down → residual } → final norm → lm_head
```

Model-specific differences are only:
- Qwen3 adds per-head Q/K RMSNorm after projection
- Qwen2 has Q/K/V bias, Qwen3 does not

Path-specific differences (forward vs forward_batch):
- KV write: single-request uses KVCache interface, batch uses scatter to blocks
- Attention: single-request uses paged decode/prefill, batch uses batched decode + per-request prefill

**Proposed design:**

```cpp
// Common layer loop in base class or free function
tensor_t transformer_forward(
    const ModelConfig &config,
    const ModelWeights &weights,
    const ForwardContext &ctx);  // abstracts single vs batch

// ForwardContext encapsulates the differences
struct ForwardContext {
    // Token IDs and position IDs (same for both modes)
    tensor_t ids, pos_ids;
    int total_tokens;

    // KV write strategy
    virtual void write_kv(int layer, tensor_t k, tensor_t v) = 0;
    // Attention strategy
    virtual tensor_t attention(int layer, tensor_t q, float scale) = 0;

    // Model-specific hooks
    virtual tensor_t apply_q_norm(tensor_t q, int layer) { return q; }
    virtual tensor_t apply_k_norm(tensor_t k, int layer) { return k; }
    virtual bool has_qkv_bias() const { return false; }
};
```

**Result:** One forward loop. Model differences are hooks. Single/batch differences are context implementations. Adding Llama = define hooks only.

**Files affected:**
- New: `src/frontend/models/transformer_forward.cpp` (shared loop)
- New: `include/frontend/models/forward_context.hpp`
- Modify: `qwen2.cpp`, `qwen3.cpp` (delegate to shared loop)
- Delete: `qwen2_forward_batch.cpp`, `qwen3_forward_batch.cpp` (merged into shared loop)

---

## R2: Unify KV Management (High Priority)

**Problem:**

Two KV management paths:

| | Session mode (`run_one`) | Batch mode (`forward_batch`) |
|--|---|---|
| KV owner | `PagedKVCache` (per-session object) | `InferenceRequest::block_table` |
| Write path | write buffer → scatter or direct-to-block | `scatter_kv_to_blocks()` static function |
| Read path | `kvcache.k_block_ids(L)` | `req->block_table.k_blocks[L].data()` |

Adding features (prefix caching, KV quantization) requires implementing in both paths.

**Proposed design:**

Converge on Request-owns-block-table for all modes:

```
Session::chat()
  → tokenize
  → build InferenceRequest with session's block_table
  → submit to scheduler
  → scheduler runs (run_one or batched)
  → update session's block_table seq_len
```

`PagedKVCache` becomes a thin wrapper that holds a `SequenceBlockTable` and delegates to `BlockPool`. The write buffer / scatter / gather logic moves to utility functions shared by both paths. Or `PagedKVCache` is eliminated entirely — the session just holds a `SequenceBlockTable`.

**Files affected:**
- Modify: `session.hpp/cpp` (hold block_table instead of KVCache)
- Modify: `scheduler.cpp` (`run_one` uses block_table directly)
- Simplify: `paged.hpp/cpp` (reduce to utility functions or delete)
- Delete: duplicate scatter functions

---

## R3: Clean KVCache Interface (Medium Priority)

**Problem:**

Base class has 6 paged-specific methods with empty default implementations:

```cpp
virtual bool is_paged() const { return false; }
virtual void *k_pool_data() const { return nullptr; }
virtual void *v_pool_data() const { return nullptr; }
virtual const int *k_block_ids(int) const { return nullptr; }
virtual const int *v_block_ids(int) const { return nullptr; }
virtual int block_size() const { return 0; }
virtual void scatter_layer_to_blocks(int) {}
```

**Proposed design:**

If R2 is done (Request owns block table), the model forward no longer queries `kvcache.is_paged()`. Instead, the `ForwardContext` (from R1) knows how to do KV write and attention. The KVCache base class shrinks back to its original interface.

If `DynamicKVCache` is still needed (warmup/profile), it keeps the original interface. `PagedKVCache` becomes internal to the batch path and doesn't need to be a `KVCache` subclass.

**Files affected:**
- Modify: `base.hpp` (remove paged methods)
- Modify: `paged.hpp` (decouple from KVCache base)

**Depends on:** R1, R2

---

## R4: Consolidate Attention Dispatch (Medium Priority)

**Problem:**

4 attention functions in `ops.hpp`:
```cpp
void self_attention(...)                   // contiguous KV
void paged_attention_decode(...)           // paged, single request
void paged_attention_prefill(...)          // paged, single request
void paged_attention_decode_batched(...)   // paged, multi request
```

Model code uses if/else chains to choose. Adding flash-attn varlen or speculative verification adds more.

**Proposed design:**

Unified attention interface:

```cpp
struct AttentionParams {
    tensor_t q;                      // [total_q_tokens, nhead, head_dim]
    const void *pool_base;           // block pool (nullptr = contiguous)
    const int *k_block_table;        // per-request K blocks (nullptr = contiguous)
    const int *v_block_table;
    const int *seq_lens;             // KV lengths per request
    int num_requests;
    int block_size;
    float scale;
    bool is_decode;                  // decode vs prefill dispatch hint
};

void attention(tensor_t out, const AttentionParams &params);
```

Internal dispatch by `params` fields: batched vs single, paged vs contiguous, decode vs prefill.

**Files affected:**
- New: `include/backend/ops/attention.hpp`
- Modify: `ops.hpp` (keep old functions as thin wrappers during transition)
- Modify: model forward code (use unified interface)

---

## R5: Split Engine Responsibilities (Low Priority)

**Problem:**

`InferenceEngine` has 12+ methods spanning model loading, session management, synchronous/asynchronous generation, profiling, and memory management.

**Proposed split:**

```
InferenceEngine    → model loading, session creation, configuration
ServingLoop        → step(), run_loop(), submit_async()
Profiler           → warmup(), profile()
```

**Files affected:**
- New: `include/zedinfer/serving_loop.hpp`
- Modify: `engine.hpp/cpp` (extract serving and profiling)

**Low urgency** — the current engine works, just large. Worth doing when HTTP API (PR-10) adds more serving logic.

---

## R6: Remove Dead Code (Low Priority)

**Problem:**

Graph execution code unused since PR-2:

```
include/frontend/graph/      → GraphBuilder, ComputeGraph
include/zedinfer/executor.hpp → GraphExecutor
src/frontend/graph/           → builder.cpp, graph.cpp
src/zedinfer/executor.cpp
```

**Action:** Delete. No backward compatibility concern — nothing calls it.

**Files affected:**
- Delete: `include/frontend/graph/`, `src/frontend/graph/`
- Delete: `include/zedinfer/executor.hpp`, `src/zedinfer/executor.cpp`
- Modify: `xmake.lua` (remove graph target if separate)

---

## Execution Order

```
R1 (Unify Forward) ──→ R3 (Clean KVCache) ──→ R4 (Attention Dispatch)
        |
R2 (Unify KV Mgmt) ──┘
                                              R5 (Split Engine) ── standalone
                                              R6 (Dead Code)    ── standalone
```

R1 and R2 are independent and can be done in parallel. R3 and R4 depend on R1/R2. R5 and R6 can be done anytime.

**Recommended approach:** Do R1 first (biggest pain point, blocks new model support), then R2 (simplifies all KV-related work including prefix caching). R3/R4 follow naturally. R5/R6 are opportunistic.

---

## Impact on Upcoming Features

| Upcoming Feature | Blocked by Current Debt? | Which Refactoring Helps |
|-----------------|-------------------------|------------------------|
| HTTP API (PR-10) | No | R5 (cleaner engine) |
| FlashAttention paged | Partially — need to add to 4 forward files | R1 (one forward), R4 (unified attention) |
| CUDA Graph capture | Partially — graph needs stable kernel sequence | R1 (one forward = one capture point) |
| New model family (Llama) | Yes — must write 4 forward files from scratch | R1 (just define hooks) |
| Prefix caching | Yes — must implement in both KV paths | R2 (unified KV path) |
| KV cache INT8 | Yes — must modify both KV paths | R2 |
| Speculative decoding | Partially — need new attention variant | R4 (unified dispatch) |
