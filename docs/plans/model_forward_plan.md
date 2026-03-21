# PR-2: Direct Model Forward - Detailed Implementation Plan

## Goal

Replace graph-based execution (`GraphExecutor::forward()` in `src/zedinfer/executor.cpp:53-128`)
with direct `Model::forward()` methods in `Qwen2Model` / `Qwen3Model`.
The graph infrastructure stays in the tree but is no longer on the inference hot path.

## Files to Modify

| Action | File | What Changes |
|--------|------|-------------|
| Modify | `include/frontend/models/base.hpp:68-103` | Add `virtual forward()` to `Model` |
| Modify | `include/frontend/models/qwen2.hpp:18-46` | Declare `forward()` override |
| Modify | `include/frontend/models/qwen3.hpp:18-46` | Declare `forward()` override |
| New | `src/frontend/models/qwen2_forward.cpp` | `Qwen2Model::forward()` implementation |
| New | `src/frontend/models/qwen3_forward.cpp` | `Qwen3Model::forward()` implementation |
| New | `include/zedinfer/scratch.hpp` | `DecodeScratch` struct |
| Modify | `include/zedinfer/engine.hpp:25-115` | Add `DecodeScratch` + `PositionIDsCache` members; keep `executor_` for warmup/profile |
| Modify | `src/zedinfer/engine.cpp:39-101` | Init scratch in `create()`; call `model_->forward()` from `generate_tokens()` |
| Modify | build config (xmake) | Add the two new `.cpp` to the `models` target |

---

## Memory Strategy: Decode-Only Scratch Buffers

### Problem with max-token-sized scratch

The original design document (`docs/design/runtime_refactor_plan.md`) proposed allocating
scratch buffers for `max_batch_tokens` (e.g., 2048). For Qwen3-8B (BF16):

```
hidden:  2048 * 4096 * 2 = 16 MB
q_proj:  2048 * 4096 * 2 = 16 MB
gate:    2048 * 11008 * 2 = 44 MB
up:      2048 * 11008 * 2 = 44 MB
logits:  2048 * 152064 * 2 = 594 MB (!)
norm_buf: 2048 * 4096 * 2 = 16 MB
attn_out: 2048 * 4096 * 2 = 16 MB
Total: ~746 MB
```

This is wasteful because:
1. Prefill happens once per prompt and its seq_len varies wildly (10 to 16K tokens).
2. Decode always processes exactly **1 token** per step — the dominant phase.
3. Allocating 2048-token buffers wastes memory that could hold more KV cache.

### Solution: Two-tier allocation

| Buffer Set | Size | Lifetime | Allocation |
|-----------|------|----------|------------|
| **DecodeScratch** | seq_len = **1** | Permanent (engine lifetime) | At `InferenceEngine::create()` |
| **Prefill temporaries** | seq_len = actual prompt length | Transient (one `forward()` call) | Via `Tensor::create()` per prefill call |

**DecodeScratch** for Qwen3-8B (BF16, seq_len=1):

```
hidden:   1 * 4096 * 2 =   8 KB
q_proj:   1 * 4096 * 2 =   8 KB
gate:     1 * 11008 * 2 =  22 KB
up:       1 * 11008 * 2 =  22 KB
logits:   1 * 152064 * 2 = 297 KB
norm_buf: 1 * 4096 * 2 =   8 KB
attn_out: 1 * 4096 * 2 =   8 KB
Total: ~373 KB  (0.04% of the 746 MB max-token approach)
```

Prefill is infrequent relative to decode, so its temporary allocations go through the
existing `BestFitMemoryPool` (which is already warmed up). The pool reclaims this
memory immediately after prefill returns.

### Why this works for CUDA Graphs (see next section)

CUDA graphs require **fixed memory addresses** across replays. The decode scratch
buffers have fixed addresses (allocated once, never moved). Prefill cannot be captured
in a CUDA graph anyway because its seq_len varies, so it doesn't need fixed buffers.

---

## CUDA Graph Acceleration for Decode

### Background

CUDA graph capture records a sequence of kernel launches into a graph, then replays
the entire graph with a single `cudaGraphLaunch()`. This eliminates:
- CPU-side kernel launch overhead (~5-10 us per kernel)
- CPU-side memory allocation overhead
- CPU-GPU synchronization points

For decode (seq_len=1, ~20 kernels per layer, 28-36 layers = 560-720 kernels per step),
the CPU-side overhead is a significant fraction of the ~5-15 ms decode step.

### Requirements for CUDA Graph Capture

1. **Fixed GPU memory addresses**: All input/output tensors must have the same device
   pointers across replays. Our `DecodeScratch` satisfies this.
2. **No CPU-side control flow dependent on GPU data**: No `if (gpu_value > threshold)`
   during the captured region. Our forward path has no such branches.
3. **No dynamic memory allocation inside the captured region**: Our decode forward
   uses only pre-allocated scratch views. Satisfied.
4. **Fixed kernel configurations**: Grid/block dims, shared memory sizes must be
   identical across replays. For decode (seq_len=1), all shapes are fixed. Satisfied.
5. **KV cache pointers may change** if `DynamicKVCache` grows. This is a problem.

### KV Cache Pointer Stability

`DynamicKVCache::grow_cache()` (`src/backend/kvcache/dynamic.cpp:61-97`) reallocates
and copies, changing tensor pointers. This invalidates a captured CUDA graph.

Solutions (in order of preference):
1. **Pre-allocate KV cache to max capacity** for the captured graph's session.
   E.g., allocate for 4096 tokens upfront. Growth never happens during capture/replay.
2. **Invalidate and re-capture** when growth occurs. Growth is rare after warmup
   (the aggressive strategy jumps to target capacity quickly).
3. **Defer CUDA graph** to after paged KV cache (PR-8/9), where block pool addresses
   are stable and only the block table changes (which can be updated in-place).

Recommendation: **Option 2** for now (re-capture on growth). Option 3 is the long-term answer.

### CUDA Graph Implementation Plan

```cpp
// In engine or a new CudaGraphRunner:
struct CudaGraphState {
    bool captured = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int captured_kv_capacity = 0;    // invalidation trigger

    // Captured input locations (copy data into these before replay)
    tensor_t input_token;            // [1] int32 — single decode token
    tensor_t position_id;            // [1] int64 — single position
};
```

Capture flow:
```
1. First decode step after warmup or KV growth:
   - cudaStreamBeginCapture(stream)
   - model_->forward(input_ids={token}, past_len, kvcache, decode_scratch, config)
   - cudaStreamEndCapture(stream, &graph)
   - cudaGraphInstantiate(&graph_exec, graph)
   - captured = true; captured_kv_capacity = kvcache.allocated_capacity()

2. Subsequent decode steps (while kv_capacity unchanged):
   - Copy next token into input_token tensor (H2D)
   - Copy position into position_id tensor (H2D)
   - cudaGraphLaunch(graph_exec, stream)
   - cudaStreamSynchronize(stream)
   - Read logits from decode_scratch.logits

3. If kvcache.allocated_capacity() != captured_kv_capacity:
   - cudaGraphExecDestroy(graph_exec)
   - captured = false
   - Fall back to step 1 to re-capture
```

### Expected Speedup

Conservative estimate for Qwen3-8B decode on RTX 4090:
- Without graph: ~8-12 ms/token (kernel launch overhead is ~10-20% of this)
- With graph: ~7-10 ms/token (save 1-2 ms from eliminated launch overhead)
- Improvement: ~10-20%

The benefit grows with smaller models (where compute per kernel is smaller and
launch overhead is a larger fraction).

### When to Implement

CUDA graph support is best added as a **sub-PR after the basic direct forward works**.
The implementation order within PR-2 should be:

1. First: implement `Model::forward()` with DecodeScratch, validate correctness.
2. Then: add CUDA graph capture/replay on top, benchmark improvement.

If CUDA graph adds too much complexity for a single PR, split it into PR-2b.

---

## DecodeScratch Definition

```cpp
// include/zedinfer/scratch.hpp

#pragma once
#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"
#include <cstddef>

namespace zedinfer {

namespace model { struct ModelConfig; }

// Pre-allocated buffers for decode phase (seq_len = 1).
// All tensors are allocated once at engine init and reused every decode step.
// Prefill uses transient Tensor::create() allocations (variable seq_len).
struct DecodeScratch {
    tensor_t hidden;     // [1, hidden_size]
    tensor_t norm_buf;   // [1, hidden_size]
    tensor_t q_proj;     // [1, hidden_size]
    tensor_t attn_out;   // [1, hidden_size]
    tensor_t gate;       // [1, intermediate_size]
    tensor_t up;         // [1, intermediate_size]
    tensor_t logits;     // [1, vocab_size]

    static DecodeScratch allocate(
        const model::ModelConfig &config,
        zedinferDeviceType_t device_type,
        int device_id,
        zedinferDataType_t dtype);
};

} // namespace zedinfer
```

Note: no `k_proj` / `v_proj` buffers — K and V are written directly into KV cache
slices (same pattern as current `executor.cpp:172-178`).

---

## Model::forward() Signature

```cpp
// Addition to include/frontend/models/base.hpp

class Model {
public:
    // ... existing virtuals ...

    // Direct forward pass (bypasses graph).
    //   input_ids:  token IDs for this step (prefill: many, decode: 1)
    //   past_len:   number of tokens already in kvcache
    //   kvcache:    session's KV cache (read K/V history, write new K/V)
    //   decode_scratch: pre-allocated buffers for seq_len=1 (nullptr for prefill)
    //   exec_config: device/dtype/max_seq_len
    // Returns: logits tensor [seq_len, vocab_size]
    virtual tensor_t forward(
        const std::vector<int> &input_ids,
        int past_len,
        kvcache::KVCache &kvcache,
        DecodeScratch *decode_scratch,
        const ExecutorConfig &exec_config) = 0;
};
```

When `decode_scratch` is non-null and `input_ids.size() == 1`, the forward pass uses
the pre-allocated decode buffers (zero allocation). When `decode_scratch` is null or
`input_ids.size() > 1`, it falls back to `Tensor::create()` per activation (same as
current graph executor, but without graph overhead).

---

## Forward Implementation Sketch (Qwen2)

```
Qwen2Model::forward(input_ids, past_len, kvcache, decode_scratch, exec_config):
    seq_len = input_ids.size()
    is_decode = (seq_len == 1 && decode_scratch != nullptr)

    // Helper: get or create buffer
    // If is_decode: return view of decode_scratch member
    // Else: Tensor::create(shape, dtype, device)
    auto buf = [&](tensor_t scratch_buf, vector<size_t> shape) -> tensor_t {
        if (is_decode) return scratch_buf->view(shape);
        else           return Tensor::create(shape, ...);
    };

    // 1. Prepare input_ids tensor
    input_ids_tensor = Tensor::create({seq_len}, I32, device); // small, OK to alloc
    input_ids_tensor->load(input_ids.data());

    // 2. Prepare position_ids
    position_ids = position_cache->get_slice(past_len, seq_len);

    // 3. Embedding
    hidden = buf(decode_scratch->hidden, {seq_len, hidden_size})
    ops::embedding(hidden, input_ids_tensor, W("embed_tokens.weight"))

    // 4. Transformer layers
    for layer in 0..num_layers:
        residual = hidden  // alias

        // 4a. Input norm
        normed = buf(decode_scratch->norm_buf, {seq_len, hidden_size})
        ops::rms_norm(normed, hidden, W(layer, "input_layernorm.weight"), eps)

        // 4b. Q projection
        q = buf(decode_scratch->q_proj, {seq_len, hidden_size})
        ops::linear(q, normed, W(layer, "self_attn.q_proj.weight"),
                                W(layer, "self_attn.q_proj.bias"))

        // 4c. K projection -> write directly to kvcache
        k_slot = kvcache.get_k_cache_slice(layer, past_len, seq_len)
        k_flat = k_slot->view({seq_len, kv_dim})
        ops::linear(k_flat, normed, W(layer, "self_attn.k_proj.weight"),
                                     W(layer, "self_attn.k_proj.bias"))

        // 4d. V projection -> write directly to kvcache
        v_slot = kvcache.get_v_cache_slice(layer, past_len, seq_len)
        v_flat = v_slot->view({seq_len, kv_dim})
        ops::linear(v_flat, normed, W(layer, "self_attn.v_proj.weight"),
                                     W(layer, "self_attn.v_proj.bias"))

        // 4e. RoPE on Q and K (in-place after reshape to 3D)
        q_3d = q->view({seq_len * nhead, head_dim})
        k_3d = k_flat->view({seq_len * nkvhead, head_dim})
        ops::rope(q_3d, q_3d, position_ids, rope_theta)
        ops::rope(k_3d, k_3d, position_ids, rope_theta)

        // 4f. Self-attention with full KV history
        k_full = kvcache.get_k_cache_slice(layer, past_len + seq_len)
        v_full = kvcache.get_v_cache_slice(layer, past_len + seq_len)
        attn_out = buf(decode_scratch->attn_out, {seq_len, nhead, head_dim})
        ops::self_attention(attn_out, q_3d->view({seq_len, nhead, head_dim}),
                            k_full, v_full, scale)

        // 4g. Output projection
        attn_flat = attn_out->view({seq_len, hidden_size})
        o_out = normed  // reuse norm_buf (consumed, safe)
        ops::linear(o_out, attn_flat, W(layer, "self_attn.o_proj.weight"), nullptr)

        // 4h. Residual 1
        ops::add(hidden, residual, o_out)   // hidden = residual + o_out

        // 4i. Post-attn norm -> MLP
        residual = hidden
        ops::rms_norm(normed, hidden, W(layer, "post_attention_layernorm.weight"), eps)

        // 4j. MLP: gate + up -> swiglu -> down
        gate = buf(decode_scratch->gate, {seq_len, intermediate_size})
        up   = buf(decode_scratch->up,   {seq_len, intermediate_size})
        ops::linear(gate, normed, W(layer, "mlp.gate_proj.weight"), nullptr)
        ops::linear(up,   normed, W(layer, "mlp.up_proj.weight"),   nullptr)
        ops::swiglu(gate, gate, up)   // in-place on gate

        down = normed  // reuse norm_buf again
        ops::linear(down, gate, W(layer, "mlp.down_proj.weight"), nullptr)

        // 4k. Residual 2
        ops::add(hidden, residual, down)

    // 5. Final norm + LM head
    normed = buf(decode_scratch->norm_buf, {seq_len, hidden_size})
    ops::rms_norm(normed, hidden, W("norm.weight"), eps)

    logits = buf(decode_scratch->logits, {seq_len, vocab_size})
    ops::linear(logits, normed, W("lm_head.weight"), nullptr)

    kvcache.update_seq_len(seq_len)
    return logits
```

### Qwen3 Differences

Qwen3 adds per-head Q/K norms and has no Q/K/V bias. The layer loop differs at:

- No bias on `q_proj`, `k_proj`, `v_proj`
- After `q_proj`: reshape to `[seq_len * nhead, head_dim]`, apply `ops::rms_norm`
  with `q_norm.weight`, then `ops::rope`
- After `k_proj`: reshape to `[seq_len * nkvhead, head_dim]`, apply `ops::rms_norm`
  with `k_norm.weight`, then `ops::rope`

Everything else (MLP, residuals, embedding, LM head) is identical.

---

## Buffer Reuse Analysis (Decode, seq_len=1)

Within one layer, tracking which scratch buffers are live at each step:

```
Step        | hidden | norm_buf | q_proj | attn_out | gate | up | logits
------------|--------|----------|--------|----------|------|----|-------
input_norm  | READ   | WRITE    |        |          |      |    |
q_proj      | .      | READ     | WRITE  |          |      |    |
k_proj      | .      | READ     | .      |          |      |    |  (writes to kvcache)
v_proj      | .      | READ     | .      |          |      |    |  (writes to kvcache)
rope(q)     | .      | .        | RW     |          |      |    |
rope(k)     | .      | .        | .      |          |      |    |  (in kvcache)
attention   | .      | .        | READ   | WRITE    |      |    |
o_proj      | .      | WRITE(*) | dead   | READ     |      |    |  (*) reuse norm_buf
residual1   | WRITE  | READ     |        | dead     |      |    |
post_norm   | READ   | WRITE    |        |          |      |    |
gate_proj   | .      | READ     |        |          | WRITE|    |
up_proj     | .      | READ     |        |          | .    |WRITE|
swiglu      | .      | .        |        |          | RW   |READ|
down_proj   | .      | WRITE(*) |        |          | READ | dead|  (*) reuse norm_buf
residual2   | WRITE  | READ     |        |          | dead |    |
```

Result: **7 buffers** suffice, with `norm_buf` reused for `o_proj` output and `down_proj` output.
No buffer is read and written simultaneously in any step.

For **prefill** (seq_len > 1), the same pattern holds but buffers are larger.
Since prefill uses `Tensor::create()`, the pool handles this automatically.

---

## Engine Wiring Changes

### `InferenceEngine::create()` — add scratch allocation

After model and tokenizer are loaded (after line `engine.cpp:91`):

```cpp
// Allocate decode scratch buffers
auto dtype = utils::str_to_dtype(model->config().torch_dtype);
auto scratch = DecodeScratch::allocate(model->config(), device.type(), device.id(), dtype);
```

Store `scratch` and `PositionIDsCache` as engine members (they outlive any session).

### `InferenceEngine::generate_tokens()` — call model_->forward()

Replace:
```cpp
tensor_t logits = executor_->forward(kvcache, input_ids, past_len);     // line 200
tensor_t logits = executor_->forward(kvcache, {next_token}, past_len);  // line 232
```

With:
```cpp
// Prefill: decode_scratch = nullptr -> allocates transient buffers
tensor_t logits = model_->forward(input_ids, past_len, kvcache, nullptr, exec_config_);

// Decode: decode_scratch = &scratch_ -> uses pre-allocated buffers
tensor_t logits = model_->forward({next_token}, past_len, kvcache, &scratch_, exec_config_);
```

### `warmup()` and `profile()` — keep using executor_

The graph executor is still useful for warmup and profiling because it validates
the graph structure. No change to `warmup()` or `profile()` in this PR.
They continue to call `executor_->forward()`.

Alternatively, if we want warmup to exercise the new path, we can call
`model_->forward()` there too. This is a minor decision — either is fine.

---

## Correctness Validation

The primary gate is: **decode output must be bit-identical** to the graph-based path
for the same input.

Validation approach:
1. Load a model (Qwen3-8B on GPU).
2. Run `generate_tokens()` with greedy sampling on a fixed prompt using the
   **old** graph executor path. Save output token IDs.
3. Switch to the **new** `model_->forward()` path. Run the same prompt.
4. Assert output token IDs are identical.

This can be done manually at first, then automated in the existing Python test
infrastructure (`tests/python/`).

The reason bit-identical output is expected: the new forward calls the exact same
`ops::*` functions with the exact same weight tensors and the same input data.
The only difference is buffer management (scratch vs fresh allocation), which
does not affect numerical output.

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| View dimension mismatch (e.g., `q_norm`/`k_norm` reshape in Qwen3) | Wrong output | Bit-identical test catches this immediately |
| `ops::add` in-place aliasing (hidden = residual + o_out where hidden == residual) | Corrupted residual | `ops::add` writes to `c` from `a` and `b`; if `c == a`, need to check add impl supports in-place. Current `add_cpu.cpp` iterates element-wise, safe for `c = a + b` when `c == a`. |
| `ops::swiglu` in-place (`gate = silu(gate) * up` where output == gate) | Corrupted gate | Current impl writes to `out` from `gate` and `up`. If `out == gate`, need to verify. Current swiglu CPU code: `out[i] = silu(gate[i]) * up[i]` — reads gate[i] before writing out[i], safe when out == gate. |
| DecodeScratch allocated on wrong device | Crash | Use same device as model weights. Assert at allocation time. |
| CUDA graph pointer invalidation on KV growth | Wrong results | Defer CUDA graph to sub-PR. Basic forward works without it. |

---

## Implementation Order (within this PR)

```
Step 1: Create include/zedinfer/scratch.hpp and src/zedinfer/scratch.cpp
        (DecodeScratch struct + allocate())

Step 2: Add forward() virtual to Model base class
        (include/frontend/models/base.hpp)

Step 3: Implement Qwen2Model::forward() in src/frontend/models/qwen2_forward.cpp
        Declare override in qwen2.hpp

Step 4: Implement Qwen3Model::forward() in src/frontend/models/qwen3_forward.cpp
        Declare override in qwen3.hpp

Step 5: Wire engine to use model_->forward() in generate_tokens()
        Add DecodeScratch + PositionIDsCache to InferenceEngine members
        Modify engine.cpp create() and generate_tokens()

Step 6: Update build files (xmake) to compile new .cpp files

Step 7: Test: run bench/chat/ping, verify output matches old path
```

## CUDA Graph — Deferred to PR-2b

CUDA graph capture is **not** included in the initial PR-2 to keep the patch focused.
It will be a follow-up PR-2b with:
- `CudaGraphRunner` class
- Capture/replay logic in `generate_tokens()` decode loop
- Re-capture on KV cache growth
- Benchmark: before/after decode latency

This separation keeps PR-2 reviewable and testable independently.

---

## Summary

| Aspect | Decision |
|--------|----------|
| Scratch buffer sizing | **Decode-only (seq_len=1)**: ~373 KB for 8B model. Prefill allocates transiently. |
| Buffer reuse | 7 named buffers; `norm_buf` reused for o_proj and down_proj within each layer. |
| CUDA graph | Deferred to PR-2b. Basic forward works without it. |
| KV cache writes | K/V written directly to cache slices (same as current executor). No K/V scratch buffers. |
| Graph code | Preserved in tree. Used by warmup/profile. Bypassed for generation. |
| Correctness gate | Bit-identical output tokens on greedy sampling vs old path. |
