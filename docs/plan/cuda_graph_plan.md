# CUDA Graph Optimization Plan for zedinfer

> Date: 2026-03-24
> Status: Draft
> Depends on: None (Phase 1 is standalone; Phase 2 depends on Phase 1)
> Builds upon: `docs/notes/optimization_research.md` Section 2

---

## Table of Contents

1. [Phase 1: DecodeScratch Pre-allocation](#phase-1-decodescratch-pre-allocation)
   - [1.1 Tensor::create() Audit](#11-tensorcreate-audit)
   - [1.2 DecodeScratch Struct Design](#12-decodescratch-struct-design)
   - [1.3 Buffer Sizing per Model](#13-buffer-sizing-per-model)
   - [1.4 Total Scratch Memory](#14-total-scratch-memory)
   - [1.5 transformer_forward Modification](#15-transformer_forward-modification)
   - [1.6 Files Affected (Phase 1)](#16-files-affected-phase-1)
2. [Phase 2: CUDA Graph Capture](#phase-2-cuda-graph-capture)
   - [2.1 What Can vs Cannot Be Captured](#21-what-can-vs-cannot-be-captured)
   - [2.2 The KV Scatter Problem](#22-the-kv-scatter-problem)
   - [2.3 The Attention Block Table Problem](#23-the-attention-block-table-problem)
   - [2.4 The Argmax D2H Sync Point](#24-the-argmax-d2h-sync-point)
   - [2.5 The cuBLASLt Descriptor Problem](#25-the-cublaslt-descriptor-problem)
   - [2.6 CudaGraphRunner Class Design](#26-cudagraphrunner-class-design)
   - [2.7 Capture / Replay / Invalidation Lifecycle](#27-capture--replay--invalidation-lifecycle)
   - [2.8 Updating Variable Inputs Between Replays](#28-updating-variable-inputs-between-replays)
   - [2.9 Files Affected (Phase 2)](#29-files-affected-phase-2)
3. [Expected Impact](#expected-impact)
4. [Implementation Order](#implementation-order)
5. [Risks](#risks)

---

## Phase 1: DecodeScratch Pre-allocation

### 1.1 Tensor::create() Audit

Every `Tensor::create()` call in the decode path (N=1) allocates device memory via
`runtime.allocateDeviceStorage()`, which goes through `BestFitMemoryPool::allocate()`.
Each call incurs:
- Memory pool lookup (size-class bin search, best-fit scan)
- Possible pool growth (`cudaMalloc`)
- `shared_ptr<Tensor>` construction + `shared_ptr<Storage>` reference counting
- On destruction (when `tensor_t` goes out of scope): `BestFitMemoryPool::deallocate()` +
  possible coalescing

The following is a **line-by-line audit** of every `Tensor::create()` call that executes
during a single decode step (N=1) through the two code paths:
`transformer_forward.cpp` and `paged_forward_context.cpp`.

#### A. Inside `prepare_inputs()` (`paged_forward_context.cpp` lines 52-58)

| # | Line | Variable | Shape (N=1) | Notes |
|---|------|----------|-------------|-------|
| 1 | 52 | `ids` | `{1}` I32 | Input token ID, uploaded via `load()` |
| 2 | 56 | `pos_ids` | `{1}` I64 | Position ID, uploaded via `load()` |

Both also call `Tensor::load()` which does a synchronous `memcpy_sync(H2D)`.

#### B. Before the layer loop (`transformer_forward.cpp`)

| # | Line | Variable | Shape (N=1) | Notes |
|---|------|----------|-------------|-------|
| 3 | 39 | `hidden` | `{1, hidden_size}` | Embedding output |

#### C. Per layer, inside the loop (`transformer_forward.cpp` lines 43-116)

For **Qwen3** (`has_qk_norm=true`, `has_qkv_bias=false`):

| # | Line | Variable | Shape (N=1) | Notes |
|---|------|----------|-------------|-------|
| 1 | 47 | `normed` | `{1, hidden_size}` | Input layernorm |
| 2 | 51 | `q` | `{1, hidden_size}` | Q projection |
| 3 | 54 | `k` | `{1, kv_dim}` | K projection |
| 4 | 57 | `v` | `{1, kv_dim}` | V projection |
| 5 | 63 | `q_normed` | `{nhead, head_dim}` | Q per-head norm (Qwen3 only) |
| 6 | 67 | `k_normed` | `{nkvhead, head_dim}` | K per-head norm (Qwen3 only) |
| 7 | 79 | `q_rope` | `{1, nhead, head_dim}` | Q after RoPE |
| 8 | 82 | `k_rope` | `{1, nkvhead, head_dim}` | K after RoPE |
| 9 | - | (write_kv) | (no allocation) | D2D memcpy only |
| 10 | 241* | `attn` | `{1, nhead, head_dim}` | Attention output (inside `attend()`) |
| 11 | 92 | `o` | `{1, hidden_size}` | O projection |
| 12 | 96 | `h1` | `{1, hidden_size}` | Residual add |
| 13 | 100 | `normed` | `{1, hidden_size}` | Post-attention layernorm |
| 14 | 103 | `gate` | `{1, inter}` | MLP gate projection |
| 15 | 104 | `up` | `{1, inter}` | MLP up projection |
| 16 | 108 | `act` | `{1, inter}` | SwiGLU activation |
| 17 | 111 | `down` | `{1, hidden_size}` | MLP down projection |
| 18 | 114 | `hidden` | `{1, hidden_size}` | Residual add (new hidden state) |

*Line 241 refers to `paged_forward_context.cpp` inside `attend()`.

**Per layer: 18 `Tensor::create()` calls for Qwen3, 16 for Qwen2** (no items 5,6).

#### D. After the layer loop (`transformer_forward.cpp`)

| # | Line | Variable | Shape (N=1) | Notes |
|---|------|----------|-------------|-------|
| 1 | 119 | `final_normed` | `{1, hidden_size}` | Final RMS norm |
| 2 | 122 | `logits` | `{1, vocab_size}` | LM head output |

#### E. Total Tensor::create() calls per decode step

| Model | Layers | Per-layer | Outside-loop | **Total** |
|-------|--------|-----------|-------------|-----------|
| **Qwen-1.5B** (Qwen3, 28 layers, has_qk_norm=true) | 28 | 18 | 5 | **509** |
| **Qwen3-8B** (Qwen3, 36 layers, has_qk_norm=true) | 36 | 18 | 5 | **653** |
| Qwen2 variant (no qk_norm) | L | 16 | 5 | 16L + 5 |

Each of these 509/653 calls goes through `BestFitMemoryPool::allocate()` and then
`BestFitMemoryPool::deallocate()` when the tensor goes out of scope -- over **1000
pool operations per decode step**.

---

### 1.2 DecodeScratch Struct Design

The key insight: when N=1, every intermediate tensor has a **fixed, known shape** that
never changes between decode steps. We can pre-allocate all buffers once at engine
initialization and reuse them across every decode step.

```cpp
// include/frontend/models/decode_scratch.hpp

namespace zedinfer::model {

/**
 * Pre-allocated scratch buffers for single-token decode (N=1).
 * Eliminates all Tensor::create() / BestFitMemoryPool traffic per decode step.
 * Allocated once at engine init, reused across all decode steps.
 */
struct DecodeScratch {
    // --- Inputs (updated each step via memcpy) ---
    tensor_t ids;       // {1}           I32
    tensor_t pos_ids;   // {1}           I64

    // --- Embedding output ---
    tensor_t hidden;    // {1, H}        (also reused as layer output)

    // --- Per-layer reusable buffers ---
    // These are reused across layers (layer L writes, layer L+1 overwrites).
    tensor_t normed;        // {1, H}        input layernorm output
    tensor_t q;             // {1, H}        Q projection
    tensor_t k;             // {1, kv_dim}   K projection
    tensor_t v;             // {1, kv_dim}   V projection
    tensor_t q_normed;      // {nhead, d}    Q per-head norm (Qwen3 only, nullptr otherwise)
    tensor_t k_normed;      // {nkvhead, d}  K per-head norm (Qwen3 only, nullptr otherwise)
    tensor_t q_rope;        // {1, nhead, d} Q after RoPE
    tensor_t k_rope;        // {1, nkvhead, d} K after RoPE
    tensor_t attn_out;      // {1, nhead, d} Attention output
    tensor_t o;             // {1, H}        O projection
    tensor_t h1;            // {1, H}        Residual connection
    tensor_t normed_post;   // {1, H}        Post-attention layernorm
    tensor_t gate;          // {1, inter}    MLP gate projection
    tensor_t up;            // {1, inter}    MLP up projection
    tensor_t act;           // {1, inter}    SwiGLU activation
    tensor_t down;          // {1, H}        MLP down projection
    tensor_t hidden_out;    // {1, H}        Layer output (new hidden)

    // --- Output head ---
    tensor_t final_normed;  // {1, H}        Final layernorm
    tensor_t logits;        // {1, V}        LM head output

    // Factory: allocate all buffers based on model config
    static std::unique_ptr<DecodeScratch> create(
        const ModelConfig &cfg,
        bool has_qk_norm,
        const ExecutorConfig &exec_config);
};

} // namespace zedinfer::model
```

**Buffer reuse across layers**: Buffers like `normed`, `q`, `k`, etc. are overwritten
each layer. We do NOT need per-layer copies because within a single layer, the
computation is sequential -- each buffer is consumed before it is overwritten for the
next layer.

The two "hidden" buffers (`hidden` and `hidden_out`) ping-pong: layer L reads from
`hidden` and writes to `hidden_out`; then for layer L+1, we swap pointers (or use
`hidden_out` as input, writing to `hidden`). This avoids aliasing issues where a kernel
reads and writes to the same buffer.

---

### 1.3 Buffer Sizing per Model

Element size for BF16 = 2 bytes.

#### Qwen-1.5B (DeepSeek-R1-0528-Qwen3-0.6B uses same arch; actual 1.5B params)

```
hidden_size (H) = 1536
num_attention_heads (nhead) = 12
num_key_value_heads (nkvhead) = 2
head_dim (d) = H / nhead = 128
kv_dim = nkvhead * d = 256
intermediate_size (inter) = 8960
vocab_size (V) = 151936
num_hidden_layers = 28
```

| Buffer | Shape (N=1) | Elements | Bytes (BF16) |
|--------|-------------|----------|-------------|
| `ids` | {1} | 1 | 4 (I32) |
| `pos_ids` | {1} | 1 | 8 (I64) |
| `hidden` | {1, 1536} | 1,536 | 3,072 |
| `normed` | {1, 1536} | 1,536 | 3,072 |
| `q` | {1, 1536} | 1,536 | 3,072 |
| `k` | {1, 256} | 256 | 512 |
| `v` | {1, 256} | 256 | 512 |
| `q_normed` | {12, 128} | 1,536 | 3,072 |
| `k_normed` | {2, 128} | 256 | 512 |
| `q_rope` | {1, 12, 128} | 1,536 | 3,072 |
| `k_rope` | {1, 2, 128} | 256 | 512 |
| `attn_out` | {1, 12, 128} | 1,536 | 3,072 |
| `o` | {1, 1536} | 1,536 | 3,072 |
| `h1` | {1, 1536} | 1,536 | 3,072 |
| `normed_post` | {1, 1536} | 1,536 | 3,072 |
| `gate` | {1, 8960} | 8,960 | 17,920 |
| `up` | {1, 8960} | 8,960 | 17,920 |
| `act` | {1, 8960} | 8,960 | 17,920 |
| `down` | {1, 1536} | 1,536 | 3,072 |
| `hidden_out` | {1, 1536} | 1,536 | 3,072 |
| `final_normed` | {1, 1536} | 1,536 | 3,072 |
| `logits` | {1, 151936} | 151,936 | 303,872 |

**Total: 393,940 bytes = ~385 KB**

The logits buffer dominates (303 KB). All other buffers combined are ~82 KB.

#### Qwen3-8B

```
hidden_size (H) = 4096
num_attention_heads (nhead) = 32
num_key_value_heads (nkvhead) = 8
head_dim (d) = H / nhead = 128
kv_dim = nkvhead * d = 1024
intermediate_size (inter) = 11008
vocab_size (V) = 152064
num_hidden_layers = 36
```

| Buffer | Shape (N=1) | Elements | Bytes (BF16) |
|--------|-------------|----------|-------------|
| `ids` | {1} | 1 | 4 (I32) |
| `pos_ids` | {1} | 1 | 8 (I64) |
| `hidden` | {1, 4096} | 4,096 | 8,192 |
| `normed` | {1, 4096} | 4,096 | 8,192 |
| `q` | {1, 4096} | 4,096 | 8,192 |
| `k` | {1, 1024} | 1,024 | 2,048 |
| `v` | {1, 1024} | 1,024 | 2,048 |
| `q_normed` | {32, 128} | 4,096 | 8,192 |
| `k_normed` | {8, 128} | 1,024 | 2,048 |
| `q_rope` | {1, 32, 128} | 4,096 | 8,192 |
| `k_rope` | {1, 8, 128} | 1,024 | 2,048 |
| `attn_out` | {1, 32, 128} | 4,096 | 8,192 |
| `o` | {1, 4096} | 4,096 | 8,192 |
| `h1` | {1, 4096} | 4,096 | 8,192 |
| `normed_post` | {1, 4096} | 4,096 | 8,192 |
| `gate` | {1, 11008} | 11,008 | 22,016 |
| `up` | {1, 11008} | 11,008 | 22,016 |
| `act` | {1, 11008} | 11,008 | 22,016 |
| `down` | {1, 4096} | 4,096 | 8,192 |
| `hidden_out` | {1, 4096} | 4,096 | 8,192 |
| `final_normed` | {1, 4096} | 4,096 | 8,192 |
| `logits` | {1, 152064} | 152,064 | 304,128 |

**Total: 479,660 bytes = ~469 KB**

---

### 1.4 Total Scratch Memory

| Model | Scratch Size | Model Weight Size | Scratch/Weights |
|-------|-------------|-------------------|-----------------|
| Qwen-1.5B | ~385 KB | ~3 GB (BF16) | 0.01% |
| Qwen3-8B | ~469 KB | ~16 GB (BF16) | 0.003% |

The scratch memory is negligible. It is allocated once and never freed during engine
lifetime.

---

### 1.5 transformer_forward Modification

The current `make` lambda at `transformer_forward.cpp` line 29-32:

```cpp
auto make = [&](std::vector<size_t> shape) {
    return Tensor::create(shape, exec_config.data_type,
                          exec_config.device_type, exec_config.device_id);
};
```

**Proposed change**: The `make` lambda is replaced with a conditional that uses scratch
buffers when N=1 and falls back to `Tensor::create()` when N>1 (prefill).

The approach works as follows:

1. Add an optional `DecodeScratch*` pointer to the function signature (or pass it
   through the `ForwardContext`).

2. When `DecodeScratch*` is non-null and N=1, the forward function uses pre-allocated
   buffers instead of calling `Tensor::create()`.

3. For prefill (N>1), `DecodeScratch*` is null, and the existing `Tensor::create()`
   path is used unchanged.

**Modified transformer_forward sketch**:

```cpp
tensor_t transformer_forward(
    const ModelForwardConfig &model,
    ForwardContext &ctx,
    const ExecutorConfig &exec_config,
    DecodeScratch *scratch = nullptr)   // NEW: optional scratch
{
    const size_t N = static_cast<size_t>(ctx.num_tokens());

    // ... (same dimension computation) ...

    // When scratch is available and N=1, use pre-allocated buffers.
    // When N>1 (prefill), always dynamically allocate.
    const bool use_scratch = (scratch != nullptr && N == 1);

    auto make = [&](std::vector<size_t> shape) {
        // For N>1 (prefill), always create fresh tensors
        return Tensor::create(shape, exec_config.data_type,
                              exec_config.device_type, exec_config.device_id);
    };

    tensor_t ids, pos_ids;
    if (use_scratch) {
        // Reuse pre-allocated input buffers, just overwrite data
        ids = scratch->ids;
        pos_ids = scratch->pos_ids;
        ctx.prepare_inputs_into(ids, pos_ids);  // new method: write into existing tensor
    } else {
        ctx.prepare_inputs(ids, pos_ids, exec_config);
    }

    auto hidden = use_scratch ? scratch->hidden : make({N, hidden_size});
    ops::embedding(hidden, ids, model.W("embed_tokens.weight"));

    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        auto p = model.prefix(L);

        auto normed = use_scratch ? scratch->normed : make({N, hidden_size});
        ops::rms_norm(normed, hidden, model.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        auto q = use_scratch ? scratch->q : make({N, hidden_size});
        ops::linear(q, normed, model.W(p + "self_attn.q_proj.weight"), model.q_bias(p));

        auto k = use_scratch ? scratch->k : make({N, kv_dim});
        ops::linear(k, normed, model.W(p + "self_attn.k_proj.weight"), model.k_bias(p));

        auto v = use_scratch ? scratch->v : make({N, kv_dim});
        ops::linear(v, normed, model.W(p + "self_attn.v_proj.weight"), model.v_bias(p));

        // ... (same pattern for all subsequent buffers) ...

        // Ping-pong hidden <-> hidden_out to avoid read-write aliasing
        if (use_scratch) {
            // Even layers: read hidden, write hidden_out
            // Odd layers:  read hidden_out, write hidden
            // (Or simply swap the two tensor_t references each iteration)
            std::swap(hidden, scratch->hidden_out);
        } else {
            hidden = make({N, hidden_size});
            ops::add(hidden, h1, down);
        }
    }

    // ... (output head uses scratch->final_normed, scratch->logits) ...
}
```

**For `attend()` in `paged_forward_context.cpp`**: The `attn` tensor allocated at
line 241 is replaced by `scratch->attn_out` when `use_scratch` is true. This requires
either passing the scratch pointer through `attend()` or having `attend()` accept a
pre-allocated output tensor.

The cleanest approach is to add an optional `tensor_t attn_out` parameter to
`ForwardContext::attend()`:

```cpp
virtual tensor_t attend(int layer, tensor_t q_rope, float scale,
                        const ExecutorConfig &exec_config,
                        size_t nhead, size_t nkvhead, size_t head_dim,
                        tensor_t pre_alloc_out = nullptr) = 0;  // NEW
```

When `pre_alloc_out` is non-null, `attend()` writes into it instead of allocating a
new tensor.

**For `prepare_inputs()`**: Add a new method `prepare_inputs_into()` that writes
token IDs and position IDs into existing tensors via `memcpy_sync` or
`memcpy_async`, instead of allocating new tensors.

```cpp
void PagedForwardContext::prepare_inputs_into(tensor_t ids, tensor_t pos_ids) {
    // Write token_ids_ into ids->data() via H2D memcpy
    // Write position_ids_ into pos_ids->data() via H2D memcpy
    // No allocation needed
}
```

---

### 1.6 Files Affected (Phase 1)

| File | Change |
|------|--------|
| `include/frontend/models/decode_scratch.hpp` | **NEW**: `DecodeScratch` struct + `create()` factory |
| `src/frontend/models/decode_scratch.cpp` | **NEW**: `DecodeScratch::create()` implementation |
| `include/frontend/models/forward_config.hpp` | Add `DecodeScratch*` param to `transformer_forward()` declaration |
| `src/frontend/models/transformer_forward.cpp` | Conditional scratch vs make() for all tensor allocations |
| `include/frontend/models/forward_context.hpp` | Add `prepare_inputs_into()` and optional `pre_alloc_out` to `attend()` |
| `include/frontend/models/paged_forward_context.hpp` | Override new methods |
| `src/frontend/models/paged_forward_context.cpp` | Implement `prepare_inputs_into()`, use `pre_alloc_out` in `attend()` |
| `include/zedinfer/engine.hpp` | Add `DecodeScratch` member |
| `src/zedinfer/engine.cpp` | Initialize `DecodeScratch` at engine creation |
| `src/zedinfer/profiler.cpp` | Pass scratch to decode loop |
| `src/zedinfer/serving_loop.cpp` | Pass scratch to decode steps |
| `CMakeLists.txt` (or relevant build file) | Add new source files |

---

## Phase 2: CUDA Graph Capture

### 2.1 What Can vs Cannot Be Captured

CUDA graph capture records a stream of GPU operations (kernels, memcpy, memset) and
replays them as a single unit with minimal CPU overhead. The constraints are:

**CAN be captured:**
- CUDA kernel launches with fixed grid/block dimensions
- cuBLASLt matmul calls (cuBLAS is CUDA-graph-aware since CUDA 11.x)
- Device-to-device memcpy with fixed addresses
- Operations that read from fixed device-memory addresses

**CANNOT be captured (must stay outside the graph):**
- `cudaMalloc` / `cudaFree` (blocked during capture)
- Host-to-device or device-to-host memcpy (requires CPU participation)
- Any CPU-side branching that affects GPU work
- Operations with addresses that change between replays (unless using graph update APIs)
- cuBLASLt descriptor create/destroy (these are CPU-side resource management calls)

**Mapping to zedinfer decode step:**

| Operation | Capturable? | Notes |
|-----------|-------------|-------|
| `ops::embedding` | Yes | Fixed input/output addresses (with scratch) |
| `ops::rms_norm` | Yes | Fixed addresses |
| `ops::linear` (cuBLASLt) | Partially | Kernel itself yes; descriptor create/destroy must be hoisted out |
| `ops::rope` | Yes | Fixed addresses |
| `write_kv` (KV scatter) | **No** | Dynamic addresses from block table (see 2.2) |
| `ops::attention` | **Partially** | `seq_len` changes per step (see 2.3) |
| `ops::swiglu` | Yes | Fixed addresses |
| `ops::add` | Yes | Fixed addresses |
| `ops::argmax` | Yes | Fixed addresses (with scratch) |
| D2H memcpy (argmax result) | **No** | Sync point (see 2.4) |

### 2.2 The KV Scatter Problem

**Current implementation** (`paged_forward_context.cpp` lines 102-150):

For GPU decode, `write_kv()` collects `(src, dst)` pairs for each token's K and V
data, then executes `cudaMemcpyAsync(D2D)` for each pair. For a single-token decode:
- 2 `cudaMemcpyAsync` calls per layer (1 for K, 1 for V)
- Destination addresses are computed from `block_table`:
  ```
  dst = pool_base + block_table[block_idx] * block_size * token_bytes + offset * token_bytes
  ```
- `block_idx` and `offset` depend on `past_len`, which increments each step
- When `past_len` crosses a block boundary, a new block is allocated and
  `block_table[block_idx]` points to a new physical block -- the destination
  address jumps discontinuously

**Why this breaks CUDA graph capture:**
- `cudaMemcpyAsync` addresses are baked into the graph at capture time
- The destination address changes every step (different offset within block, or
  different block entirely)

**Proposed solution: Replace scatter memcpy with a custom CUDA kernel**

Instead of calling `cudaMemcpyAsync(src, dst, size, D2D)` per token, implement a
small scatter kernel:

```cuda
// kv_scatter_kernel.cu
__global__ void kv_scatter_decode_kernel(
    const void *__restrict__ k_src,      // scratch->k_rope->data()   (fixed address)
    const void *__restrict__ v_src,      // scratch->v->data()        (fixed address)
    void *__restrict__ pool_base,         // block pool base           (fixed address)
    const int *__restrict__ k_block_ids,  // device buffer: k block IDs for this layer
    const int *__restrict__ v_block_ids,  // device buffer: v block IDs for this layer
    int past_len,                         // changes per step (kernel argument)
    int block_size,                       // constant
    int token_bytes)                      // constant
{
    // Compute destination: pool_base + block_id * block_size * token_bytes + offset * token_bytes
    int block_idx = past_len / block_size;
    int offset = past_len % block_size;

    // Copy K
    int k_physical_block = k_block_ids[block_idx];
    void *k_dst = (char*)pool_base + (size_t)k_physical_block * block_size * token_bytes
                  + (size_t)offset * token_bytes;
    memcpy(k_dst, k_src, token_bytes);  // device-side memcpy

    // Copy V
    int v_physical_block = v_block_ids[block_idx];
    void *v_dst = (char*)pool_base + (size_t)v_physical_block * block_size * token_bytes
                  + (size_t)offset * token_bytes;
    memcpy(v_dst, v_src, token_bytes);
}
```

**Key insight**: The kernel reads `k_block_ids[block_idx]` and `v_block_ids[block_idx]`
from **device memory** at runtime. The block table is already on the device (or can be
uploaded once and incrementally appended). The `past_len` argument changes per step,
but kernel arguments can be updated between graph replays via
`cudaGraphExecKernelNodeSetParams()`.

This kernel replaces 2 `cudaMemcpyAsync` calls per layer (56 total for Qwen-1.5B)
with 1 kernel launch per layer (28 total), and all kernel addresses are fixed because
they read from device-resident block tables.

**Block table on device**: For single-request decode, the block table is a small array
(e.g., 16 ints for 256 tokens with block_size=16). Upload it once to a pre-allocated
device buffer at the start of the request, and append new block IDs when blocks are
allocated. The device buffer address stays fixed -- only the content changes, which
the kernel reads at runtime.

### 2.3 The Attention Block Table Problem

**Current implementation** (`paged_forward_context.cpp` lines 248-266, single decode):

The paged attention kernel receives these parameters that change per step:
- `params.seq_len = dt->seq_len + 1` -- increments by 1 each step
- `params.k_block_table = dt->k_blocks[layer].data()` -- host pointer to block IDs
- `params.v_block_table = dt->v_blocks[layer].data()` -- host pointer to block IDs

The block table pointers are **host memory** addresses passed as kernel arguments.
The kernel reads them to find where each KV block lives in the pool. When a new block
is allocated (crossing a block boundary), the host vector grows and may reallocate,
changing the pointer.

**Why this is a problem for CUDA graph:**
- Host pointers are baked into captured kernel arguments
- `seq_len` changes every step
- Block table pointers may change when new blocks are allocated

**Proposed solution:**

1. **seq_len**: Pass via a device-memory location. Allocate a small device buffer
   `int *d_seq_len` (4 bytes). Before each graph replay, update it via
   `cudaMemcpyAsync(H2D)` (outside the graph). The attention kernel reads
   `*d_seq_len` instead of taking `seq_len` as a kernel argument.

   Alternatively, since `cudaGraphExecKernelNodeSetParams()` can update kernel
   arguments between replays, we can update `seq_len` as a kernel argument directly.
   However, this requires identifying the specific kernel node in the graph, which is
   more complex.

   **Simpler approach**: Use a device-side `int*` for seq_len. The kernel dereferences
   it at runtime. The pointer itself is fixed (captured in the graph). Only the pointed-to
   value changes (updated via a small H2D copy before graph replay).

2. **Block table**: Move the block table to a pre-allocated device buffer (same buffer
   used by the KV scatter kernel). The attention kernel reads block IDs from device
   memory instead of host memory. The device buffer pointer is fixed; only the content
   changes.

   This requires modifying `AttentionParams` to accept device-resident block tables
   and modifying the paged attention CUDA kernel to read from device memory.

### 2.4 The Argmax D2H Sync Point

**Current implementation** (`sampler.cpp` lines 86-94):

```cpp
ops::argmax(max_idx_dev_, max_val_dev_, last_logits);
core::context().runtime().api()->memcpy_sync(
    max_idx_host_->data(), max_idx_dev_->data(),
    sizeof(int64_t), ZEDINFER_MEMCPY_D2H);
return static_cast<int>(*reinterpret_cast<const int64_t *>(max_idx_host_->data()));
```

The `memcpy_sync(D2H)` is a **hard sync point**: the CPU blocks until the GPU
completes all preceding work, then copies 8 bytes. This must stay outside the CUDA
graph because:
1. The CPU needs the result to decide the next input token
2. D2H sync is inherently a CPU-GPU synchronization

**Proposed graph boundary:**

```
[Outside graph] Update input token + position + seq_len
[Outside graph] H2D copy: token_id -> scratch->ids
[Outside graph] H2D copy: position_id -> scratch->pos_ids
[Outside graph] Update d_seq_len if using device-side seq_len
    |
    v
[CUDA Graph Replay] embedding -> layers (norm, QKV, rope, KV scatter, attention,
                     O proj, residual, MLP) -> final_norm -> lm_head -> argmax
    |
    v
[Outside graph] D2H copy: argmax result
[Outside graph] CPU reads next token
```

The graph captures everything from embedding through argmax. The only operations
outside the graph are:
- Input preparation (3 small H2D copies: token, position, seq_len = ~20 bytes total)
- Output reading (1 D2H copy: 8 bytes)

These outside-graph operations are ~4 API calls total, costing ~20us of CPU overhead.
This is a massive reduction from the current ~500+ API calls.

### 2.5 The cuBLASLt Descriptor Problem

**Current implementation** (`linear_cublas.cu` lines 114-206):

Every `ops::linear()` call creates and destroys cuBLASLt descriptors:
```cpp
cublasLtMatmulDescCreate(&matmulDesc, ...);    // CPU allocation
cublasLtMatrixLayoutCreate(&layoutA, ...);     // CPU allocation
cublasLtMatrixLayoutCreate(&layoutB, ...);     // CPU allocation
cublasLtMatrixLayoutCreate(&layoutC, ...);     // CPU allocation
// ... execute matmul ...
cublasLtMatrixLayoutDestroy(layoutA);          // CPU free
cublasLtMatrixLayoutDestroy(layoutB);          // CPU free
cublasLtMatrixLayoutDestroy(layoutC);          // CPU free
cublasLtMatmulDescDestroy(matmulDesc);         // CPU free
```

These are **CPU-side operations** (not GPU kernel launches). They are not captured by
CUDA graph, but they add CPU overhead: 4 creates + 4 destroys = 8 API calls per linear,
times ~10 linear calls per layer, times 28 layers = ~2240 descriptor management calls
per decode step.

**This is NOT a capture blocker** -- cuBLASLt descriptor management happens on the CPU
and cuBLAS is graph-capture-aware. During capture, cuBLAS records the kernel launch
into the stream graph, and the descriptor management still happens on the CPU side.

However, for Phase 2 optimization, we should **cache descriptors** for the decode path
since all dimensions are fixed for N=1. This eliminates the descriptor create/destroy
overhead entirely:

```cpp
struct CachedMatmulDescriptors {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
};
// Cache keyed by (M=1, N, K, dtype, has_bias) -- same as existing algo_cache key
```

This is a complementary optimization to CUDA graph, not a prerequisite.

### 2.6 CudaGraphRunner Class Design

```cpp
// include/backend/cuda/cuda_graph_runner.hpp

namespace zedinfer::cuda {

/**
 * Manages CUDA graph capture, replay, and invalidation for decode steps.
 *
 * Lifecycle:
 *   1. First decode step: runs normally (no graph), captures the stream
 *   2. Subsequent steps: replays the captured graph
 *   3. If graph is invalidated (e.g., block table grows): re-capture
 */
class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    // Returns true if graph is ready for replay
    bool is_captured() const;

    // Begin capture on the given stream. All subsequent CUDA operations on
    // this stream are recorded into the graph until end_capture().
    void begin_capture(cudaStream_t stream);

    // End capture and compile the graph into an executable.
    void end_capture(cudaStream_t stream);

    // Replay the captured graph on the given stream.
    void replay(cudaStream_t stream);

    // Mark graph as invalid (must re-capture on next step).
    // Called when block table grows or other structural change.
    void invalidate();

    // Stats
    size_t replay_count() const { return replay_count_; }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool captured_ = false;
    size_t replay_count_ = 0;
};

} // namespace zedinfer::cuda
```

### 2.7 Capture / Replay / Invalidation Lifecycle

```
decode_step(token, position, seq_len):
  // --- Outside graph: update inputs ---
  memcpy_async(scratch->ids->data(), &token, 4, H2D)
  memcpy_async(scratch->pos_ids->data(), &position, 8, H2D)
  memcpy_async(d_seq_len, &seq_len, 4, H2D)
  // Update device block table if new block was allocated
  if (new_block_allocated):
      memcpy_async(d_block_table + old_len, &new_block_id, 4, H2D)
      graph_runner.invalidate()  // block table structure changed

  // --- Graph execution ---
  if (!graph_runner.is_captured()):
      graph_runner.begin_capture(stream)
      // Run the full decode forward pass (all kernels recorded)
      transformer_forward(model, ctx, exec_config, scratch)
      ops::argmax(scratch->max_idx_dev, scratch->max_val_dev, scratch->logits)
      graph_runner.end_capture(stream)
  else:
      graph_runner.replay(stream)

  // --- Outside graph: read result ---
  memcpy_sync(host_result, scratch->max_idx_dev->data(), 8, D2H)
  return *host_result
```

**When to invalidate:**
1. Block table grows (new block allocated when `past_len` crosses a block boundary).
   For block_size=16, this happens every 16 tokens. Re-capture cost is amortized
   over 16 replays.
2. Batch size changes (not applicable for single-request decode).

**Re-capture frequency estimate:**
- Block size = 16, typical generation = 256 tokens
- Re-captures: 256 / 16 = 16 times
- Replays between re-captures: 15
- Re-capture overhead: ~same as one normal decode step
- Overall amortized overhead: 1/16 = 6.25% of steps require re-capture

**Optimization**: If the device block table is pre-allocated large enough (e.g., for
max_seq_len), and the attention kernel reads `seq_len` from device memory to know how
many blocks to iterate, then the graph does NOT need invalidation when a new block is
appended. The block table content changes but the kernel reads it at runtime from a
fixed device address. Only the `seq_len` value needs updating, which happens outside
the graph.

With this optimization: **zero re-captures during a single request's decode phase**.
The graph is captured once on the first decode step and replayed for all subsequent
steps.

### 2.8 Updating Variable Inputs Between Replays

Three values change between decode steps:

| Value | Size | Update Method |
|-------|------|---------------|
| `input_token` (int32) | 4 bytes | `cudaMemcpyAsync(H2D)` into `scratch->ids->data()` |
| `position_id` (int64) | 8 bytes | `cudaMemcpyAsync(H2D)` into `scratch->pos_ids->data()` |
| `seq_len` (int32) | 4 bytes | `cudaMemcpyAsync(H2D)` into device `d_seq_len` buffer |

All three updates happen **before** graph replay on the same stream. Since
`cudaMemcpyAsync` is ordered with respect to subsequent operations on the same stream,
the graph replay will see the updated values.

**Block table updates** (new block appended):
- `cudaMemcpyAsync(H2D)` to append the new block ID to the device block table
- Done before graph replay, same stream ordering guarantees correctness
- No graph invalidation needed if the device buffer is large enough

**Total pre-replay CPU cost**: 3-4 `cudaMemcpyAsync` calls (~20us CPU time).

### 2.9 Files Affected (Phase 2)

| File | Change |
|------|--------|
| `include/backend/cuda/cuda_graph_runner.hpp` | **NEW**: `CudaGraphRunner` class |
| `src/backend/cuda/cuda_graph_runner.cu` | **NEW**: Implementation |
| `include/backend/ops/attention_params.hpp` | Add device-resident block table fields + `d_seq_len` |
| `src/backend/ops/attention/nvidia/paged_attention_nvidia.cu` | Read block table from device memory; read `seq_len` from device pointer |
| `src/frontend/models/paged_forward_context.cpp` | Use device block table; pass `d_seq_len` to attention; implement scatter kernel path |
| `include/frontend/models/paged_forward_context.hpp` | Add device block table members |
| `src/backend/ops/linear/nvidia/linear_cublas.cu` | (Optional) Cache descriptors for N=1 shapes |
| `src/zedinfer/profiler.cpp` | Use `CudaGraphRunner` in decode loop |
| `src/zedinfer/serving_loop.cpp` | Use `CudaGraphRunner` in decode path |
| `src/backend/ops/kv_scatter/nvidia/kv_scatter_kernel.cu` | **NEW**: KV scatter CUDA kernel |
| `CMakeLists.txt` | Add new source files, link CUDA runtime for graph APIs |

---

## Expected Impact

### Current Overhead per Decode Step

From nsys profiling data (Qwen-1.5B, B200, BF16), documented in
`docs/notes/optimization_research.md` Section 2.1:

| Overhead Source | Per-step Cost | Evidence |
|----------------|---------------|----------|
| Kernel launch overhead | ~2.1 ms | 425 launches x ~5us avg (`cudaLaunchKernel` + `cuLaunchKernelEx`) |
| cudaMemcpyAsync overhead | ~0.2 ms | 56 calls x ~4us avg |
| Tensor::create/destroy (pool ops) | ~0.3-0.5 ms | 509 alloc + 509 dealloc through BestFitMemoryPool |
| cuBLASLt descriptor mgmt | ~0.1-0.2 ms | ~2240 create/destroy calls |
| **Total CPU-side overhead** | **~2.7-3.0 ms** | |

Current decode latency: ~9.1 ms/token (110 tok/s). CPU overhead is ~30% of total.

### Expected After Phase 1 (DecodeScratch only)

| Change | Savings |
|--------|---------|
| Eliminate 509 `Tensor::create()` + 509 destructor calls | ~0.3-0.5 ms |
| Eliminate pool allocate/deallocate churn | (included above) |

**Expected: 8.6-8.8 ms/token (~114-116 tok/s), 4-6% improvement.**

Phase 1 is modest on its own but is the **prerequisite** for Phase 2 and
independently improves memory pool stability (no fragmentation from decode traffic).

### Expected After Phase 2 (CUDA Graph)

| Change | Savings |
|--------|---------|
| Replace ~425 kernel launches with 1 graph replay | ~2.0 ms |
| Replace 56 cudaMemcpyAsync with scatter kernel (captured in graph) | ~0.2 ms |
| Replace 509 Tensor::create (already eliminated in Phase 1) | (already counted) |
| Overhead: 3-4 pre-replay cudaMemcpyAsync + 1 post D2H sync | +~0.02 ms |

**Expected: 6.3-6.8 ms/token (~147-159 tok/s), total improvement ~30-45% from baseline.**

### Combined with Paged Attention Optimization

If paged attention kernel is also optimized (separate work, see `optimization_research.md`
Section 1), reducing kernel compute time from ~9ms to ~3-4ms:

| State | tok/s | Latency |
|-------|-------|---------|
| Current baseline | ~110 | 9.1 ms |
| + Phase 1 (scratch) | ~115 | 8.7 ms |
| + Phase 2 (CUDA graph) | ~155 | 6.5 ms |
| + Paged attention opt | ~300 | 3.3 ms |
| + Phase 1 + Phase 2 + Paged attn opt | **~400-500** | **2.0-2.5 ms** |

CUDA graph becomes proportionally more impactful when kernel compute time is reduced,
because the fixed CPU overhead becomes a larger fraction of total time.

---

## Implementation Order

```
Phase 1: DecodeScratch Pre-allocation
  Step 1.1: Implement DecodeScratch struct and factory      [1-2 days]
  Step 1.2: Add prepare_inputs_into() to ForwardContext     [0.5 day]
  Step 1.3: Modify transformer_forward() for scratch path   [1-2 days]
  Step 1.4: Modify attend() to accept pre-alloc output      [0.5 day]
  Step 1.5: Wire scratch into engine, profiler, serving     [0.5 day]
  Step 1.6: Test correctness (bit-identical output)         [1 day]
  Step 1.7: Benchmark before/after                          [0.5 day]

Phase 2: CUDA Graph Capture
  Step 2.1: Implement KV scatter CUDA kernel                [1 day]
  Step 2.2: Move block table to device-resident buffer      [1-2 days]
  Step 2.3: Add device-side seq_len to attention params     [0.5 day]
  Step 2.4: Implement CudaGraphRunner                       [1-2 days]
  Step 2.5: Integrate graph runner into decode loop         [1 day]
  Step 2.6: Test correctness (bit-identical output)         [1 day]
  Step 2.7: Benchmark before/after                          [0.5 day]
  Step 2.8: (Optional) Cache cuBLASLt descriptors           [0.5 day]
```

**Phase 1 is independently valuable**: even without CUDA graph, eliminating 1000+ pool
operations per decode step reduces memory fragmentation, improves determinism, and saves
0.3-0.5ms. It also simplifies the code path for decode.

**Phase 2 builds on Phase 1**: CUDA graph capture requires fixed GPU addresses, which
is exactly what DecodeScratch provides. Without Phase 1, Phase 2 is not feasible.

**Both phases are independent of paged attention kernel optimization**: they can be
developed in parallel. The benefits are additive.

---

## Risks

### Phase 1 Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Buffer aliasing bugs**: A kernel reads from a buffer that was already overwritten by the next operation in the same layer. | Medium | Careful audit of data dependencies. The hidden/hidden_out ping-pong pattern prevents the most dangerous case. All other buffers within a layer are consumed before reuse. Add assertions in debug mode to check for aliasing. |
| **Prefill path regression**: Changes to `transformer_forward` accidentally affect the N>1 path. | Low | The `use_scratch` flag cleanly separates the two paths. The `make()` lambda remains unchanged for prefill. Add correctness tests for both prefill and decode. |
| **Model-specific bugs**: Qwen2 (has_qkv_bias=true) and Qwen3 (has_qk_norm=true) take different code paths. Scratch must handle both. | Low | `DecodeScratch::create()` checks `has_qk_norm` to decide whether to allocate `q_normed`/`k_normed`. Test both model families. |
| **shared_ptr lifetime**: Scratch tensors are long-lived. If any code path stores a reference to a scratch tensor beyond the decode step (e.g., in a cache), the data will be overwritten on the next step. | Medium | Audit all call sites. Scratch tensors must NOT be stored in any persistent data structure. The KV cache stores data by copying (scatter), not by reference, so this should be safe. |
| **Memory pool interaction**: Scratch tensors are allocated through the same pool. If the pool is near capacity, the scratch allocation at engine init could fail. | Low | Scratch is tiny (~400-470 KB). Allocate it early, before the KV cache block pool claims the majority of GPU memory. |

### Phase 2 Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **cuBLAS graph capture compatibility**: cuBLASLt must be invoked correctly during graph capture. Some cuBLAS configurations or fallback algorithms may not be graph-capture-safe. | Medium | Test with `CUDA_LAUNCH_BLOCKING=1` disabled. Use `cublasLtMatmulAlgoGetHeuristic` to select algorithms before capture. The existing algorithm cache (`algo_cache` in `linear_cublas.cu` line 82) already ensures a consistent algorithm is used. |
| **Graph capture failure on specific GPU architectures**: CUDA graph capture behavior can differ across GPU generations. | Low | Test on both target platforms (B200 and RTX 4090/3090). CUDA graphs have been stable since CUDA 11.x. |
| **Performance regression from re-capture**: If block table changes trigger re-capture too often, the overhead of re-capture could negate the replay savings. | Medium | With the device-resident block table optimization (Section 2.7), re-capture should be needed only once per request. Even without it, re-capture every 16 steps (block_size=16) amortizes well: 15 fast replays for 1 slow capture. |
| **Debugging difficulty**: CUDA graph errors are harder to diagnose than normal kernel launches. | Medium | Implement a `ZEDINFER_DISABLE_CUDA_GRAPH` environment variable that bypasses graph capture for debugging. Keep the non-graph path functional. |
| **Stream ordering with external operations**: If any code path injects work on a different stream between graph replay and the post-replay D2H copy, the ordering guarantee breaks. | Low | All decode operations currently use the default stream (stream 0). Document this assumption. If multi-stream is introduced later, the graph runner must be updated. |
| **KV scatter kernel correctness**: The custom scatter kernel replaces tested `cudaMemcpyAsync` calls. Off-by-one errors in block/offset calculation could corrupt the KV cache. | Medium | Unit test the scatter kernel against the existing `cudaMemcpyAsync` path. Compare KV cache contents byte-for-byte after each layer. |
