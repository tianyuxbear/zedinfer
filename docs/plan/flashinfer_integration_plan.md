# FlashInfer Integration Plan

> Replace custom paged attention kernels (decode + prefill) with FlashInfer's optimized kernels.
> Covers: single decode, batched decode, single prefill, batched prefill.

---

## 1. Why FlashInfer

Current paged attention decode kernel: **129us/call** (5x slower than contiguous).
Root causes:
- Dependent load chain (block_table → address → K/V data)
- Only 12 CUDA blocks for 12 attention heads (90% of B200 SMs idle)
- No split-K parallelism for long sequences
- Naive prefill (no IO-aware tiling)

FlashInfer provides:
- **Split-K decode** (FlashDecoding): multiple blocks per head → full SM utilization
- **IO-aware tiled prefill** (FlashAttention-2/3): 2-4x faster for long prompts
- **Batched varlen prefill**: one kernel launch for all prefill requests (`cu_seqlens`)
- **Native paged KV support**: block_table + CSR indexing
- Apache 2.0 license, supports sm_80 through sm_120+

Expected decode improvement: 129us → ~25-30us (4-5x), matching contiguous performance.

---

## 2. Key Finding: JIT Compilation

FlashInfer (v0.2+) uses **JIT compilation** — kernels are CUDA/CUTLASS templates instantiated at runtime via their `init()` API, not at compile time. This means:

- **NOT** pure header-only `#include` (despite the C++ headers existing)
- The Wrapper classes (`BatchDecodeWithPagedKVCacheWrapper` etc.) compile specialized kernels on first use
- Compiled kernels are cached for reuse

**Implication for zedinfer:** We have three integration strategies (see Section 5).

---

## 3. KV Layout Compatibility

### FlashInfer's Expected Layout

FlashInfer accepts paged KV cache in two forms:

**Form A: Unified 5D tensor**
```
paged_kv_cache: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
                                 ^
                                K=0, V=1
```

**Form B: Separate K/V tensors (tuple)**
```
k_cache: [max_num_pages, page_size, num_kv_heads, head_dim]   (NHD layout)
v_cache: [max_num_pages, page_size, num_kv_heads, head_dim]
```

Both forms use **shared page indices** — one `page_indices` array indexes both K and V.

### Current zedinfer Layout

```
Single BlockPool: [num_blocks * block_size, num_kv_heads, head_dim]
                   ↑ K and V blocks mixed in the same pool

SequenceBlockTable:
  k_blocks[layer][i] = physical_block_id_for_K    // separate from V
  v_blocks[layer][i] = physical_block_id_for_V    // different ID!
```

**Incompatible:** FlashInfer requires K and V to share the same page indices.

### Required Change: Separate K/V Pools

```
Before:
  pool: [total_blocks, block_size, nkvhead, head_dim]  (mixed K/V)
  k_blocks[layer][i] ≠ v_blocks[layer][i]

After:
  k_pool: [num_pages, page_size, nkvhead, head_dim]
  v_pool: [num_pages, page_size, nkvhead, head_dim]
  pages[layer][i] = page_id  (shared, indexes both k_pool and v_pool)
```

Allocating K/V in pairs with shared page IDs directly maps to FlashInfer's Form B.

---

## 4. CSR Indexing

FlashInfer uses CSR (Compressed Sparse Row) format for batched operations:

```
kv_indptr:        [0, num_pages_req0, num_pages_req0 + num_pages_req1, ...]  (int32)
kv_page_indices:  [page_ids for req0..., page_ids for req1..., ...]          (int32)
kv_last_page_len: [valid_tokens_in_last_page per request]                    (int32)
```

zedinfer's `SequenceBlockTable` has `pages[layer][page_idx]` per layer. Need to flatten this into CSR format per batch.

This conversion happens in `PagedForwardContext::attend()` at batch assembly time — similar to how block tables are currently uploaded to GPU.

---

## 5. Integration Strategies

### Strategy A: Build FlashInfer as a static library

```
1. Git submodule: third_party/flashinfer
2. Build via CMake → libflashinfer.a with pre-instantiated kernels
3. Link from xmake: add_links("flashinfer")
4. Call C++ functions from ops::attention dispatch
```

**Pros:** Clean separation, fast incremental builds
**Cons:** Complex build setup (CMake + xmake interop), need to specify template instantiations upfront

### Strategy B: AOT (Ahead-Of-Time) kernel compilation

FlashInfer supports pre-compiling kernels for specific configurations:

```
AOT compile for: head_dim=128, dtype=bf16, page_size=16, GQA ratios=(6,4,1)
Output: .cu files with concrete kernel functions (no templates)
```

```
1. Run FlashInfer's AOT compiler once (Python script, offline)
2. Output: generated .cu files with concrete kernels
3. Include generated files in zedinfer build (just another .cu)
4. No JIT, no Python at runtime
```

**Pros:** Zero runtime dependency, compiles like any other .cu file, maximum control
**Cons:** Must regenerate when changing config (head_dim, dtype), Python needed for codegen step

### Strategy C: Extract and adapt kernel source

```
1. Copy specific CUDA kernel files from FlashInfer
2. Adapt to zedinfer's API (remove PyTorch/FlashInfer abstractions)
3. Compile as regular .cu files in xmake
```

**Pros:** Maximum control, no external dependency at all
**Cons:** Hard to maintain, need deep understanding of FlashInfer internals, miss future optimizations

### Recommendation: **Strategy B (AOT)**

Reasons:
- No Python runtime dependency (codegen is offline, one-time)
- No complex CMake interop
- Generated .cu files compile as regular CUDA code
- Easy to regenerate for new model configs
- Most practical for a C++ framework without Python in the serving path

---

## 6. Implementation Plan

### Phase 1: KV Pool Refactor (prerequisite)

**Goal:** Change from mixed K/V pool to separate K/V pools with shared page IDs.

**Files affected:**
- `include/backend/kvcache/block_pool.hpp` → rename Block → Page, add KV pool pair
- `src/backend/kvcache/block_pool.cpp` → allocate two pools, paired page allocation
- `include/zedinfer/request.hpp` → simplify SequenceBlockTable (one page list, not separate K/V)
- `src/frontend/models/paged_forward_context.cpp` → adapt write_kv and attend
- `src/zedinfer/scheduler.cpp` → adapt block allocation
- `src/zedinfer/session.cpp` → adapt session block table
- `src/zedinfer/profiler.cpp` → adapt warmup/profile

**Key changes:**

```cpp
// Before:
struct SequenceBlockTable {
    vector<vector<int>> k_blocks;  // [num_layers][blocks_per_layer]
    vector<vector<int>> v_blocks;  // separate
    int seq_len;
};

// After:
struct SequencePageTable {
    vector<vector<int>> pages;     // [num_layers][pages_per_layer] (shared K/V)
    int seq_len;
};

// Before:
class BlockPool {
    void *pool_memory_;  // single allocation, mixed K/V
};

// After:
class PagedKVPool {
    void *k_pool_;  // [num_pages, page_size, nkvhead, head_dim]
    void *v_pool_;  // [num_pages, page_size, nkvhead, head_dim]
    // Same page_id indexes both k_pool_ and v_pool_
};
```

**Validation:** Existing custom paged attention kernels must still work after refactor (adapt `pool_base` + block_table → k_pool/v_pool + page_table).

### Phase 2: AOT Kernel Generation

**Goal:** Generate FlashInfer decode/prefill kernels for zedinfer's target configs.

**Steps:**
1. Install FlashInfer Python package (development machine only, not serving)
2. Write a codegen script: `scripts/generate_flashinfer_kernels.py`
3. Generate kernels for target configurations:
   - head_dim: 128 (Qwen family)
   - dtype: bf16, fp16
   - page_size: 16
   - GQA: nhead/nkvhead ratios for supported models
   - causal: true
   - pos_encoding: NONE (RoPE handled separately in zedinfer)
4. Output: `src/backend/ops/self_attention/nvidia/flashinfer_generated/` directory with .cu files
5. Add to xmake build

**Generated files example:**
```
flashinfer_generated/
  decode_bf16_h128_p16.cu       # single decode kernel
  batch_decode_bf16_h128_p16.cu # batched decode with plan/run
  prefill_bf16_h128_p16.cu      # single prefill
  batch_prefill_bf16_h128_p16.cu# batched prefill with cu_seqlens
  flashinfer_ops.h              # C function declarations
```

### Phase 3: Integration into ops::attention

**Goal:** Route paged attention calls to FlashInfer kernels.

**Changes to `src/backend/ops/self_attention/op.cpp`:**

```cpp
#ifdef USE_FLASHINFER
#include "flashinfer_generated/flashinfer_ops.h"
#endif

void attention(const AttentionParams &params) {
#ifdef USE_FLASHINFER
    if (params.config.device_type == ZEDINFER_DEVICE_NVIDIA) {
        if (params.is_batched()) return flashinfer_batch_decode(params);
        if (params.is_decode())  return flashinfer_single_decode(params);
        return flashinfer_prefill(params);
    }
#endif
    // Fallback to custom kernels (CPU, or GPU without FlashInfer)
    if (params.is_batched()) return dispatch_paged_decode_batched(params);
    if (params.is_decode())  return dispatch_paged_decode(params);
    return dispatch_paged_prefill(params);
}
```

**Wrapper functions** (in `src/backend/ops/self_attention/nvidia/flashinfer_wrapper.cu`):

```cpp
void flashinfer_single_decode(const AttentionParams &p) {
    // Convert zedinfer params → FlashInfer API call
    // - p.pool_base → k_pool, v_pool (from PagedKVPool)
    // - p.k_block_table → page_indices
    // - p.seq_len → last_page_len computation
}

void flashinfer_batch_decode(const AttentionParams &p) {
    // Build CSR format: indptr, page_indices, last_page_lens
    // Call FlashInfer batched decode
}
```

**Changes to `AttentionParams`:**
```cpp
struct AttentionParams {
    // ... existing fields ...

    // For FlashInfer: separate K/V pool pointers
    const void *k_pool_base = nullptr;
    const void *v_pool_base = nullptr;
    // page_table replaces k_block_table/v_block_table
    const int *page_table = nullptr;  // shared K/V page indices
};
```

### Phase 4: Batch Prefill with cu_seqlens

**Goal:** Replace per-request sequential prefill launches with a single batched call.

Currently in `PagedForwardContext::attend()`:
```cpp
// Prefill: per-request sequential (N kernel launches)
for (const auto &slot : slots_) {
    if (slot.is_decode) continue;
    ops::paged_attention_prefill(slot...);  // one launch per request
}
```

With FlashInfer:
```cpp
// Prefill: single batched call with cu_seqlens
flashinfer_batch_prefill(all_prefill_slots, cu_seqlens, page_tables);
```

This requires building `qo_indptr` (query offset CSR) and `kv_indptr` in `PagedForwardContext`.

### Phase 5: Remove Custom GPU Kernels (optional)

Once FlashInfer is validated:
- Remove `paged_attention_decode_kernel` from `paged_attention_nvidia.cu`
- Remove `paged_attention_decode_batched_kernel`
- Remove `paged_attention_prefill_kernel`
- Keep CPU kernels (FlashInfer is GPU-only)
- Keep dispatch logic for CPU fallback

---

## 7. Build System Changes

```lua
-- xmake.lua
option("flashinfer")
    set_default(false)
    set_showmenu(true)
    set_description("Use FlashInfer AOT kernels for optimized paged attention (GPU)")
option_end()

if has_config("flashinfer") then
    add_defines("USE_FLASHINFER")
end
```

```lua
-- xmake/device/nvidia.lua
-- FlashInfer AOT-generated kernels
if has_config("flashinfer") then
    add_files("../../src/backend/ops/self_attention/nvidia/flashinfer_generated/*.cu")
    add_files("../../src/backend/ops/self_attention/nvidia/flashinfer_wrapper.cu")
end
```

**Codegen script** (run offline, not part of build):
```bash
# Generate FlashInfer kernels for zedinfer's model configs
python scripts/generate_flashinfer_kernels.py \
  --head-dims 128 \
  --dtypes bf16 fp16 \
  --page-size 16 \
  --output src/backend/ops/self_attention/nvidia/flashinfer_generated/
```

---

## 8. Testing Strategy

1. **Correctness:** Compare FlashInfer output vs custom kernel output for same input
   - Single decode: same Q, same KV cache → same output (within tolerance)
   - Batched decode: same
   - Prefill: same

2. **Performance:** nsys profile before/after
   - Decode kernel time: expect 4-5x improvement
   - Prefill kernel time: expect 2-4x improvement for long prompts
   - End-to-end bench: expect 2-3x throughput improvement

3. **Regression:** Existing CPU path unaffected (FlashInfer is GPU-only)

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| FlashInfer AOT doesn't support all needed configs | Medium | Verify upfront for head_dim=128, bf16, page_size=16, GQA ratios |
| sm_100 (Blackwell) not yet in FlashInfer | Low | PTX forward-compat works; or pin to latest release with sm_100 |
| KV pool refactor breaks existing paths | High | Phase 1 validates with custom kernels before FlashInfer |
| AOT codegen script requires Python | Low | One-time offline step, not in serving path |
| FlashInfer kernel bugs | Medium | Keep custom kernels as fallback via compile flag |
| Compilation time for generated .cu files | Low | Only compile once; incremental builds fast |

---

## 10. Execution Order

```
Phase 1: KV Pool Refactor           [~2 days, prerequisite]
  - Separate K/V pools
  - Shared page IDs
  - Validate with existing custom kernels

Phase 2: AOT Kernel Generation      [~1 day]
  - Install FlashInfer
  - Write codegen script
  - Generate decode + prefill kernels

Phase 3: Decode Integration          [~1 day]
  - flashinfer_wrapper.cu
  - Wire into ops::attention
  - Benchmark decode improvement

Phase 4: Prefill Integration         [~1 day]
  - Batched prefill with cu_seqlens
  - Benchmark prefill improvement

Phase 5: Cleanup                     [~0.5 day]
  - Remove custom GPU kernels (optional)
  - Update docs
```

Total: ~5-6 days. Phase 1 is the most invasive change (touches many files).

---

## 11. Open Questions

1. **FlashInfer version:** Which release tag to pin? Need one with stable AOT support + sm_100.

2. **AOT vs JIT:** Is AOT codegen mature enough? FlashInfer's primary path is JIT via Python. The AOT path may have less community testing.

3. **page_size flexibility:** If we change block_size from 16 to 32 (better for GPU), do we regenerate? Yes — AOT kernels are specialized per page_size.

4. **RoPE in FlashInfer:** FlashInfer supports fused RoPE (`pos_encoding_mode='ROPE_LLAMA'`). Should we use it or keep separate RoPE kernel? Separate is safer for compatibility; fused is faster.

5. **NHD vs HND layout:** FlashInfer defaults to NHD `[page_size, nkvhead, head_dim]`. zedinfer's current pool layout is `[block_size * nkvhead * head_dim]` which is NHD-compatible. No layout change needed within pages.
