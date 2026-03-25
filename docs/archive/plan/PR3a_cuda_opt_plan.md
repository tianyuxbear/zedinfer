# CUDA Optimization Research & Execution Plan

## 1. Current State: Operator-by-Operator Analysis

### 1.1 linear (GEMM / GEMV) — **Critical, ~80% of GPU inference time**

**Files:**
- `src/backend/ops/linear/nvidia/linear_nvidia.cu` — dispatch layer
- `src/backend/ops/linear/nvidia/linear_bf16_kernel.cuh` — BF16 WMMA kernels (128x256, 128x128, 64x128, 64x64, 32x64)
- `src/backend/ops/linear/nvidia/linear_fp16_kernel.cuh` — FP16 WMMA kernel (128x256)
- `src/backend/ops/linear/nvidia/linear_fp32_kernel.cuh` — FP32 kernel (128x128)
- `src/backend/ops/linear/nvidia/matvec.cuh` — GEMV (M=1) with warp reduction + vectorized loads
- `src/backend/ops/linear/nvidia/helper_cuda_ptx.cuh` — PTX inline assembly helpers

**Current implementation:**
- **GEMM (M>1)**: Custom WMMA-based kernels using `nvcuda::wmma` fragments (16x16x16).
  BF16 has 5 tile configurations with auto-tuned dispatch by (M, N, K) ranges.
  FP16 has a single 128x256 configuration. FP32 uses a 128x128 kernel.
  All use manual shared memory tiling, double-buffering, and PTX-level
  load instructions (`ld.global.f32`, `st.shared.v4.f32`).
- **GEMV (M=1)**: Per-row kernel, one block per output element. Uses 128-bit
  vectorized loads (`dot_packed_128b`), warp shuffle reduction, shared memory
  for block reduction.
- Bias is handled as an initial value before accumulation (GEMV) or not fused (GEMM).

**Issues:**
1. Multiple hand-tuned tile configurations maintain complexity but still miss optimal
   shapes. cuBLAS/cuBLASLt auto-tunes for every (M, N, K, dtype) on the target GPU.
2. No epilogue fusion — bias add requires a separate kernel (or pre-initialization)
   for GEMM. cuBLASLt supports fused bias + activation epilogues.
3. FP16 only has one tile config (128x256) — suboptimal for small M (decode).
4. FP32 only has one tile config (128x128) — limited tuning.
5. WMMA API is the "legacy" tensor core API. cuBLAS uses the newer MMA/WGMMA
   instructions on Hopper/Blackwell for higher utilization.
6. The PTX helpers (`helper_cuda_ptx.cuh`) use architecture-specific inline assembly
   that may not be optimal across SM generations.
7. No INT8/INT4 support — needed for future quantization.

**Optimization potential: VERY HIGH**

---

### 1.2 self_attention — **High, ~15% of GPU inference time**

**File:** `src/backend/ops/self_attention/nvidia/self_attention_nvidia.cu`

**Current implementation:**
- Two separate kernels: decode (seqlen=1) and prefill (seqlen>1).
- **Decode kernel**: Grid=(nhead,), Block=(256). Loads Q into shared memory,
  iterates KV cache in tiles of TILE_KV=256. Uses online softmax
  (numerically stable, single pass). Warp-level dot product for Q*K scores,
  block-level reduction for max/sum. V aggregation limited to `tid < dv`
  (only 128 threads active for head_dim=128, wastes 50% of the block).
- **Prefill kernel**: Grid=(seqlen, nhead), Block=(256). Same tiled structure
  as decode but launches one block per (query_position, head). No Q-Q tiling
  across query positions — each block recomputes K attention independently.
  V aggregation has the same 50% utilization problem.
- Both use `__restrict__` pointers.

**Issues:**
1. **V aggregation bottleneck**: Only `dv` threads (128 for head_dim=128) participate
   in the V weighted sum loop (lines 174, 301). The other 128 threads idle.
   This wastes 50% of compute during the memory-bound V phase.
2. **Prefill is O(seqlen * nhead) blocks**: No query tiling. For seqlen=2048 and
   nhead=32, that's 65,536 blocks. Each re-reads all K up to its causal boundary.
   FlashAttention tiles both Q and KV dimensions for better reuse.
3. **No shared memory for V**: K is accessed via shared memory (scores), but V is
   read directly from global memory in the inner loop — no caching.
4. **No FlashAttention-2/3**: Missing IO-aware tiling, multi-pass Q tiling,
   and HBM read optimization. FlashAttention-2 achieves ~2-4x speedup over
   naive attention for prefill.
5. **TILE_KV=256 is fixed**: Not tunable per architecture or sequence length.
6. **No paged attention support**: Needed for future paged KV cache.

**Optimization potential: VERY HIGH**

---

### 1.3 rms_norm — **Low-Medium**

**File:** `src/backend/ops/rms_norm/nvidia/rms_norm_nvidia.cu`

**Current implementation:**
- Three kernel variants: block_reduce, warp_reduce, warp_reduce_packed.
  Dispatch uses `warp_reduce_packed` (the best one) for all dtypes.
- Grid=(seq_len,), Block=(256). Each block processes one row.
- Phase 1: Vectorized 128-bit loads for sum-of-squares. Warp shuffle + shared
  memory reduction. Phase 2: Vectorized normalize-and-store.
- Already handles FP32, FP16, BF16 with type punning for vectorized loads.

**Issues:**
1. Fixed block size 256 — may not be optimal for all hidden sizes.
   For hidden_size=1536 (Qwen2-1.5B), each thread processes ~6 elements.
   For hidden_size=4096 (Qwen3-8B), ~16 elements. Reasonable.
2. Two passes over input data (reduction + normalize). Could fuse into one
   pass with online variance computation, but the current approach has better
   numerical stability.

**Assessment:** Already well-optimized. The vectorized packed kernel with warp+block
reduction is close to optimal. Not a bottleneck.

**Optimization potential: LOW**

---

### 1.4 rope — **Low**

**File:** `src/backend/ops/rope/nvidia/rope_nvidia.cu`

**Current implementation:**
- One thread per (token, head, rotary_pair). Computes angle on-the-fly with `powf`
  and `__sincosf`. Loads two elements, rotates, writes two elements.
- Simple and memory-bound. Compute per element is minimal.

**Issues:**
1. `powf(theta, freq_exponent)` per thread — redundant. Could precompute
   inv_freq table once. But for GPU, the compute is hidden by memory latency.
2. Scalar loads/stores — could use vectorized loads for the pair (both elements
   needed per thread are `head_dim/2` apart, so not contiguous — vectorization
   is hard here).

**Assessment:** Memory-bound, compute is cheap. Not a bottleneck.

**Optimization potential: VERY LOW**

---

### 1.5 add, swiglu, embedding, argmax — **Very Low**

**Files:**
- `add_nvidia.cu` — Vectorized 128-bit load/store, packed arithmetic.
  Already near bandwidth-limited.
- `swiglu_nvidia.cu` — Vectorized with `__expf`, packed load/store.
  Already near bandwidth-limited.
- `embedding_nvidia.cu` — Vectorized lookup. Pure memory copy.
  Already near bandwidth-limited.
- `argmax_nvidia.cu` — Block-scope reduction with atomic. Called once per step.

**Assessment:** All element-wise / memory-bound ops are already well-vectorized
with 128-bit packed load/stores. No library can make them faster — they're
limited by HBM bandwidth, not compute.

**Optimization potential: NEGLIGIBLE**

---

## 2. Acceleration Library Options

### 2.1 cuBLAS / cuBLASLt — **For linear (GEMM/GEMV)**

**What it provides:**
- Industry-standard GEMM library, auto-tuned for every NVIDIA GPU generation.
- cuBLASLt: extended API with matmul descriptor, epilogue fusion (bias, ReLU, GELU),
  layout selection, and workspace-based optimization.
- Native FP32, FP16, BF16, INT8, FP8 GEMM support.
- Uses the latest tensor core instructions automatically (WMMA on Volta/Turing,
  MMA on Ampere, WGMMA on Hopper, Blackwell tensor cores).
- Automatic heuristic-based kernel selection for given (M, N, K, dtype, GPU).
- cuBLAS is part of the CUDA Toolkit — already installed, zero extra dependency.

**Best for:** Replacing all 5 custom GEMM tile configurations + GEMV.

**Integration complexity:** Low — just link `-lcublas -lcublasLt`.

**Expected speedup:**
- GEMM (M>1): 1.5-3x over current WMMA kernels, especially on Hopper/Blackwell
  where cuBLAS uses WGMMA instructions that the current code doesn't.
- GEMV (M=1): 1.0-1.5x — custom GEMV is already reasonable, cuBLAS may or may
  not be faster for M=1 (depends on shape).

**Recommendation: YES — highest priority. Replaces ~1500 lines of kernel code.**

---

### 2.2 FlashAttention — **For self_attention**

**What it provides:**
- IO-aware attention algorithm that tiles both Q and KV dimensions.
- Achieves 2-4x speedup over naive attention for prefill.
- Reduces HBM reads from O(N^2) to O(N) by never materializing the full
  attention matrix.
- FlashAttention-2: improved parallelism across sequence length.
- FlashAttention-3: Hopper-specific optimizations (warp specialization, FP8).

**Integration options:**

| Option | Source | License | Effort |
|--------|--------|---------|--------|
| **flash-attn C++ API** | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3 | Medium — compile from source, call C++ API |
| **cuDNN FlashAttention** | NVIDIA cuDNN 8.9+ | NVIDIA EULA | Low — `cudnnSetAttnDescriptor` + `cudnnMultiHeadAttnForward`. Part of CUDA toolkit. |
| **Custom paged attention** | Write from scratch | N/A | High — needed later for paged KV cache anyway |

**Expected speedup:**
- Prefill: 2-4x for seq_len >= 128.
- Decode: 1.0-1.2x — decode is already single-query, the main bottleneck is
  memory bandwidth for loading K/V cache, which FlashAttention doesn't improve.

**Recommendation:**
- **Short term: cuDNN FlashAttention** — easiest integration, already in CUDA toolkit.
- **Medium term: flash-attn C++ API** — better performance, more features (GQA, ALiBi).
- **Long term: custom paged attention** — needed for paged KV cache (PR-8/9).

---

### 2.3 cuDNN — **For fused operations**

**What it provides beyond attention:**
- Fused operations: matmul + bias + activation, layer norm, etc.
- Graph API (cuDNN 8.7+): build a subgraph of operations and let cuDNN
  fuse and optimize them.
- Could potentially fuse: linear + bias + residual_add, or rms_norm + linear.

**Recommendation: DEFERRED — overkill for current stage. Consider later for
fusion optimization.**

---

### 2.4 CUTLASS — **For custom GEMM / quantized GEMM**

**What it provides:**
- NVIDIA's template-based CUDA GEMM library. Header-only C++.
- Supports FP32, FP16, BF16, INT8, INT4, FP8.
- Highly customizable epilogues (bias, activation, quantization scale).
- Used as the backend for many production inference engines.

**Best for:**
- INT8/INT4 quantized GEMM (future PR-11/12).
- Custom fused epilogues (linear + bias + residual).
- If cuBLAS doesn't provide enough control for specific shapes.

**Recommendation: DEFERRED — use cuBLAS first (simpler). Bring in CUTLASS
when implementing INT4 quantization, which needs custom dequant epilogues.**

---

### 2.5 Triton — **Not applicable**

Triton is a Python-based GPU kernel compiler. Violates the no-Python-in-serving constraint.

---

### 2.6 Summary: Recommended Libraries

| Library | Target Operator | Priority | Reason |
|---------|----------------|----------|--------|
| **cuBLAS/cuBLASLt** | linear (GEMM/GEMV) | **P0** | Biggest compute bottleneck, drop-in replacement, already in CUDA toolkit |
| **FlashAttention** (cuDNN or flash-attn) | self_attention (prefill) | **P1** | 2-4x prefill speedup, critical for long context |
| CUTLASS | linear (INT8/INT4) | P2 (future) | Needed for quantization support |
| cuDNN graph API | fused ops | P3 (future) | Fusion optimization |

---

## 3. Integration Plan

### 3.1 cuBLAS/cuBLASLt for Linear

**xmake integration:**
```lua
-- xmake/device/nvidia.lua (modify ops-nvidia target)
target("ops-nvidia")
    -- existing config...
    add_links("cublas", "cublasLt")  -- already in CUDA toolkit, no extra install
```

cuBLAS is part of the CUDA Toolkit. No `add_requires` needed — just link it.

**Code structure:**

```
src/backend/ops/linear/nvidia/
    linear_nvidia.cu        -- dispatch: cuBLAS path + fallback to custom kernels
    linear_cublas.cu        -- NEW: cuBLAS/cuBLASLt wrapper
    linear_cublas.cuh       -- NEW: header
    linear_bf16_kernel.cuh  -- KEPT: fallback for edge cases
    linear_fp16_kernel.cuh  -- KEPT: fallback
    linear_fp32_kernel.cuh  -- KEPT: fallback
    matvec.cuh              -- KEPT: may still be faster than cuBLAS for M=1
```

**Implementation approach:**

```cpp
// linear_cublas.cu (sketch)
#include <cublas_v2.h>
#include <cublasLt.h>

// Singleton handle (created once, reused)
static cublasLtHandle_t &get_handle() {
    static cublasLtHandle_t handle = []() {
        cublasLtHandle_t h;
        cublasLtCreate(&h);
        return h;
    }();
    return handle;
}

// C[M,N] = A[M,K] * B^T[K,N], B stored as [N,K] row-major
void linear_cublas(void *C, const void *A, const void *B, const void *bias,
                   cudaDataType_t dtype, int M, int N, int K, cudaStream_t stream) {
    auto handle = get_handle();

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set B transposed
    cublasOperation_t transB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &transB, sizeof(transB));

    // Fuse bias if provided
    if (bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                        &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                        &bias, sizeof(bias));
    }

    // Layout descriptors
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, dtype, M, K, K);  // row-major
    cublasLtMatrixLayoutCreate(&layoutB, dtype, N, K, K);  // [N,K] row-major
    cublasLtMatrixLayoutCreate(&layoutC, dtype, M, N, N);  // row-major

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(handle, matmulDesc,
                    &alpha, A, layoutA, B, layoutB,
                    &beta, C, layoutC, C, layoutC,
                    nullptr, nullptr, 0, stream);

    // Cleanup descriptors (handle is persistent)
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
}
```

**Key benefits of cuBLASLt over plain cuBLAS:**
1. **Fused bias epilogue** — eliminates separate bias add kernel
2. **Heuristic-based kernel selection** — auto-tuned for the GPU
3. **Workspace** — cuBLASLt can use extra memory for better algorithms
4. **Layout flexibility** — supports row-major natively (no transpose tricks)

**Dispatch strategy:**
```cpp
void linear(byte *out, byte *in, byte *weight, byte *bias, dtype, M, N, K) {
    // cuBLASLt for GEMM (M > 1) — always better than custom WMMA
    if (M > 1) {
        return linear_cublas(out, in, weight, bias, dtype, M, N, K);
    }
    // Custom GEMV for M=1 — may match or beat cuBLAS for vector-matrix
    // Benchmark to decide: keep custom or switch to cuBLAS
    return matvec_kernel_warp_vec<<<...>>>(out, in, weight, bias, N, K);
}
```

### 3.2 FlashAttention for Self-Attention

**Option A: cuDNN FlashAttention**

cuDNN 8.9+ includes FlashAttention as part of its attention API.
Already available in the CUDA Toolkit.

```lua
-- xmake/device/nvidia.lua
target("ops-nvidia")
    add_links("cudnn")
```

**Option B: flash-attn C++ API**

Build from source and link as a static library.

```lua
-- xmake.lua
option("flash-attn")
    set_default(false)
    set_showmenu(true)
    set_description("Use FlashAttention for optimized GPU attention")
option_end()
```

**Recommendation: Start with cuDNN (simpler, no extra build). Switch to flash-attn
if cuDNN's GQA support or performance is insufficient.**

**Code structure:**
```
src/backend/ops/self_attention/nvidia/
    self_attention_nvidia.cu       -- dispatch: flash path for prefill, custom for decode
    flash_attention_cudnn.cu       -- NEW: cuDNN FlashAttention wrapper
    flash_attention_cudnn.cuh      -- NEW: header
    self_attention_nvidia_decode.cu -- RENAMED: keep custom decode kernel (already good for M=1)
```

**Dispatch strategy:**
```cpp
void self_attention(byte *out, byte *q, byte *k, byte *v, scale, ...) {
    if (seqlen == 1) {
        // Decode: custom kernel (already optimized, flash doesn't help for M=1)
        return self_attention_decode_kernel<<<...>>>(...);
    } else {
        // Prefill: FlashAttention via cuDNN
        return flash_attention_cudnn(out, q, k, v, scale, ...);
    }
}
```

---

## 4. Code-Level Optimizations (No External Libraries)

### 4.1 Decode attention V aggregation

**Problem:** Only `dv` threads (128) out of 256 participate in V aggregation
(line 174: `if (tid < dv)`). 50% thread waste.

**Fix:** Redistribute V aggregation across all threads. Each thread accumulates
a partial V result for a subset of the dv dimensions, then warp shuffle to
combine. Or increase TILE_KV to amortize the idle threads.

**File:** `self_attention_nvidia.cu`

### 4.2 argmax per-call cudaMalloc

**Problem:** `argmax_nvidia.cu:273-275` calls `cudaMalloc` + `cudaMemcpy` on every
invocation to allocate a temporary result buffer. `cudaMalloc` is expensive (~50us).

**Fix:** Pre-allocate the temporary buffer once (static or member of engine).

**File:** `argmax_nvidia.cu`

### 4.3 GEMV bias handling

**Problem:** The GEMV kernel (`matvec.cuh:53`) reads bias from global memory per thread.
For the common no-bias case, each thread still does `bias ? to_float(bias[bid]) : 0.0f`
branch.

**Fix:** Template on has_bias to eliminate the branch entirely:
```cpp
template <typename T, bool HAS_BIAS>
__global__ void matvec_kernel(...) {
    float bias_val = HAS_BIAS ? to_float(bias[bid]) : 0.0f;
}
```

**File:** `matvec.cuh`

---

## 5. Priority-Ordered Execution Plan

### Phase 1: cuBLAS/cuBLASLt for Linear (Highest Impact)

**Expected improvement:** 1.5-3x for GEMM, especially on Hopper/Blackwell.

| Step | Action | File |
|------|--------|------|
| 1 | Add `-lcublas -lcublasLt` to nvidia.lua | `xmake/device/nvidia.lua` |
| 2 | Create cuBLASLt wrapper with handle caching | `src/backend/ops/linear/nvidia/linear_cublas.cu` (new) |
| 3 | Create header | `include/backend/ops/linear/nvidia/linear_cublas.cuh` (new) |
| 4 | Modify dispatch: cuBLAS for M>1, keep custom GEMV for M=1 | `src/backend/ops/linear/nvidia/linear_nvidia.cu` |
| 5 | Benchmark: compare cuBLAS vs custom for M=1 shapes | Decide whether to replace GEMV too |
| 6 | Test: operator correctness across all dtypes | `tests/python/test_linear.py --device nvidia` |

### Phase 2: FlashAttention for Prefill (High Impact)

**Expected improvement:** 2-4x for prefill, no change for decode.

| Step | Action | File |
|------|--------|------|
| 1 | Add `-lcudnn` to nvidia.lua | `xmake/device/nvidia.lua` |
| 2 | Create cuDNN FlashAttention wrapper | `src/backend/ops/self_attention/nvidia/flash_attention_cudnn.cu` (new) |
| 3 | Modify dispatch: flash for prefill, custom for decode | `self_attention_nvidia.cu` |
| 4 | Test: attention correctness against reference | `tests/python/test_attention.py --device nvidia` |

### Phase 3: Code-Level Fixes (Low-Hanging Fruit)

| Step | Action | File |
|------|--------|------|
| 1 | Fix argmax per-call cudaMalloc | `argmax_nvidia.cu` |
| 2 | Template GEMV on has_bias | `matvec.cuh` |
| 3 | Improve decode attention V utilization | `self_attention_nvidia.cu` |

### Phase 4: CUTLASS for Quantized GEMM (Future)

| Step | Action | File |
|------|--------|------|
| 1 | Integrate CUTLASS headers | `third_party/` or xmake package |
| 2 | INT8 GEMM kernel | `linear_int8_nvidia.cu` (new) |
| 3 | INT4 dequant + GEMM | `linear_int4_nvidia.cu` (new) |

---

## 6. Complete File Inventory

### New Files

| File | Purpose |
|------|---------|
| `src/backend/ops/linear/nvidia/linear_cublas.cu` | cuBLASLt GEMM wrapper |
| `include/backend/ops/linear/nvidia/linear_cublas.cuh` | cuBLASLt header |
| `src/backend/ops/self_attention/nvidia/flash_attention_cudnn.cu` | cuDNN FlashAttention wrapper (Phase 2) |
| `include/backend/ops/self_attention/nvidia/flash_attention_cudnn.cuh` | cuDNN FlashAttention header (Phase 2) |

### Modified Files

| File | Change |
|------|--------|
| `xmake/device/nvidia.lua` | Add `add_links("cublas", "cublasLt")`, optionally `add_links("cudnn")` |
| `src/backend/ops/linear/nvidia/linear_nvidia.cu` | cuBLAS dispatch for M>1 |
| `src/backend/ops/self_attention/nvidia/self_attention_nvidia.cu` | FlashAttention dispatch for prefill (Phase 2) |
| `src/backend/ops/argmax/nvidia/argmax_nvidia.cu` | Remove per-call cudaMalloc (Phase 3) |
| `src/backend/ops/linear/nvidia/matvec.cuh` | Template on has_bias (Phase 3) |

### Kept Unchanged (Already Adequate)

| File | Reason |
|------|--------|
| `rms_norm_nvidia.cu` | Already well-optimized (vectorized, warp+block reduction) |
| `rope_nvidia.cu` | Memory-bound, compute trivial |
| `add_nvidia.cu` | Bandwidth-limited, already vectorized |
| `swiglu_nvidia.cu` | Bandwidth-limited, already vectorized |
| `embedding_nvidia.cu` | Pure memory copy, already vectorized |

---

## 7. Expected Impact Summary

| Optimization | Affected Op | Expected Speedup | Effort |
|-------------|------------|-------------------|--------|
| cuBLAS/cuBLASLt GEMM | linear (M>1) | 1.5-3x | Low |
| cuBLAS GEMV (if faster) | linear (M=1) | 1.0-1.5x | Low |
| cuBLASLt fused bias | linear | ~5% (eliminates bias kernel) | Low |
| FlashAttention prefill | self_attention | 2-4x for prefill | Medium |
| Fix argmax cudaMalloc | argmax | ~50us/call saved | Very Low |
| Decode V utilization | self_attention decode | 1.1-1.3x | Low |
| CUTLASS INT8/INT4 | linear (quantized) | N/A (enables quantization) | Medium-High |

**Overall expected end-to-end GPU inference speedup:**
- **Decode (single token):** 1.3-2x (cuBLAS GEMV improvement + argmax fix)
- **Prefill (prompt processing):** 2-4x (cuBLAS GEMM + FlashAttention)
- **Long context (2K+ tokens):** 3-5x (FlashAttention dominates)

---

## 8. Key Architectural Note

Unlike the CPU side (where oneDNN replaces everything), on GPU the situation is more
nuanced:

| Compute Pattern | Best Solution | Reason |
|----------------|--------------|--------|
| GEMM (large M) | cuBLAS/cuBLASLt | Auto-tuned, uses latest tensor core instructions |
| GEMV (M=1) | Custom kernel OR cuBLAS | Custom may win due to less overhead; benchmark needed |
| Prefill attention | FlashAttention | IO-aware tiling is algorithmic, not just tuning |
| Decode attention | Custom kernel | Single-query attention is simple; flash doesn't help |
| Element-wise (add, swiglu, rope, rms_norm, embedding) | Custom kernels | Already bandwidth-limited; no library can beat HBM bandwidth |
| Argmax | Custom kernel | Trivial reduction; just fix the cudaMalloc issue |

The GPU optimization is a **two-library** story: cuBLAS for linear + FlashAttention
for prefill attention. Everything else stays custom.
