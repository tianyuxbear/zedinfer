# CPU Optimization Research & Execution Plan

## 1. Current State: Operator-by-Operator Analysis

### 1.1 linear (GEMM / GEMV) — **Critical, ~85% of inference time**

**Files:**
- `src/backend/ops/linear/cpu/linear_cpu.cpp` — dispatch + BF16/FP16 conversion
- `src/backend/ops/linear/cpu/matmul.cpp` — handwritten AVX-512 blocked GEMM (M>1)
- `src/backend/ops/linear/cpu/vecmul.cpp` — handwritten AVX-512 GEMV (M=1)

**Current implementation:**
- FP32 path: handwritten AVX-512 micro-kernel with 14x32 tile (MR=14, NR=32),
  blocked layout (MC/NC/KC), pack A/B into contiguous blocks, OMP parallelism.
  Uses `_mm512_fmadd_ps` for compute. Static thread count `NTHREADS=24` and
  static global buffers `blockA_packed[MC*KC]`, `blockB_packed[NC*KC]`.
- BF16/FP16 path: allocates `std::vector<float>` temporaries, batch-converts
  entire input + weight to FP32 via `bf16_to_fp32_batch` / `fp16_to_fp32_batch_f16c`,
  then calls the FP32 matmul/vecmul, then converts output back.

**Issues:**
1. Hardcoded `NTHREADS=24` — wrong on machines with different core counts.
2. Static global buffers `blockA_packed` / `blockB_packed` — not thread-safe
   if multiple inference sessions run concurrently; also wastes memory on
   machines with fewer cores.
3. BF16/FP16: full weight conversion on every call (`N*K` elements). For
   Qwen2-1.5B decode, each linear converts a 1536x1536 weight (4.7M FP16->FP32
   conversions) per call. There are ~10 linear calls per layer, 28 layers = ~280
   weight conversions per decode step. This dominates CPU decode time.
4. Only FP32 micro-kernel — no native BF16 AMX / VNNI support.
5. The GEMM tiling parameters (MC/NC/KC) are tuned for a specific machine.

**Optimization potential: VERY HIGH**

---

### 1.2 self_attention — **High, ~10% of inference time**

**File:** `src/backend/ops/self_attention/cpu/self_attention_cpu.cpp`

**Current implementation:**
- Naive per-position, per-head loop with GQA support.
- Allocates `std::vector<float> scores(total_len)` per (token, head) iteration.
- Inner dot product uses `#pragma omp simd` with scalar cast for BF16/FP16.
- OMP parallel over `(seqlen, nhead)` with `collapse(2)`.
- Softmax done per-head sequentially.

**Issues:**
1. `std::vector<float> scores(total_len)` allocated per (i,h) pair inside the
   OMP parallel region — millions of heap allocations for a long-context prefill.
2. No tiling / blocking for cache locality on K/V matrices.
3. Q*K dot product uses scalar `cast<float>` per element for BF16/FP16 — no
   vectorized BF16->FP32 conversion.
4. V aggregation iterates over `dv` in the outer loop and `total_len` in the
   inner loop — suboptimal memory access pattern (V is [total_len, nkvhead, dv],
   inner loop should stride over dv for spatial locality).

**Optimization potential: HIGH**

---

### 1.3 rms_norm — **Medium**

**File:** `src/backend/ops/rms_norm/cpu/rms_norm_cpu.cpp`, `sdot.cpp`

**Current implementation:**
- OMP parallel over seq_len. Per-row: compute `sdot(x, x, hidden_size)` for
  variance, then element-wise `output[j] = weight[j] * input[j] / rms`.
- `sdot` uses AVX-512 FMA (`_mm512_fmadd_ps`, `_mm512_reduce_add_ps`).
- BF16/FP16: allocates 3 `std::vector<float>` per row (input, weight, output),
  batch-converts, computes in FP32, converts back.

**Issues:**
1. BF16/FP16 path allocates 3 vectors per row inside OMP parallel for — heavy
   heap allocation. For Qwen3-8B with hidden_size=4096, that's 3 * 4096 * 4 =
   48 KB per row, allocated and freed for each of the ~100+ rms_norm calls per
   forward pass.
2. Weight is converted BF16->FP32 on every call even though weights don't change.
3. Two passes over data (sdot + normalize) — could be fused into one pass using
   online Welford-style computation.

**Optimization potential: MEDIUM**

---

### 1.4 rope — **Low-Medium**

**File:** `src/backend/ops/rope/cpu/rope_cpu.cpp`

**Current implementation:**
- OMP parallel over seq_len. Per-position: compute sin/cos cache, then apply
  rotation per-head.
- Uses `sincosf` on Linux for fused sin+cos computation.
- `#pragma omp simd` for the rotation loop.

**Issues:**
1. `std::vector<float> cos_cache(half_dim)`, `sin_cache(half_dim)` allocated
   per position inside OMP parallel for.
2. `inv_theta` computation uses `std::pow` per dimension — could precompute once
   at model init.
3. BF16/FP16 uses scalar `cast<float>` per element.

**Optimization potential: LOW-MEDIUM**

---

### 1.5 add — **Low**

**Files:** `src/backend/ops/add/cpu/add_cpu.cpp`, `add_cpu_opt.cpp`

**Current implementation:**
- Already well-optimized: AVX-512 vectorized with 3-tier dispatch
  (tiny/scalar < 64, medium/SIMD < 16384, large/OMP+SIMD).
- Native BF16 handling via shift-convert (no library needed).
- FP16 via F16C intrinsics.

**Issues:** Minor — the parallel threshold and unrolling could be tuned further,
but this op is not a bottleneck.

**Optimization potential: VERY LOW**

---

### 1.6 swiglu — **Low**

**File:** `src/backend/ops/swiglu/cpu/swiglu_cpu.cpp`

**Current implementation:**
- `#pragma omp parallel for simd`. Element-wise: `output = up * gate / (1 + exp(-gate))`.
- BF16/FP16: scalar cast per element.

**Issues:**
1. Division by `(1 + exp(-x))` — could use `1 / (1 + exp(-x)) = sigmoid(x)`
   then `x * sigmoid(x)` = SiLU(x). The compiler may not optimize the division.
2. `std::exp` is slow; could use fast approximation or vectorized exp.

**Optimization potential: LOW**

---

### 1.7 embedding — **Very Low**

**File:** `src/backend/ops/embedding/cpu/embedding_cpu.cpp`

**Current implementation:**
- OMP parallel `memcpy` per row. Already optimal — memory-bound gather.

**Optimization potential: NEGLIGIBLE**

---

### 1.8 argmax — **Very Low**

**File:** `src/backend/ops/argmax/cpu/argmax_cpu.cpp`

**Current implementation:**
- OMP parallel reduction. Already reasonable for a single call per decode step.

**Optimization potential: NEGLIGIBLE**

---

### 1.9 Type conversion utilities

**File:** `src/utils/types.cpp`

**Current implementation:**
- BF16<->FP32: AVX-512 (16 elements/iter) and AVX2 (8 elements/iter) with
  compile-time `#ifdef` dispatch. OMP parallel for batch conversions.
- FP16<->FP32: F16C intrinsics (8 elements/iter).

**Issues:**
1. Compile-time dispatch (`#ifdef __AVX512F__`) — cannot use AVX-512 if compiled
   on AVX2 machine. Runtime dispatch needed (as discussed previously).
2. These conversions are called from linear, rms_norm, and other ops for every
   BF16/FP16 operation. The conversions themselves are fast, but the total
   volume is huge (especially weight conversion in linear).

**Optimization potential: MEDIUM (via eliminating redundant conversions)**

---

## 2. Acceleration Library Options

### 2.1 oneDNN (Intel oneAPI Deep Neural Network Library)

**What it provides:**
- Highly optimized GEMM for FP32, BF16, FP16, INT8 on Intel CPUs
- Native BF16 GEMM (no convert-compute-convert) on CPUs with AVX-512 BF16 or AMX
- Fused post-ops (bias add, ReLU, SiLU can be fused into GEMM)
- Runtime ISA dispatch (AVX2 / AVX-512 / AMX automatically selected)
- Matmul primitive, inner product primitive, layer normalization primitive

**Best for:** `linear` (GEMM/GEMV), potentially `rms_norm`

**License:** Apache 2.0

**Size:** ~50 MB shared library, or static link ~20 MB

**Integration effort:** Medium — need to set up `dnnl::memory`, `dnnl::matmul` primitive.
Primitive creation is expensive (one-time), execution is fast.

**Recommendation: YES — for linear operator. This is the single biggest win.**

---

### 2.2 OpenBLAS

**What it provides:**
- Optimized GEMM (sgemm, dgemm) for FP32/FP64
- CBLAS interface (standard, portable)
- Works on Intel, AMD, ARM

**Best for:** FP32 GEMM

**Limitation:** No native BF16/FP16 GEMM. No fused post-ops.
For BF16/FP16, still need convert-compute-convert.

**License:** BSD-3

**Integration effort:** Very low — just call `cblas_sgemm`.

**Recommendation: POSSIBLE — as a lighter alternative to oneDNN if only FP32 is
needed. But oneDNN is strictly better for this project's needs (BF16 support).**

---

### 2.3 Intel MKL (oneMKL)

**What it provides:**
- Everything OpenBLAS provides, plus BF16 GEMM (`cblas_gemm_bf16bf16f32`)
- JIT code generation for optimal kernel selection
- FFT, sparse BLAS, LAPACK

**Best for:** Same as oneDNN for GEMM

**Limitation:** Intel-only license terms, larger binary, overlaps with oneDNN.

**Recommendation: NO — oneDNN is preferred (open-source, same performance for
inference GEMM, lighter).**

---

### 2.4 Eigen

**What it provides:**
- Header-only C++ template library for linear algebra
- Reasonably optimized GEMM for small-to-medium sizes
- No external dependencies

**Best for:** Prototyping, small matrices

**Limitation:** Slower than oneDNN/MKL/OpenBLAS for large GEMM. No BF16.

**Recommendation: NO — not competitive for LLM-scale GEMM.**

---

### 2.5 XSIMD / Highway / Sleef

**What they provide:**
- XSIMD / Highway: portable SIMD abstraction (write once, run on SSE/AVX/AVX-512/NEON)
- Sleef: vectorized math functions (exp, sin, cos, log) for AVX/AVX-512

**Best for:** Element-wise ops (swiglu, rope, softmax) that need vectorized math

**Limitation:** Don't help with GEMM. Only help with math-heavy element-wise kernels.

**Recommendation:**
- **Sleef: YES** — for `swiglu` (vectorized exp/sigmoid) and `rope` (vectorized sincos).
  Small, header-only option available.
- **Highway: MAYBE** — for portable SIMD wrapper if we need to support non-x86.
  Not critical now since x86 is the only CPU target.
- **XSIMD: NO** — similar to Highway but less mature.

---

### 2.6 Summary: Recommended Libraries

| Library | Target Operators | Priority | Reason |
|---------|-----------------|----------|--------|
| **oneDNN** | linear (GEMM/GEMV) | **P0** | Native BF16 GEMM, 5-10x speedup, runtime ISA dispatch |
| **Sleef** | swiglu, rope | P2 | Vectorized exp/sincos, small integration |
| OpenBLAS | (fallback for non-Intel) | P3 | If oneDNN unavailable on AMD CPUs |

---

## 3. Integration Plan

### 3.1 oneDNN Integration for Linear

**xmake integration:**
```lua
-- xmake.lua or xmake/device/cpu.lua
option("onednn")
    set_default(false)
    set_showmenu(true)
    set_description("Use oneDNN for optimized CPU linear (GEMM)")
option_end()

if has_config("onednn") then
    add_requires("onednn")
    add_defines("USE_ONEDNN")
end

target("ops-cpu")
    -- existing config...
    if has_config("onednn") then
        add_packages("onednn")
    end
```

xmake has built-in onednn package support (`add_requires("onednn")`).
Alternatively, use system-installed oneDNN via `find_package`.

**Code integration pattern:**

```cpp
// src/backend/ops/linear/cpu/linear_cpu.cpp

#ifdef USE_ONEDNN
#include <dnnl.hpp>

// One-time engine/stream creation (at Context init or first call)
static dnnl::engine &get_dnnl_engine() {
    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}
static dnnl::stream &get_dnnl_stream() {
    static dnnl::stream strm(get_dnnl_engine());
    return strm;
}

void linear_onednn(float *output, const float *input, const float *weight,
                   const float *bias, size_t M, size_t N, size_t K) {
    auto &eng = get_dnnl_engine();
    auto &strm = get_dnnl_stream();

    // C[M,N] = A[M,K] * B^T[K,N]  where B stored as [N,K]
    auto a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::ab);
    auto b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::ba); // transposed
    auto c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32,
                                    dnnl::memory::format_tag::ab);

    auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
    auto matmul_prim = dnnl::matmul(matmul_pd);

    auto a_mem = dnnl::memory(a_md, eng, (void*)input);
    auto b_mem = dnnl::memory(b_md, eng, (void*)weight);
    auto c_mem = dnnl::memory(c_md, eng, (void*)output);

    matmul_prim.execute(strm, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    strm.wait();
}
#endif
```

**For BF16 — the key win:**
oneDNN supports `data_type::bf16` natively. On CPUs with AVX-512 BF16 or AMX-BF16,
it computes the GEMM directly in BF16 without conversion. This eliminates the
entire `bf16_to_fp32_batch` + `fp32_to_bf16_batch` cost.

```cpp
// BF16 GEMM with oneDNN — no manual conversion needed
auto a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::bf16, ...);
auto b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::bf16, ...);
auto c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::bf16, ...);
// oneDNN handles BF16 compute internally, using AMX if available
```

**Primitive caching:**
oneDNN primitive creation is expensive (~1ms). For LLM inference, the same shapes
repeat every forward pass. Cache primitives by (M, N, K, dtype) key:

```cpp
struct MatmulKey { size_t M, N, K; dnnl::memory::data_type dt; };
static std::unordered_map<MatmulKey, dnnl::matmul> primitive_cache;
```

### 3.2 Sleef Integration for SwiGLU / RoPE

**xmake integration:**
```lua
option("sleef")
    set_default(false)
    set_showmenu(true)
    set_description("Use Sleef for vectorized math (exp, sincos)")
option_end()

if has_config("sleef") then
    add_requires("sleef")
    add_defines("USE_SLEEF")
end
```

**Code integration for swiglu:**
```cpp
#ifdef USE_SLEEF
#include <sleef.h>

// Vectorized SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// Using Sleef_expf16_u10 for 16-wide AVX-512 exp
for (size_t i = 0; i + 15 < numel; i += 16) {
    __m512 g = _mm512_loadu_ps(gate + i);
    __m512 u = _mm512_loadu_ps(up + i);
    __m512 neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g);
    __m512 exp_neg = Sleef_expf16_u10(neg_g);  // vectorized exp
    __m512 sigmoid = _mm512_div_ps(_mm512_set1_ps(1.0f),
                                    _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg));
    __m512 silu = _mm512_mul_ps(g, sigmoid);
    __m512 result = _mm512_mul_ps(silu, u);
    _mm512_storeu_ps(output + i, result);
}
#endif
```

**Code integration for rope:**
```cpp
#ifdef USE_SLEEF
// Replace sincosf loop with vectorized sincos
for (size_t k = 0; k + 15 < half_dim; k += 16) {
    __m512 angles = _mm512_loadu_ps(angle_buf + k);
    __m512 sin_vec, cos_vec;
    // Sleef_sincosf16_u10 computes both sin and cos in one call
    Sleef___m512_2 sc = Sleef_sincosf16_u10(angles);
    sin_vec = sc.x;
    cos_vec = sc.y;
    _mm512_storeu_ps(sin_cache + k, sin_vec);
    _mm512_storeu_ps(cos_cache + k, cos_vec);
}
#endif
```

---

## 4. Non-Library Optimizations (Code-Level)

These don't require new libraries but would improve performance:

### 4.1 Eliminate per-call heap allocation in BF16/FP16 paths

**Problem:** `linear_cpu.cpp` allocates `std::vector<float>` for the entire weight
matrix on every call. For decode, this is M*K + N*K + M*N floats per linear op.

**Fix:** Pre-allocate conversion buffers at engine init (similar to DecodeScratch)
or use thread-local static buffers.

**Files:** `linear_cpu.cpp`, `rms_norm_cpu.cpp`, `rope_cpu.cpp`

### 4.2 Fix hardcoded NTHREADS=24 in matmul.cpp

**Problem:** `#define NTHREADS 24` and static buffer sizes depend on it.

**Fix:** Query `omp_get_max_threads()` at runtime. Use dynamic allocation for
pack buffers or make them thread-local.

**File:** `matmul.cpp`

### 4.3 Pre-compute and cache RoPE inv_theta

**Problem:** `inv_theta[k] = 1.0 / std::pow(theta, expo)` computed on every
forward call.

**Fix:** Compute once at model init, store as member.

**File:** `rope_cpu.cpp`, model forward code

### 4.4 Reduce allocation in self_attention

**Problem:** `std::vector<float> scores(total_len)` allocated per (token, head).

**Fix:** Pre-allocate per-thread score buffers using thread-local storage.

**File:** `self_attention_cpu.cpp`

---

## 5. Priority-Ordered Execution Plan

### Phase 1: oneDNN for Linear (Highest Impact)

**Expected improvement:** 3-10x for CPU linear, especially BF16.

| Step | Action | File |
|------|--------|------|
| 1 | Add `onednn` option to xmake | `xmake.lua`, `xmake/device/cpu.lua` |
| 2 | Create oneDNN linear wrapper | `src/backend/ops/linear/cpu/linear_onednn.cpp` (new) |
| 3 | Add primitive cache | same file |
| 4 | Modify dispatch in `linear_cpu.cpp` | `src/backend/ops/linear/cpu/linear_cpu.cpp` |
| 5 | Test: operator correctness (FP32, BF16, FP16) | `tests/python/test_linear.py` |
| 6 | Benchmark before/after | `scripts/benchmark.sh` |

### Phase 2: Code-Level Fixes (No New Dependencies)

**Expected improvement:** 20-50% for decode on CPU, mostly from allocation reduction.

| Step | Action | File |
|------|--------|------|
| 1 | Fix NTHREADS=24 hardcode | `matmul.cpp` |
| 2 | Remove per-call std::vector in BF16/FP16 linear | `linear_cpu.cpp` |
| 3 | Remove per-call std::vector in rms_norm | `rms_norm_cpu.cpp` |
| 4 | Remove per-(token,head) score allocation in attention | `self_attention_cpu.cpp` |
| 5 | Pre-compute inv_theta for RoPE | `rope_cpu.cpp`, model forward |
| 6 | Remove per-call std::vector in rope | `rope_cpu.cpp` |

### Phase 3: Sleef for Element-wise Ops (Lower Priority)

**Expected improvement:** 10-30% for swiglu and rope individually (small overall).

| Step | Action | File |
|------|--------|------|
| 1 | Add `sleef` option to xmake | `xmake.lua` |
| 2 | Vectorized exp in swiglu | `swiglu_cpu.cpp` |
| 3 | Vectorized sincos in rope | `rope_cpu.cpp` |

### Phase 4: Runtime ISA Dispatch (Cross-Machine Portability)

**Expected improvement:** Enables single binary for AVX2 + AVX-512 machines.

| Step | Action | File |
|------|--------|------|
| 1 | Add CPU feature detection | `include/utils/cpu_features.hpp` (new), `src/utils/cpu_features.cpp` (new) |
| 2 | Split matmul/vecmul into AVX2 and AVX-512 variants | `matmul_avx2.cpp`, `matmul_avx512.cpp` (new) |
| 3 | Function pointer dispatch in matmul.cpp | `matmul.cpp` |
| 4 | Change `-march=native` to `-mavx2` baseline | `xmake/device/cpu.lua`, `xmake.lua` |
| 5 | AVX-512 files compiled with `-mavx512f -mavx512bw` | `xmake/device/cpu.lua` |

---

## 6. Complete File Inventory

### New Files

| File | Purpose |
|------|---------|
| `src/backend/ops/linear/cpu/linear_onednn.cpp` | oneDNN GEMM wrapper + primitive cache |
| `include/utils/cpu_features.hpp` | Runtime CPU feature detection |
| `src/utils/cpu_features.cpp` | CPUID implementation |
| `src/backend/ops/linear/cpu/matmul_avx2.cpp` | AVX2 matmul variant (Phase 4) |
| `src/backend/ops/linear/cpu/vecmul_avx2.cpp` | AVX2 vecmul variant (Phase 4) |

### Modified Files

| File | Change |
|------|--------|
| `xmake.lua` | Add `onednn`, `sleef` optional deps |
| `xmake/device/cpu.lua` | ISA-specific compile targets, replace `-march=native` |
| `src/backend/ops/linear/cpu/linear_cpu.cpp` | oneDNN dispatch, remove per-call allocation |
| `src/backend/ops/linear/cpu/matmul.cpp` | Fix NTHREADS, dynamic buffers |
| `src/backend/ops/linear/cpu/vecmul.cpp` | Runtime dispatch (Phase 4) |
| `src/backend/ops/self_attention/cpu/self_attention_cpu.cpp` | Thread-local score buffers |
| `src/backend/ops/rms_norm/cpu/rms_norm_cpu.cpp` | Remove per-call vectors |
| `src/backend/ops/rope/cpu/rope_cpu.cpp` | Pre-compute inv_theta, vectorized sincos |
| `src/backend/ops/swiglu/cpu/swiglu_cpu.cpp` | Vectorized exp (Sleef) |
| `src/utils/types.cpp` | Runtime ISA dispatch for bf16/fp16 conversions |

### Unchanged (Already Adequate)

| File | Reason |
|------|--------|
| `src/backend/ops/add/cpu/add_cpu.cpp` | Already well-optimized with AVX-512 |
| `src/backend/ops/add/cpu/add_cpu_opt.cpp` | 3-tier dispatch already good |
| `src/backend/ops/embedding/cpu/embedding_cpu.cpp` | Memory-bound, memcpy-based |
| `src/backend/ops/argmax/cpu/argmax_cpu.cpp` | Called once per step, not a bottleneck |
| `src/backend/ops/rearrange/cpu/rearrange_cpu.cpp` | Rarely called |

---

## 7. Expected Impact Summary

| Optimization | Affected Op | Expected Speedup | Effort |
|-------------|------------|-------------------|--------|
| oneDNN GEMM (FP32) | linear | 2-5x | Medium |
| oneDNN GEMM (BF16 native) | linear | 5-10x | Medium |
| Remove per-call allocation | linear, rms_norm, rope | 1.2-1.5x | Low |
| Fix NTHREADS hardcode | linear (matmul) | correctness fix | Low |
| Thread-local score buffers | self_attention | 1.1-1.3x | Low |
| Sleef vectorized exp | swiglu | 1.2-1.5x (for swiglu) | Low |
| Sleef vectorized sincos | rope | 1.1-1.3x (for rope) | Low |
| Runtime ISA dispatch | all SIMD ops | portability (no perf change) | Medium |

**Overall expected end-to-end CPU inference speedup: 3-8x**, dominated by the
oneDNN linear optimization (linear is ~85% of CPU inference time).
