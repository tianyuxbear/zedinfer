#include "backend/ops/mamba/qk_l2norm.hpp"

#include "backend/core/context/context.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops::mamba {

namespace {

// One CTA per (n, head). 32 threads (1 warp) cooperate over `head_dim`
// elements. Each lane handles `head_dim / 32` strided elements. Works for
// any `head_dim` that's a multiple of 32 (Qwen3.5 uses 128).
__global__ void qk_l2norm_kernel(__nv_bfloat16* __restrict__ qk, int head_dim, float scale, float eps) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;

    __nv_bfloat16* row_ptr = qk + (size_t)row * head_dim;

    // Phase 1: sum of squares, warp-reduced.
    float sq = 0.f;
    for (int j = lane; j < head_dim; j += 32) {
        float v = __bfloat162float(row_ptr[j]);
        sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) { sq += __shfl_xor_sync(0xffffffff, sq, off); }

    const float norm = rsqrtf(sq + eps) * scale;

    // Phase 2: write normalized row.
    for (int j = lane; j < head_dim; j += 32) {
        float v = __bfloat162float(row_ptr[j]);
        row_ptr[j] = __float2bfloat16(v * norm);
    }
}

} // namespace

void qk_l2norm_inplace(tensor_t qk, int num_heads, int head_dim, float scale, float eps) {
    if (qk->dtype() != ZEDINFER_DTYPE_BF16) {
        throw std::runtime_error("[qk_l2norm_inplace] only bf16 is supported");
    }
    if (head_dim <= 0 || (head_dim % 32) != 0) {
        throw std::runtime_error("[qk_l2norm_inplace] head_dim must be a positive multiple of 32");
    }

    const size_t total = qk->numel();
    const size_t row_count = total / static_cast<size_t>(head_dim);
    if (row_count == 0) {
        return;
    }

    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    dim3 grid(static_cast<unsigned>(row_count));
    dim3 block(32);
    qk_l2norm_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qk->data()), head_dim, scale, eps);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("qk_l2norm_inplace: ") + cudaGetErrorString(e));
    }
    (void)num_heads; // num_heads is implicit in row_count = N * num_heads
}

} // namespace zedinfer::ops::mamba
