#include "backend/core/context/context.hpp"
#include "backend/ops/silu_mul/silu_mul.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops {

namespace {

__global__ void silu_mul_kernel(      __nv_bfloat16* __restrict__ out,
                                const __nv_bfloat16* __restrict__ z,
                                const __nv_bfloat16* __restrict__ x,
                                int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const float zv = __bfloat162float(z[i]);
    const float xv = __bfloat162float(x[i]);
    // silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    const float silu_z = zv / (1.0f + __expf(-zv));
    out[i] = __float2bfloat16(silu_z * xv);
}

} // namespace

void silu_mul(tensor_t out, tensor_t z, tensor_t x) {
    const int total = static_cast<int>(out->numel());
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned int>((total + 255) / 256));
    silu_mul_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<      __nv_bfloat16*>(out->data()),
        reinterpret_cast<const __nv_bfloat16*>(z->data()),
        reinterpret_cast<const __nv_bfloat16*>(x->data()),
        total);
}

} // namespace zedinfer::ops
