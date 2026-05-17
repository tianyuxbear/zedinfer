#include "backend/core/context/context.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops {

namespace {

__global__ void attn_output_gate_kernel(__nv_bfloat16* __restrict__ attn,
                                          const __nv_bfloat16* __restrict__ g,
                                          int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const float a = __bfloat162float(attn[i]);
    const float gv = __bfloat162float(g[i]);
    // Sigmoid via __expf to keep the fast-math path on the compute stream.
    const float s = 1.0f / (1.0f + __expf(-gv));
    attn[i] = __float2bfloat16(a * s);
}

} // namespace

void attn_output_gate(tensor_t attn, tensor_t g) {
    const int total = static_cast<int>(attn->numel());
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned int>((total + 255) / 256));
    attn_output_gate_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(attn->data()),
        reinterpret_cast<const __nv_bfloat16*>(g->data()), total);
}

} // namespace zedinfer::ops
