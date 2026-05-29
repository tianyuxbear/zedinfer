#include "backend/core/context/context.hpp"
#include "backend/ops/shared_expert_gate/shared_expert_gate.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops {

namespace {

// One CTA per token. Each thread strides over the hidden dim.
__global__ void shared_expert_gate_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gate,
                                          int H) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    // All threads in the CTA share the same per-token sigmoid scalar.
    const float g = __bfloat162float(gate[n]);
    const float s = 1.0f / (1.0f + __expf(-g));

    __nv_bfloat16* row = x + (size_t)n * H;
    for (int h = tid; h < H; h += blockDim.x) {
        float v = __bfloat162float(row[h]);
        row[h] = __float2bfloat16(v * s);
    }
}

} // namespace

void shared_expert_gate(tensor_t x, tensor_t gate) {
    if (!x || !gate) {
        throw std::runtime_error("[ops::shared_expert_gate] null input tensor");
    }
    if (x->dtype() != ZEDINFER_DTYPE_BF16 || gate->dtype() != ZEDINFER_DTYPE_BF16) {
        throw std::runtime_error("[ops::shared_expert_gate] only bf16 is supported");
    }
    const auto& xs = x->shape();
    if (xs.size() != 2) {
        throw std::runtime_error("[ops::shared_expert_gate] x must be 2D [N, hidden]");
    }
    const int N = static_cast<int>(xs[0]);
    const int H = static_cast<int>(xs[1]);
    if (gate->numel() != static_cast<size_t>(N)) {
        throw std::runtime_error("[ops::shared_expert_gate] gate.numel must equal N (got "
                                 + std::to_string(gate->numel()) + " vs " + std::to_string(N) + ")");
    }

    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    dim3 grid(static_cast<unsigned>(N));
    dim3 block(256);
    shared_expert_gate_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(x->data()),
                                                          reinterpret_cast<const __nv_bfloat16*>(gate->data()), H);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("shared_expert_gate: ") + cudaGetErrorString(e));
    }
}

} // namespace zedinfer::ops
