#include "backend/core/context/context.hpp"
#include "backend/ops/mamba/causal_conv1d.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace zedinfer::ops::mamba {

namespace {

// One thread per output channel d. Each thread:
//   1. Loads K-1 state values + K weight taps into registers (depthwise: one
//      filter per channel — weights[d, 0, 0..K-1] is a contiguous K-tap row).
//   2. Iterates over N input tokens. For each token n it appends x[n, d] to
//      the window tail, computes y = silu(sum_i w_i * window_i), and slides
//      the window left.
//   3. After processing all tokens, writes the trailing K-1 window values
//      back to the state slot — this becomes the next call's history.
//
// Float accumulation matches the CPU reference impl so the GPU path stays
// numerically aligned with the BF16-truncate-to-store CPU baseline.
template <int K>
__global__ void causal_conv1d_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ out,
                                     __nv_bfloat16* __restrict__ state,
                                     int N, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    // Register window holds (K-1) state taps plus the current input slot.
    float window[K];
#pragma unroll
    for (int i = 0; i < K - 1; ++i) {
        window[i] = __bfloat162float(state[i * D + d]);
    }

    // Per-channel depthwise weight tap row.
    float wv[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        wv[i] = __bfloat162float(w[d * K + i]);
    }

    for (int n = 0; n < N; ++n) {
        window[K - 1] = __bfloat162float(x[n * D + d]);
        float acc     = 0.0f;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            acc += window[i] * wv[i];
        }
        // SiLU: acc * sigmoid(acc). __expf is the fast-math intrinsic; the
        // small dynamic range error is dominated by BF16 store quantization.
        const float s     = 1.0f / (1.0f + __expf(-acc));
        out[n * D + d]    = __float2bfloat16(acc * s);

        // Slide window left so window[K-1] is free for the next input.
#pragma unroll
        for (int i = 0; i < K - 1; ++i) {
            window[i] = window[i + 1];
        }
    }

    // After the loop window[0..K-2] are the most recent K-1 inputs in time
    // order — exactly the new persistent state.
#pragma unroll
    for (int i = 0; i < K - 1; ++i) {
        state[i * D + d] = __float2bfloat16(window[i]);
    }
}

} // namespace

void causal_conv1d(tensor_t out, tensor_t x, tensor_t weight,
                   model::SSMStateView v, int slot_idx, int layer_idx) {
    if (!out || !x || !weight) {
        throw std::runtime_error("ops::mamba::causal_conv1d: null tensor input");
    }
    if (v.conv_base == nullptr) {
        throw std::runtime_error("ops::mamba::causal_conv1d: conv state base is null");
    }
    const int N = static_cast<int>(x->shape()[0]);
    const int D = static_cast<int>(x->shape()[1]);
    const int K = v.conv_kernel_dim;
    if (K != 4) {
        throw std::runtime_error("ops::mamba::causal_conv1d: only K=4 is implemented (Qwen3.5)");
    }
    if (D != v.qkv_dim) {
        throw std::runtime_error("ops::mamba::causal_conv1d: x channel dim must match conv state qkv_dim");
    }
    if (N <= 0) {
        return; // No tokens this step; state stays untouched.
    }

    auto* x_ptr   = reinterpret_cast<const __nv_bfloat16*>(x->data());
    auto* w_ptr   = reinterpret_cast<const __nv_bfloat16*>(weight->data());
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out->data());

    // Conv state is stored row-major [slots, layers, K-1, qkv_dim] by
    // SSMStatePool; stride fields are bytes (see SSMStateView contract).
    auto* state_layer = reinterpret_cast<char*>(v.conv_base)
                       + static_cast<int64_t>(slot_idx)  * v.conv_stride_slot
                       + static_cast<int64_t>(layer_idx) * v.conv_stride_layer;
    auto* state_ptr   = reinterpret_cast<__nv_bfloat16*>(state_layer);

    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    constexpr int kBlock = 128;
    const dim3 block(kBlock);
    const dim3 grid(static_cast<unsigned int>((D + kBlock - 1) / kBlock));
    causal_conv1d_kernel<4><<<grid, block, 0, stream>>>(x_ptr, w_ptr, out_ptr, state_ptr, N, D);

    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("ops::mamba::causal_conv1d: kernel launch failed: ")
                                 + cudaGetErrorString(err));
    }
}

} // namespace zedinfer::ops::mamba
