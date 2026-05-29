#include "backend/core/context/context.hpp"
#include "backend/ops/layer_norm/nvidia/layer_norm_bias_nvidia.cuh"
#include "utils/check.hpp"

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

// Per-type FP32 conversion helpers used inside device kernels.
template <typename T> __device__ __forceinline__ float to_f32_dev(T v);
template <> __device__ __forceinline__ float to_f32_dev<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ float to_f32_dev<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_f32_dev<half>(half v) {
    return __half2float(v);
}

template <typename T> __device__ __forceinline__ T from_f32_dev(float v);
template <> __device__ __forceinline__ float from_f32_dev<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ __nv_bfloat16 from_f32_dev<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}
template <> __device__ __forceinline__ half from_f32_dev<half>(float v) {
    return __float2half(v);
}

// One CTA per row, intra-block reduction via shared memory. Each thread strides
// across the row to load values, computes a partial sum / sum-of-squares, then
// the block reduces. Block size is fixed at 256 — works well for hidden_size
// up to a few thousand which covers Qwen3.5-VL's 1152.
template <typename T, int BLOCK = 256>
__global__ void layer_norm_bias_kernel(T* y, const T* x, const T* w, const T* b, int D, float eps) {
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<size_t>(n) * D;
    T*       yr = y + static_cast<size_t>(n) * D;

    // 1. Compute mean.
    float local_sum = 0.0f;
    for (int j = tid; j < D; j += BLOCK) {
        local_sum += to_f32_dev<T>(xr[j]);
    }
    __shared__ float ssum;
    typedef cub::BlockReduce<float, BLOCK>     BR;
    __shared__ typename BR::TempStorage         tmp;
    float                                       total = BR(tmp).Sum(local_sum);
    if (tid == 0) {
        ssum = total;
    }
    __syncthreads();
    const float mean = ssum / static_cast<float>(D);

    // 2. Compute variance.
    float local_sq = 0.0f;
    for (int j = tid; j < D; j += BLOCK) {
        const float d = to_f32_dev<T>(xr[j]) - mean;
        local_sq += d * d;
    }
    __shared__ float svar;
    float            var_total = BR(tmp).Sum(local_sq);
    if (tid == 0) {
        svar = var_total / static_cast<float>(D) + eps;
    }
    __syncthreads();
    const float inv_std = rsqrtf(svar);

    // 3. Affine + bias.
    for (int j = tid; j < D; j += BLOCK) {
        const float normed = (to_f32_dev<T>(xr[j]) - mean) * inv_std;
        const float wv     = to_f32_dev<T>(w[j]);
        const float bv     = to_f32_dev<T>(b[j]);
        yr[j]              = from_f32_dev<T>(normed * wv + bv);
    }
}

template <typename T>
void launch_layer_norm_bias(T* y, const T* x, const T* w, const T* b, float eps, size_t seq_len, size_t hidden_size,
                            cudaStream_t stream) {
    constexpr int BLOCK = 256;
    dim3          grid(static_cast<unsigned int>(seq_len));
    dim3          block(BLOCK);
    layer_norm_bias_kernel<T, BLOCK><<<grid, block, 0, stream>>>(y, x, w, b, static_cast<int>(hidden_size), eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("layer_norm_bias kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace

void layer_norm_bias(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                     float eps, zedinferDataType_t type, size_t seq_len, size_t hidden_size) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return launch_layer_norm_bias<float>(reinterpret_cast<float*>(output),
                                                 reinterpret_cast<const float*>(input),
                                                 reinterpret_cast<const float*>(weight),
                                                 reinterpret_cast<const float*>(bias), eps, seq_len, hidden_size,
                                                 stream);
        case ZEDINFER_DTYPE_F16:
            return launch_layer_norm_bias<half>(reinterpret_cast<half*>(output),
                                                reinterpret_cast<const half*>(input),
                                                reinterpret_cast<const half*>(weight),
                                                reinterpret_cast<const half*>(bias), eps, seq_len, hidden_size, stream);
        case ZEDINFER_DTYPE_BF16:
            return launch_layer_norm_bias<__nv_bfloat16>(
                reinterpret_cast<__nv_bfloat16*>(output), reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<const __nv_bfloat16*>(weight), reinterpret_cast<const __nv_bfloat16*>(bias), eps,
                seq_len, hidden_size, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
