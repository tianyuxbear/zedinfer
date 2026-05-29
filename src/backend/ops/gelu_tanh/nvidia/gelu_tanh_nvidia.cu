#include "backend/core/context/context.hpp"
#include "backend/ops/gelu_tanh/nvidia/gelu_tanh_nvidia.cuh"
#include "utils/check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

__device__ __forceinline__ float gelu_tanh_scalar(float x) {
    constexpr float k     = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float c     = 0.044715f;
    const float     inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

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

template <typename T> __global__ void gelu_tanh_kernel(T* y, const T* x, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = from_f32_dev<T>(gelu_tanh_scalar(to_f32_dev<T>(x[idx])));
    }
}

template <typename T> void launch_gelu_tanh(T* y, const T* x, size_t n, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const size_t  grid  = (n + BLOCK - 1) / BLOCK;
    gelu_tanh_kernel<T><<<static_cast<unsigned int>(grid), BLOCK, 0, stream>>>(y, x, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("gelu_tanh kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace

void gelu_tanh(std::byte* output, const std::byte* input, zedinferDataType_t type, size_t numel) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return launch_gelu_tanh<float>(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input),
                                           numel, stream);
        case ZEDINFER_DTYPE_F16:
            return launch_gelu_tanh<half>(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input), numel,
                                          stream);
        case ZEDINFER_DTYPE_BF16:
            return launch_gelu_tanh<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(output),
                                                   reinterpret_cast<const __nv_bfloat16*>(input), numel, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
