#include "backend/core/context/context.hpp"
#include "backend/ops/apply_rotary_emb/nvidia/apply_rotary_emb_nvidia.cuh"
#include "utils/check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

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

// One thread per (n, h, d_in_lower_half). Reads BOTH halves once into
// registers, then writes BOTH halves with the rotated values. No syncthreads
// needed because each lower-half element pairs with a unique upper-half
// element handled by the same thread.
template <typename T>
__global__ void apply_rotary_emb_kernel(T* q, const T* cos, const T* sin, int N, int H, int D) {
    const int half = D / 2;
    const int n    = blockIdx.x;
    const int h    = blockIdx.y;
    const int d    = blockIdx.z * blockDim.x + threadIdx.x;
    if (d >= half) {
        return;
    }
    const size_t base    = (static_cast<size_t>(n) * H + h) * D;
    const size_t cos_lo  = static_cast<size_t>(n) * D + d;
    const size_t cos_hi  = cos_lo + half;

    const float q_lo = to_f32_dev<T>(q[base + d]);
    const float q_hi = to_f32_dev<T>(q[base + d + half]);
    const float c_lo = to_f32_dev<T>(cos[cos_lo]);
    const float c_hi = to_f32_dev<T>(cos[cos_hi]);
    const float s_lo = to_f32_dev<T>(sin[cos_lo]);
    const float s_hi = to_f32_dev<T>(sin[cos_hi]);

    q[base + d]        = from_f32_dev<T>(q_lo * c_lo - q_hi * s_lo);
    q[base + d + half] = from_f32_dev<T>(q_hi * c_hi + q_lo * s_hi);
}

template <typename T>
void launch(T* q, const T* cos, const T* sin, int N, int H, int D, cudaStream_t stream) {
    constexpr int BLOCK = 64;
    const int     half  = D / 2;
    dim3          grid(static_cast<unsigned int>(N), static_cast<unsigned int>(H),
                       static_cast<unsigned int>((half + BLOCK - 1) / BLOCK));
    dim3          block(BLOCK);
    apply_rotary_emb_kernel<T><<<grid, block, 0, stream>>>(q, cos, sin, N, H, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("apply_rotary_emb kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace

void apply_rotary_emb_inplace(std::byte* q, const std::byte* cos, const std::byte* sin, zedinferDataType_t type, int N,
                              int H, int D) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (type) {
        case ZEDINFER_DTYPE_BF16:
            return launch<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(q),
                                         reinterpret_cast<const __nv_bfloat16*>(cos),
                                         reinterpret_cast<const __nv_bfloat16*>(sin), N, H, D, stream);
        case ZEDINFER_DTYPE_F16:
            return launch<half>(reinterpret_cast<half*>(q), reinterpret_cast<const half*>(cos),
                                reinterpret_cast<const half*>(sin), N, H, D, stream);
        case ZEDINFER_DTYPE_F32:
            return launch<float>(reinterpret_cast<float*>(q), reinterpret_cast<const float*>(cos),
                                 reinterpret_cast<const float*>(sin), N, H, D, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
