#include "backend/core/context/context.hpp"
#include "backend/ops/vision_attention/nvidia/vision_attention_nvidia.cuh"
#include "utils/check.hpp"

#include <cub/cub.cuh>
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

// One CTA per (query position, head). Each CTA streams over all kv rows.
template <typename T, int BLOCK = 128>
__global__ void vision_attention_kernel(T* __restrict__ out, const T* __restrict__ q, const T* __restrict__ k,
                                        const T* __restrict__ v, float* __restrict__ scratch, int N, int H, int D,
                                        float scale) {
    const int i   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    const T* qrow    = q + (static_cast<size_t>(i) * H + h) * D;
    const T* k_start = k + h * D;
    const T* v_start = v + h * D;
    T*       orow    = out + (static_cast<size_t>(i) * H + h) * D;

    // Per-(i, h) scratch row of size N (fp32 scores then softmaxed).
    float* sc = scratch + (static_cast<size_t>(i) * H + h) * static_cast<size_t>(N);

    // 1. Scores: sc[j] = (q . k_j) * scale
    for (int j = tid; j < N; j += BLOCK) {
        const T* krow = k_start + static_cast<size_t>(j) * H * D;
        float    dot  = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += to_f32_dev<T>(qrow[d]) * to_f32_dev<T>(krow[d]);
        }
        sc[j] = dot * scale;
    }
    __syncthreads();

    // 2. Row max (block reduction).
    using BR = cub::BlockReduce<float, BLOCK>;
    __shared__ typename BR::TempStorage tmp;
    float                               local_max = -INFINITY;
    for (int j = tid; j < N; j += BLOCK) {
        if (sc[j] > local_max) {
            local_max = sc[j];
        }
    }
    // Use built-in max via lambda to dodge the cub::Max deprecation in newer CCCL.
    float row_max = BR(tmp).Reduce(local_max, [] __device__(float a, float b) { return a > b ? a : b; });
    __shared__ float s_max;
    if (tid == 0) {
        s_max = row_max;
    }
    __syncthreads();

    // 3. Exp + sum.
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += BLOCK) {
        const float e = expf(sc[j] - s_max);
        sc[j]         = e;
        local_sum += e;
    }
    float row_sum = BR(tmp).Sum(local_sum);
    __shared__ float s_sum;
    if (tid == 0) {
        s_sum = row_sum;
    }
    __syncthreads();
    const float inv_sum = 1.0f / s_sum;

    // 4. out[i, h, d] = sum_j (sc[j]/sum) * v[j, h, d]
    for (int d = tid; d < D; d += BLOCK) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            const T* vrow = v_start + static_cast<size_t>(j) * H * D;
            acc += sc[j] * inv_sum * to_f32_dev<T>(vrow[d]);
        }
        orow[d] = from_f32_dev<T>(acc);
    }
}

template <typename T>
void launch(T* out, const T* q, const T* k, const T* v, int N, int H, int D, float scale, cudaStream_t stream) {
    float*       scratch  = nullptr;
    const size_t sc_bytes = static_cast<size_t>(N) * H * N * sizeof(float);
    cudaError_t  err      = cudaMallocAsync(reinterpret_cast<void**>(&scratch), sc_bytes, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("vision_attention scratch alloc failed: ") + cudaGetErrorString(err));
    }

    constexpr int BLOCK = 128;
    dim3          grid(static_cast<unsigned int>(N), static_cast<unsigned int>(H));
    dim3          block(BLOCK);
    vision_attention_kernel<T, BLOCK><<<grid, block, 0, stream>>>(out, q, k, v, scratch, N, H, D, scale);
    err = cudaGetLastError();
    cudaFreeAsync(scratch, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("vision_attention kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace

void vision_attention(std::byte* out, const std::byte* q, const std::byte* k, const std::byte* v,
                      zedinferDataType_t type, int N, int H, int D, float scale) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (type) {
        case ZEDINFER_DTYPE_BF16:
            return launch<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(out),
                                         reinterpret_cast<const __nv_bfloat16*>(q),
                                         reinterpret_cast<const __nv_bfloat16*>(k),
                                         reinterpret_cast<const __nv_bfloat16*>(v), N, H, D, scale, stream);
        case ZEDINFER_DTYPE_F16:
            return launch<half>(reinterpret_cast<half*>(out), reinterpret_cast<const half*>(q),
                                reinterpret_cast<const half*>(k), reinterpret_cast<const half*>(v), N, H, D, scale,
                                stream);
        case ZEDINFER_DTYPE_F32:
            return launch<float>(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(q),
                                 reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v), N, H, D, scale,
                                 stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
