#include "quantize.cuh"

#include "utils/nvidia/math.cuh"

#include <algorithm>

namespace zedinfer::ops::nvidia::detail {

inline constexpr int kBlockSize = 256;
inline constexpr int kWarpSize = 32;
inline constexpr float kQ8Max = 127.0f;

union PackedQuantizedValues {
    int8_t values[8];
    int2 packed;
};

template <typename T> __device__ float vector_abs_max(int4 packed_values) {
    static_assert(sizeof(T) == 2, "Quantized CUDA fallback supports only BF16/FP16 activations.");
    constexpr int kValuesPerVector = sizeof(int4) / sizeof(T);
    float local_max = 0.0f;
    const T* elements = reinterpret_cast<const T*>(&packed_values);

#pragma unroll
    for (int i = 0; i < kValuesPerVector; ++i) { local_max = fmaxf(local_max, fabsf(to_float(elements[i]))); }
    return local_max;
}

template <typename T> __device__ int2 quantize_vector(int4 packed_values, float inverse_scale) {
    static_assert(sizeof(T) == 2, "Quantized CUDA fallback supports only BF16/FP16 activations.");
    constexpr int kValuesPerVector = sizeof(int4) / sizeof(T);
    PackedQuantizedValues result{};
    const T* elements = reinterpret_cast<const T*>(&packed_values);

#pragma unroll
    for (int i = 0; i < kValuesPerVector; ++i) {
        result.values[i] = static_cast<int8_t>(rintf(to_float(elements[i]) * inverse_scale));
    }
    return result.packed;
}

__device__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float block_reduce_max(float value, float* shared_max, int tid, int num_threads) {
    const int lane_id = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    const int num_warps = (num_threads + kWarpSize - 1) / kWarpSize;

    value = warp_reduce_max(value);
    if (lane_id == 0) {
        shared_max[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = (lane_id < num_warps) ? shared_max[lane_id] : 0.0f;
        value = warp_reduce_max(value);
        if (lane_id == 0) {
            shared_max[0] = value;
        }
    }
    __syncthreads();
    return shared_max[0];
}

template <typename T>
__global__ void quantize_q8_row_vectorized_kernel(const T* __restrict__ input, int8_t* __restrict__ q_out,
                                                  T* __restrict__ scale_out, size_t K) {
    static_assert(sizeof(T) == 2, "Quantized CUDA fallback supports only BF16/FP16 activations.");
    constexpr int kValuesPerVector = sizeof(int4) / sizeof(T);
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_chunks = static_cast<int>(K) / kValuesPerVector;

    const T* input_row = input + row * K;
    int8_t* q_row = q_out + row * K;

    float local_max = 0.0f;
    for (int chunk = tid; chunk < num_chunks; chunk += num_threads) {
        const int4 packed_values = reinterpret_cast<const int4*>(input_row + chunk * kValuesPerVector)[0];
        local_max = fmaxf(local_max, vector_abs_max<T>(packed_values));
    }

    __shared__ float shared_max[kBlockSize / kWarpSize];
    const float amax = block_reduce_max(local_max, shared_max, tid, num_threads);
    const float scale = amax / kQ8Max;
    const float inverse_scale = (amax == 0.0f) ? 0.0f : 1.0f / scale;

    for (int chunk = tid; chunk < num_chunks; chunk += num_threads) {
        const int4 packed_values = reinterpret_cast<const int4*>(input_row + chunk * kValuesPerVector)[0];
        reinterpret_cast<int2*>(q_row + chunk * kValuesPerVector)[0] = quantize_vector<T>(packed_values, inverse_scale);
    }

    if (tid == 0) {
        scale_out[row] = from_float<T>(scale);
    }
}

template <typename T>
__global__ void quantize_q8_row_grouped_kernel(const T* __restrict__ input, int8_t* __restrict__ q_out,
                                               T* __restrict__ scale_out, size_t K, int group_size, int num_groups) {
    static_assert(sizeof(T) == 2, "Quantized CUDA fallback supports only BF16/FP16 activations.");
    constexpr int kValuesPerVector = sizeof(int4) / sizeof(T);
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const T* input_row = input + row * K;
    int8_t* q_row = q_out + row * K;

    __shared__ float shared_max[kBlockSize / kWarpSize];

    for (int group = 0; group < num_groups; ++group) {
        const int group_offset = group * group_size;
        const int num_chunks = group_size / kValuesPerVector;

        float local_max = 0.0f;
        for (int chunk = tid; chunk < num_chunks; chunk += num_threads) {
            const int4 packed_values
                = reinterpret_cast<const int4*>(input_row + group_offset + chunk * kValuesPerVector)[0];
            local_max = fmaxf(local_max, vector_abs_max<T>(packed_values));
        }

        const float amax = block_reduce_max(local_max, shared_max, tid, num_threads);
        const float scale = amax / kQ8Max;
        const float inverse_scale = (amax == 0.0f) ? 0.0f : 1.0f / scale;

        for (int chunk = tid; chunk < num_chunks; chunk += num_threads) {
            const int4 packed_values
                = reinterpret_cast<const int4*>(input_row + group_offset + chunk * kValuesPerVector)[0];
            reinterpret_cast<int2*>(q_row + group_offset + chunk * kValuesPerVector)[0]
                = quantize_vector<T>(packed_values, inverse_scale);
        }

        if (tid == 0) {
            scale_out[row * num_groups + group] = from_float<T>(scale);
        }
        __syncthreads();
    }
}

} // namespace zedinfer::ops::nvidia::detail

namespace zedinfer::ops::nvidia {

template <typename T>
void launch_quantize_q8_row(const T* input, int8_t* q_out, T* scale_out, size_t M, size_t K, cudaStream_t stream) {
    dim3 block(detail::kBlockSize);
    dim3 grid(M);
    detail::quantize_q8_row_vectorized_kernel<T><<<grid, block, 0, stream>>>(input, q_out, scale_out, K);
}

template <typename T>
void launch_quantize_q8_row_grouped(const T* input, int8_t* q_out, T* scale_out, size_t M, size_t K, int group_size,
                                    cudaStream_t stream) {
    if (group_size <= 0 || group_size >= static_cast<int>(K)) {
        group_size = static_cast<int>(K);
    }

    const int num_groups = static_cast<int>(K) / group_size;
    static_assert(sizeof(T) == 2, "Quantized CUDA fallback supports only BF16/FP16 activations.");
    constexpr int kValuesPerVector = sizeof(int4) / sizeof(T);
    const int chunks_per_group = group_size / kValuesPerVector;
    const int num_threads
        = std::clamp(((chunks_per_group + detail::kWarpSize - 1) / detail::kWarpSize) * detail::kWarpSize,
                     detail::kWarpSize, detail::kBlockSize);

    dim3 block(num_threads);
    dim3 grid(M);
    detail::quantize_q8_row_grouped_kernel<T>
        <<<grid, block, 0, stream>>>(input, q_out, scale_out, K, group_size, num_groups);
}

template void launch_quantize_q8_row<cuda_bfloat16>(const cuda_bfloat16*, int8_t*, cuda_bfloat16*, size_t, size_t,
                                                    cudaStream_t);

template void launch_quantize_q8_row<half>(const half*, int8_t*, half*, size_t, size_t, cudaStream_t);

template void launch_quantize_q8_row_grouped<cuda_bfloat16>(const cuda_bfloat16*, int8_t*, cuda_bfloat16*, size_t,
                                                            size_t, int, cudaStream_t);

template void launch_quantize_q8_row_grouped<half>(const half*, int8_t*, half*, size_t, size_t, int, cudaStream_t);

} // namespace zedinfer::ops::nvidia
