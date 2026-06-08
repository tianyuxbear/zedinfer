#include "backend/core/context/context.hpp"
#include "backend/ops/sample/nvidia/sample_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "utils/nvidia/types.cuh"

#include <cub/cub.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace zedinfer::ops::nvidia {
namespace {

constexpr int kThreads = 128;
constexpr int kMaxTopK = 32;

template <typename T> __global__ void convert_logits_kernel(float* dst, const T* src, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = idx; i < n; i += stride) { dst[i] = to_float<T>(src[i]); }
}

__global__ void init_token_ids_kernel(int32_t* ids, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = idx; i < n; i += stride) { ids[i] = static_cast<int32_t>(i); }
}

__global__ void apply_repetition_penalty_kernel(float* logits, const int32_t* recent_tokens, size_t recent_count,
                                                size_t vocab_size, float penalty) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || penalty == 1.0f) {
        return;
    }
    for (size_t i = 0; i < recent_count; ++i) {
        const int32_t token = recent_tokens[i];
        if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
            continue;
        }
        float& logit = logits[token];
        logit = logit > 0.0f ? (logit / penalty) : (logit * penalty);
    }
}

__device__ __forceinline__ bool better_pair(float lhs_val, int lhs_id, float rhs_val, int rhs_id) {
    return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_id < rhs_id);
}

template <int MAX_K>
__device__ __forceinline__ void insert_topk(float val, int id, float (&vals)[MAX_K], int (&ids)[MAX_K], int k) {
    if (!better_pair(val, id, vals[k - 1], ids[k - 1])) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && better_pair(val, id, vals[pos - 1], ids[pos - 1])) {
        vals[pos] = vals[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
    }
    vals[pos] = val;
    ids[pos] = id;
}

template <int THREADS, int MAX_K>
__global__ void sample_topk_kernel(int64_t* out_token, const float* logits, size_t vocab_size, float temperature,
                                   int top_k, float top_p, float random) {
    __shared__ float reduce_buf[THREADS];
    __shared__ float shared_top_vals[THREADS * MAX_K];
    __shared__ int shared_top_ids[THREADS * MAX_K];
    __shared__ float global_max;
    __shared__ float global_sum_exp;

    const int tid = threadIdx.x;
    float local_top_vals[MAX_K];
    int local_top_ids[MAX_K];
    for (int i = 0; i < MAX_K; ++i) {
        local_top_vals[i] = -FLT_MAX;
        local_top_ids[i] = INT_MAX;
    }

    float local_max = -FLT_MAX;
    for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += THREADS) {
        const float val = logits[i] / temperature;
        local_max = fmaxf(local_max, val);
        insert_topk<MAX_K>(val, static_cast<int>(i), local_top_vals, local_top_ids, top_k);
    }

    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_max = reduce_buf[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += THREADS) {
        local_sum += __expf((logits[i] / temperature) - global_max);
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] += reduce_buf[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_sum_exp = reduce_buf[0];
    }

    for (int i = 0; i < MAX_K; ++i) {
        shared_top_vals[tid * MAX_K + i] = local_top_vals[i];
        shared_top_ids[tid * MAX_K + i] = local_top_ids[i];
    }
    __syncthreads();

    if (tid != 0) {
        return;
    }

    float final_vals[MAX_K];
    int final_ids[MAX_K];
    for (int i = 0; i < MAX_K; ++i) {
        final_vals[i] = -FLT_MAX;
        final_ids[i] = INT_MAX;
    }

    for (int t = 0; t < THREADS; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const float val = shared_top_vals[t * MAX_K + k];
            const int id = shared_top_ids[t * MAX_K + k];
            if (id != INT_MAX) {
                insert_topk<MAX_K>(val, id, final_vals, final_ids, top_k);
            }
        }
    }

    int keep = top_k;
    if (top_p < 1.0f) {
        float cumsum_prob = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            cumsum_prob += __expf(final_vals[k] - global_max) / global_sum_exp;
            if (cumsum_prob >= top_p) {
                keep = k + 1;
                break;
            }
        }
    }

    float kept_exp_sum = 0.0f;
    for (int k = 0; k < keep; ++k) { kept_exp_sum += __expf(final_vals[k] - global_max); }

    const float target = random * kept_exp_sum;
    float cumulative = 0.0f;
    int selected = final_ids[keep - 1];
    for (int k = 0; k < keep; ++k) {
        cumulative += __expf(final_vals[k] - global_max);
        if (target < cumulative) {
            selected = final_ids[k];
            break;
        }
    }

    *out_token = static_cast<int64_t>(selected);
}

template <int THREADS>
__global__ void sample_full_kernel(int64_t* out_token, const float* logits, size_t vocab_size, float temperature,
                                   float random) {
    __shared__ float reduce_buf[THREADS];
    __shared__ float global_max;
    __shared__ float global_sum_exp;

    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += THREADS) {
        local_max = fmaxf(local_max, logits[i] / temperature);
    }

    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_max = reduce_buf[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += THREADS) {
        local_sum += __expf((logits[i] / temperature) - global_max);
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] += reduce_buf[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_sum_exp = reduce_buf[0];
    }
    __syncthreads();

    if (tid != 0) {
        return;
    }

    const float target = random * global_sum_exp;
    float cumulative = 0.0f;
    int selected = static_cast<int>(vocab_size - 1);
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += __expf((logits[i] / temperature) - global_max);
        if (target < cumulative) {
            selected = static_cast<int>(i);
            break;
        }
    }

    *out_token = static_cast<int64_t>(selected);
}

template <int THREADS>
__global__ void sample_sorted_kernel(int64_t* out_token, const float* sorted_logits, const int32_t* sorted_ids,
                                     size_t vocab_size, float temperature, int top_k, float top_p, float random) {
    __shared__ float reduce_buf[THREADS];
    __shared__ float global_max;
    __shared__ float global_sum_exp;

    const int tid = threadIdx.x;
    const size_t limit
        = (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) ? static_cast<size_t>(top_k) : vocab_size;

    float local_max = -FLT_MAX;
    for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += THREADS) {
        local_max = fmaxf(local_max, sorted_logits[i] / temperature);
    }

    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_max = reduce_buf[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += THREADS) {
        local_sum += __expf((sorted_logits[i] / temperature) - global_max);
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduce_buf[tid] += reduce_buf[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_sum_exp = reduce_buf[0];
    }
    __syncthreads();

    if (tid != 0) {
        return;
    }

    size_t keep = limit;
    if (top_p < 1.0f) {
        float cumsum_prob = 0.0f;
        for (size_t k = 0; k < limit; ++k) {
            cumsum_prob += __expf((sorted_logits[k] / temperature) - global_max) / global_sum_exp;
            if (cumsum_prob >= top_p) {
                keep = k + 1;
                break;
            }
        }
    }

    float kept_exp_sum = 0.0f;
    for (size_t k = 0; k < keep; ++k) { kept_exp_sum += __expf((sorted_logits[k] / temperature) - global_max); }

    const float target = random * kept_exp_sum;
    float cumulative = 0.0f;
    int selected = sorted_ids[keep - 1];
    for (size_t k = 0; k < keep; ++k) {
        cumulative += __expf((sorted_logits[k] / temperature) - global_max);
        if (target < cumulative) {
            selected = sorted_ids[k];
            break;
        }
    }

    *out_token = static_cast<int64_t>(selected);
}

template <typename T> void launch_convert(float* dst, const std::byte* src, size_t n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>(1024, div_ceil(n, static_cast<size_t>(threads))));
    convert_logits_kernel<T><<<blocks, threads, 0, stream>>>(dst, reinterpret_cast<const T*>(src), n);
}

} // namespace

size_t sample_sort_workspace_bytes(size_t vocab_size) {
    size_t temp_bytes = 0;
    cudaError_t err = cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr), static_cast<int>(vocab_size));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("sample_sort_workspace_bytes failed: ") + cudaGetErrorString(err));
    }
    return temp_bytes;
}

void sample_token(std::byte* out_token, const std::byte* logits, float* work_logits, int32_t* work_ids,
                  float* sorted_logits, int32_t* sorted_ids, void* sort_temp, size_t sort_temp_bytes,
                  const int32_t* recent_tokens, size_t recent_count, zedinferDataType_t dtype, size_t vocab_size,
                  float temperature, int top_k, float top_p, float repetition_penalty, float random) {
    if (top_k < 0) {
        throw std::runtime_error("sample_token_nvidia: top_k must be non-negative");
    }
    if (vocab_size == 0) {
        throw std::runtime_error("sample_token_nvidia: vocab size must be positive");
    }

    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (dtype) {
        case ZEDINFER_DTYPE_F32:
            launch_convert<float>(work_logits, logits, vocab_size, stream);
            break;
        case ZEDINFER_DTYPE_F16:
            launch_convert<half>(work_logits, logits, vocab_size, stream);
            break;
        case ZEDINFER_DTYPE_BF16:
            launch_convert<cuda_bfloat16>(work_logits, logits, vocab_size, stream);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    if (recent_tokens != nullptr && recent_count > 0 && repetition_penalty != 1.0f) {
        apply_repetition_penalty_kernel<<<1, 1, 0, stream>>>(work_logits, recent_tokens, recent_count, vocab_size,
                                                             repetition_penalty);
    }

    const bool has_top_k_filter = top_k > 0 && static_cast<size_t>(top_k) < vocab_size;
    if (has_top_k_filter && top_k <= kMaxTopK) {
        sample_topk_kernel<kThreads, kMaxTopK><<<1, kThreads, 0, stream>>>(
            reinterpret_cast<int64_t*>(out_token), work_logits, vocab_size, temperature, top_k, top_p, random);
    } else if (!has_top_k_filter && top_p >= 1.0f) {
        sample_full_kernel<kThreads><<<1, kThreads, 0, stream>>>(reinterpret_cast<int64_t*>(out_token), work_logits,
                                                                 vocab_size, temperature, random);
    } else {
        if (work_ids == nullptr || sorted_logits == nullptr || sorted_ids == nullptr || sort_temp == nullptr) {
            throw std::runtime_error("sample_token_nvidia: sorted sampling workspace is required");
        }
        if (sort_temp_bytes == 0) {
            throw std::runtime_error("sample_token_nvidia: sort_temp workspace is too small");
        }

        const int threads = 256;
        const int blocks = static_cast<int>(std::min<size_t>(1024, div_ceil(vocab_size, static_cast<size_t>(threads))));
        init_token_ids_kernel<<<blocks, threads, 0, stream>>>(work_ids, vocab_size);
        cudaError_t sort_err = cub::DeviceRadixSort::SortPairsDescending(
            sort_temp, sort_temp_bytes, work_logits, sorted_logits, work_ids, sorted_ids, static_cast<int>(vocab_size),
            0, sizeof(float) * CHAR_BIT, stream);
        if (sort_err != cudaSuccess) {
            throw std::runtime_error(std::string("sample_token_nvidia sort failed: ") + cudaGetErrorString(sort_err));
        }
        sample_sorted_kernel<kThreads><<<1, kThreads, 0, stream>>>(reinterpret_cast<int64_t*>(out_token), sorted_logits,
                                                                   sorted_ids, vocab_size, temperature, top_k, top_p,
                                                                   random);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("sample_token_nvidia launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace zedinfer::ops::nvidia
