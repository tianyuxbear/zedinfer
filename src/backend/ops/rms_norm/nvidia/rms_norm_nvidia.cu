#include "backend/ops/rms_norm/nvidia/rms_norm_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/math.cuh"

namespace zedinfer::ops::nvidia {

// ----------------------------------------------------------------------
// Kernel 1: Block-level Reduction (Shared Memory Tree)
// Strategy: Standard tree reduction using shared memory.
// ----------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_kernel_block_reduce(
    T *output,
    const T *input,
    const T *weight,
    const float eps,
    size_t hidden_size) {
    // Dynamic shared memory required: blockDim.x * sizeof(float)
    __shared__ float s_sq_sums[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const size_t row_offset = blockIdx.x * hidden_size;

    // Resolve row pointers
    const T *row_input = input + row_offset;
    T *row_output = output + row_offset;

    // -------------------------------------------------------
    // Phase 1: Variance Calculation (Reduction)
    // -------------------------------------------------------
    float thread_sq_sum = 0.0f;

    // Grid-stride loop for accumulation
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        thread_sq_sum += val * val;
    }
    s_sq_sums[tid] = thread_sq_sum;
    __syncthreads();

    // Tree reduction input shared memory
    //
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq_sums[tid] += s_sq_sums[tid + stride];
        }
        __syncthreads();
    }

    // Compute Reciprocal Standard Deviation (RSTD)
    const float mean_sq = s_sq_sums[0] / static_cast<float>(hidden_size);
    const float rstd = rsqrtf(mean_sq + eps);

    // -------------------------------------------------------
    // Phase 2: Element-wise Normalization
    // -------------------------------------------------------
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        float w = to_float(weight[i]);
        float norm_val = val * w * rstd;
        row_output[i] = from_float<T>(norm_val);
    }
}

// ----------------------------------------------------------------------
// Kernel 2: Warp-level Reduction (Shuffle + Shared Mem)
// Strategy: Reduce within warps via shuffles first, then reduce warps.
// ----------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_kernel_warp_reduce(
    T *output,
    const T *input,
    const T *weight,
    const float eps,
    size_t hidden_size) {
    // Shared memory required: NUM_WARPS * sizeof(float)
    __shared__ float s_warp_sums[MAX_NUM_WARPS];

    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1f; // tid % 32
    const int warp_id = tid >> 5;   // tid / 32
    const size_t row_offset = blockIdx.x * hidden_size;

    const T *row_input = input + row_offset;
    T *row_output = output + row_offset;

    // -------------------------------------------------------
    // Phase 1: Variance Calculation
    // -------------------------------------------------------
    float thread_sq_sum = 0.0f;

    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        thread_sq_sum += val * val;
    }

    // Warp-level reduction (Shuffle)
    //
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        thread_sq_sum += __shfl_down_sync(0xffffffffu, thread_sq_sum, offset);
    }

    // Warp leader stores result
    if (lane_id == 0) {
        s_warp_sums[warp_id] = thread_sq_sum;
    }
    __syncthreads();

    // Block-level reduction (First warp reduces the warp sums)
    float block_sq_sum = 0.0f;
    if (tid < WARP_SIZE) { // Only first warp active
        // Load warp sum if within valid range
        int num_warps = blockDim.x / WARP_SIZE;
        if (tid < num_warps) {
            block_sq_sum = s_warp_sums[tid];
        }

        // Final shuffle reduction
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            block_sq_sum += __shfl_down_sync(0xffffffffu, block_sq_sum, offset);
        }

        // Broadcast final result via shared memory
        if (tid == 0) {
            s_warp_sums[0] = block_sq_sum;
        }
    }
    __syncthreads();

    const float mean_sq = s_warp_sums[0] / static_cast<float>(hidden_size);
    const float rstd = rsqrtf(mean_sq + eps);

    // -------------------------------------------------------
    // Phase 2: Element-wise Normalization
    // -------------------------------------------------------
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        float w = to_float(weight[i]);
        row_output[i] = from_float<T>(val * w * rstd);
    }
}

// ----------------------------------------------------------------------
// Kernel 3: Packed Vectorized Kernel (128-bit Loads)
// Strategy: Maximize memory bandwidth using float4/half8 load/stores.
// ----------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_kernel_warp_reduce_packed(
    T *output,
    const T *input,
    const T *weight,
    const float eps,
    size_t hidden_size) {
    __shared__ float s_warp_sums[MAX_NUM_WARPS];

    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1f;
    const int warp_id = tid >> 5;

    const size_t row_offset = blockIdx.x * hidden_size;
    const T *row_input = input + row_offset;
    T *row_output = output + row_offset;

    constexpr int PackSize = PackedTraits<T>::size;
    const size_t num_packs = hidden_size / PackSize;

    // -------------------------------------------------------
    // Phase 1: Variance Calculation (Vectorized)
    // -------------------------------------------------------
    float thread_sq_sum = 0.0f;

    // 128-bit packed accumulation
    for (size_t i = tid; i < num_packs; i += blockDim.x) {
        // Assume dot_packed_128b returns dot product of vector with itself
        thread_sq_sum += dot_packed_128b(row_input + i * PackSize, row_input + i * PackSize);
    }

    // Scalar tail handling
    size_t offset = num_packs * PackSize;
    for (size_t i = offset + tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        thread_sq_sum += val * val;
    }

    // Warp + Block Reduction (Same as Kernel 2)
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        thread_sq_sum += __shfl_down_sync(0xffffffffu, thread_sq_sum, offset);
    }
    if (lane_id == 0) {
        s_warp_sums[warp_id] = thread_sq_sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? s_warp_sums[tid] : 0.0f;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }
        if (tid == 0) {
            s_warp_sums[0] = block_sum;
        }
    }
    __syncthreads();

    const float rstd = rsqrtf((s_warp_sums[0] / static_cast<float>(hidden_size)) + eps);

    // -------------------------------------------------------
    // Phase 2: Normalization (Vectorized)
    // -------------------------------------------------------
    for (size_t i = tid; i < num_packs; i += blockDim.x) {
        size_t idx = i * PackSize;

        // Load 128-bit chunks
        float4 vec_vals = load_128b(&row_input[idx]);
        float4 vec_weights = load_128b(&weight[idx]);
        float4 vec_res; // Result accumulator

        // Process based on type (Manual SIMD unrolling)
        if constexpr (std::is_same_v<T, float>) {
            vec_res.x = vec_vals.x * vec_weights.x * rstd;
            vec_res.y = vec_vals.y * vec_weights.y * rstd;
            vec_res.z = vec_vals.z * vec_weights.z * rstd;
            vec_res.w = vec_vals.w * vec_weights.w * rstd;
        } else {
            // Half / BFloat16 handling via type punning
            // Simplified for clarity: reinterpret as array of T to iterate
            const T *vals_arr = reinterpret_cast<const T *>(&vec_vals);
            const T *w_arr = reinterpret_cast<const T *>(&vec_weights);
            T *res_arr = reinterpret_cast<T *>(&vec_res);

#pragma unroll
            for (int k = 0; k < PackSize; ++k) {
                float v = to_float(vals_arr[k]);
                float w = to_float(w_arr[k]);
                res_arr[k] = from_float<T>(v * w * rstd);
            }
        }
        store_128b(&row_output[idx], vec_res);
    }

    // Scalar tail normalization
    for (size_t i = offset + tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        float w = to_float(weight[i]);
        row_output[i] = from_float<T>(val * w * rstd);
    }
}

void rms_norm(std::byte *output, const std::byte *input, const std::byte *weight, float eps, zedinferDataType_t type, size_t seq_len, size_t hidden_size) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(seq_len);

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return rmsnorm_kernel_warp_reduce_packed<<<grid, block>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input),
            reinterpret_cast<const float *>(weight),
            eps, hidden_size);
    case ZEDINFER_DTYPE_F16:
        return rmsnorm_kernel_warp_reduce_packed<<<grid, block>>>(
            reinterpret_cast<half *>(output),
            reinterpret_cast<const half *>(input),
            reinterpret_cast<const half *>(weight),
            eps, hidden_size);
    case ZEDINFER_DTYPE_BF16:
        return rmsnorm_kernel_warp_reduce_packed<<<grid, block>>>(
            reinterpret_cast<cuda_bfloat16 *>(output),
            reinterpret_cast<const cuda_bfloat16 *>(input),
            reinterpret_cast<const cuda_bfloat16 *>(weight),
            eps, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
