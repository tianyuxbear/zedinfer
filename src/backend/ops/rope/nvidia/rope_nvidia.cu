#include "backend/ops/rope/nvidia/rope_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/types.cuh"

namespace zedinfer::ops::nvidia {

// ----------------------------------------------------------------------
// Kernel: Apply Rotary Positional Embeddings (RoPE)
// Computes angles on-the-fly based on position IDs and theta.
// Logic: Rotates adjacent pairs (x[i], x[i + half_dim]) by calculated angle.
// ----------------------------------------------------------------------
template <typename T>
__global__ void RoPE_kernel(T* __restrict__ output, const T* __restrict__ input, const int64_t* __restrict__ pos_ids,
                            const float theta, const int seq_len, const int num_heads, const int head_dim) {
    const int half_dim = head_dim / 2;
    const int total_pairs = seq_len * num_heads * half_dim;

    // 1. Global thread index (linearized)
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= total_pairs) {
        return;
    }

    // 2. Coordinate Reconstruction (Decompose linear index)
    // Layout assumed: [seq_len, num_heads, head_dim]
    const int rotary_idx = global_idx % half_dim; // Index within the half-dimension
    const int remnant = global_idx / half_dim;
    const int head_idx = remnant % num_heads;     // Head index
    const int token_idx = remnant / num_heads;    // Token/Sequence index

    // 3. Compute Rotation Angle
    // inv_freq = 1.0 / (theta ^ (2 * i / dim))
    const float freq_exponent = (2.0f * rotary_idx) / static_cast<float>(head_dim);
    const float inv_freq = 1.0f / powf(theta, freq_exponent);
    const float angle = static_cast<float>(pos_ids[token_idx]) * inv_freq;

    // Compute sin/cos (fast intrinsic)
    float cos_val, sin_val;
    __sincosf(angle, &sin_val, &cos_val);

    // 4. Load & Rotate
    // Calculate base offset for the specific token and head
    // Note: Cast to size_t to prevent overflow on large tensors
    const size_t base_offset = (static_cast<size_t>(token_idx) * num_heads + head_idx) * head_dim;

    // Load pair elements (scalar load, potential optimization: vectorized load if layout allows)
    const float val_r = to_float(input[base_offset + rotary_idx]);            // Real part (logical)
    const float val_i = to_float(input[base_offset + rotary_idx + half_dim]); // Imaginary part (logical)

    // Apply rotation matrix:
    // [ cos -sin ] [ r ]
    // [ sin  cos ] [ i ]
    const float rot_r = val_r * cos_val - val_i * sin_val;
    const float rot_i = val_r * sin_val + val_i * cos_val;

    // Write back
    output[base_offset + rotary_idx] = from_float<T>(rot_r);
    output[base_offset + rotary_idx + half_dim] = from_float<T>(rot_i);
}

void rope(std::byte* output, const std::byte* input, const std::byte* pos_ids, float theta, zedinferDataType_t type,
          size_t seq_len, size_t num_heads, size_t head_dim) {
    // Total number of pairs to process (each thread handles 2 elements)
    const size_t total_pairs = seq_len * num_heads * head_dim / 2;

    dim3 block(BLOCK_SIZE);
    dim3 grid(div_ceil(total_pairs, BLOCK_SIZE));

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            RoPE_kernel<<<grid, block>>>(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input),
                                         reinterpret_cast<const int64_t*>(pos_ids), theta, seq_len, num_heads,
                                         head_dim);
            break;

        case ZEDINFER_DTYPE_F16:
            RoPE_kernel<<<grid, block>>>(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input),
                                         reinterpret_cast<const int64_t*>(pos_ids), theta, seq_len, num_heads,
                                         head_dim);
            break;

        case ZEDINFER_DTYPE_BF16:
            RoPE_kernel<<<grid, block>>>(
                reinterpret_cast<cuda_bfloat16*>(output), reinterpret_cast<const cuda_bfloat16*>(input),
                reinterpret_cast<const int64_t*>(pos_ids), theta, seq_len, num_heads, head_dim);
            break;

        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
