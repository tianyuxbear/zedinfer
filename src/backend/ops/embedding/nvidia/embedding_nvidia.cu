#include "backend/ops/embedding/nvidia/embedding_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/memory.cuh"

namespace zedinfer::ops::nvidia {

// ----------------------------------------------------------------------
// Kernel 1: Scalar Embedding Lookup
// Baseline implementation: One block per sequence token, scalar copy.
// ----------------------------------------------------------------------
template <typename T>
__global__ void embedding_lookup_scalar_kernel(
    T *output,
    const T *table,
    const int *indices,
    size_t hidden_size) {
    // Each block processes one token in the sequence
    size_t seq_idx = blockIdx.x;
    size_t tid = threadIdx.x;

    // Resolve the target row index from the embedding table
    int table_idx = indices[seq_idx];

    // Calculate offsets (use size_t to prevent overflow)
    size_t table_offset = static_cast<size_t>(table_idx) * hidden_size;
    size_t out_offset = seq_idx * hidden_size;

    // Parallel copy within the block
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        output[out_offset + i] = table[table_offset + i];
    }
}

// ----------------------------------------------------------------------
// Kernel 2: Packed Embedding Lookup (Vectorized)
// Optimized: Uses 128-bit vectorized load/store for high bandwidth.
// ----------------------------------------------------------------------
template <typename T>
__global__ void embedding_lookup_packed_kernel(
    T *output,
    const T *table,
    const int *indices,
    size_t hidden_size) {

    constexpr int PackSize = PackedTraits<T>::size; // e.g., 4 for float, 8 for half

    size_t seq_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    int table_idx = indices[seq_idx];

    // Base pointers for the current token
    size_t table_offset = static_cast<size_t>(table_idx) * hidden_size;
    size_t out_offset = seq_idx * hidden_size;

    // 1. Vectorized Loop (128-bit chunks)
    size_t num_packs = hidden_size / PackSize;

    for (size_t i = tid; i < num_packs; i += blockDim.x) {
        size_t elem_idx = i * PackSize;
        float4 val = load_128b(&table[table_offset + elem_idx]);
        store_128b(&output[out_offset + elem_idx], val);
    }

    // 2. Scalar Tail Loop (Handle remaining elements)
    // Only necessary if hidden_size is not a multiple of PackSize
    size_t processed = num_packs * PackSize;

    for (size_t i = processed + tid; i < hidden_size; i += blockDim.x) {
        output[out_offset + i] = table[table_offset + i];
    }
}

void embedding(std::byte *output, const std::byte *indices, const std::byte *weight, zedinferDataType_t type, size_t numel, size_t hidden_size) {
    const size_t seqlen = numel / hidden_size;
    dim3 block(BLOCK_SIZE);
    dim3 grid(seqlen);

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return embedding_lookup_packed_kernel<<<grid, block>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const int *>(indices),
            hidden_size);
    case ZEDINFER_DTYPE_F16:
        return embedding_lookup_packed_kernel<<<grid, block>>>(
            reinterpret_cast<half *>(output),
            reinterpret_cast<const half *>(weight),
            reinterpret_cast<const int *>(indices),
            hidden_size);
    case ZEDINFER_DTYPE_BF16:
        return embedding_lookup_packed_kernel<<<grid, block>>>(
            reinterpret_cast<cuda_bfloat16 *>(output),
            reinterpret_cast<const cuda_bfloat16 *>(weight),
            reinterpret_cast<const int *>(indices),
            hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
