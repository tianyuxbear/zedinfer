#include "backend/core/context/context.hpp"
#include "backend/ops/gather_rows/nvidia/gather_rows_nvidia.cuh"
#include "utils/nvidia/common.cuh"

#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

// One block per output row; threads collaborate on copying the row.
// row_words = row_bytes / 4; tail bytes (rare for normal hidden_size) handled separately.
__global__ void gather_rows_kernel(char* dst, const char* src, const int* indices, size_t num_rows, size_t row_bytes) {
    const size_t r = blockIdx.x;
    if (r >= num_rows) {
        return;
    }
    const size_t src_row = static_cast<size_t>(indices[r]);

    const char* sp = src + src_row * row_bytes;
    char* dp = dst + r * row_bytes;

    // 4-byte word copy when row_bytes is 4-aligned (true for F32/F16/BF16 with even hidden_size).
    const size_t row_words = row_bytes >> 2;
    const unsigned int* sw = reinterpret_cast<const unsigned int*>(sp);
    unsigned int* dw = reinterpret_cast<unsigned int*>(dp);
    for (size_t i = threadIdx.x; i < row_words; i += blockDim.x) {
        dw[i] = sw[i];
    }
    // Tail bytes (only fires when row_bytes % 4 != 0 — uncommon for our shapes).
    const size_t tail_start = row_words << 2;
    for (size_t i = tail_start + threadIdx.x; i < row_bytes; i += blockDim.x) {
        dp[i] = sp[i];
    }
}

} // namespace

void gather_rows(std::byte* dst, const std::byte* src, const std::int32_t* indices_device, std::size_t num_rows,
                 std::size_t row_bytes) {
    if (num_rows == 0) {
        return;
    }
    auto stream = reinterpret_cast<cudaStream_t>(zedinfer::core::context().runtime().stream());
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_rows);
    gather_rows_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<char*>(dst), reinterpret_cast<const char*>(src),
                                                   reinterpret_cast<const int*>(indices_device), num_rows, row_bytes);
}

} // namespace zedinfer::ops::nvidia
