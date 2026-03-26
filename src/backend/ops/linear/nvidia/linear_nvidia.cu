#include "backend/ops/linear/nvidia/linear_cublas.cuh"
#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "utils/check.hpp"
#include "zedinfer.h"

#include <cstddef>
#include <stdexcept>

namespace zedinfer::ops::nvidia {

void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K) {
    // cuBLASLt handles all shapes (M=1 GEMV and M>1 GEMM) and all dtypes
    // (FP32, FP16, BF16) with automatic kernel selection and fused bias.
    cublas::linear(output, input, weight, bias, type, M, N, K);
}

void linear_quantized(std::byte *output, const std::byte *input, const std::byte *weight, const std::byte *bias, const std::byte *scale, const std::byte *g_idx, zedinferDataType_t type, int num_bits, int group_size, size_t M, size_t N, size_t K) {
    (void)output;
    (void)input;
    (void)weight;
    (void)bias;
    (void)scale;
    (void)g_idx;
    (void)type;
    (void)num_bits;
    (void)group_size;
    (void)M;
    (void)N;
    (void)K;
    throw std::runtime_error(
        "linear_quantized NVIDIA backend is not implemented in this commit");
}

} // namespace zedinfer::ops::nvidia
