#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "backend/ops/linear/nvidia/linear_cublas.cuh"
#include "utils/check.hpp"
#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

void linear(std::byte *output, const std::byte *input, const std::byte *weight,
            const std::byte *bias, zedinferDataType_t type,
            size_t M, size_t N, size_t K) {
    // cuBLASLt handles all shapes (M=1 GEMV and M>1 GEMM) and all dtypes
    // (FP32, FP16, BF16) with automatic kernel selection and fused bias.
    cublas::linear(output, input, weight, bias, type, M, N, K);
}

} // namespace zedinfer::ops::nvidia
