#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia::cublas {

// cuBLASLt-accelerated linear: C[M,N] = A[M,K] * B^T[K,N] + bias[N]
// B stored as [N,K] row-major.
// Handles FP32, FP16, BF16 with automatic kernel selection.
// Bias fused via cuBLASLt epilogue (no separate kernel needed).
void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K, cudaStream_t stream = nullptr);

} // namespace zedinfer::ops::nvidia::cublas
