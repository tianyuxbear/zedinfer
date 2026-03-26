#pragma once

#ifdef USE_ONEDNN

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu::onednn {

// oneDNN-accelerated linear: C[M,N] = A[M,K] * B^T[K,N] + bias[N]
// B is stored as [N,K] row-major (transposed weight).
// Handles FP32, BF16, FP16 with automatic ISA dispatch.
// Primitives are cached by (M, N, K, dtype) for zero re-creation overhead.
void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K);

} // namespace zedinfer::ops::cpu::onednn

#endif // USE_ONEDNN
