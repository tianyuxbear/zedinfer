#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K);
void linear_quantized(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                      const std::byte* scale, const std::byte* g_idx, zedinferDataType_t type, int num_bits,
                      int group_size, size_t M, size_t N, size_t K);
} // namespace zedinfer::ops::nvidia
