#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void rope(std::byte* output, const std::byte* input, const std::byte* pos_ids, float theta, zedinferDataType_t type,
          size_t seq_len, size_t num_heads, size_t head_dim);
}