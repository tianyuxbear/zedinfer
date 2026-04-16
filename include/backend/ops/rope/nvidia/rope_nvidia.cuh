#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void rope(std::byte* output, const std::byte* input, const std::byte* pos_ids, float theta, zedinferDataType_t type,
          size_t seq_len, size_t num_heads, size_t head_dim);
void rope_qk(std::byte* q_output, std::byte* k_output, const std::byte* q_input, const std::byte* k_input,
             const std::byte* pos_ids, float theta, zedinferDataType_t type, size_t seq_len, size_t num_q_heads,
             size_t num_kv_heads, size_t head_dim);
} // namespace zedinfer::ops::nvidia
