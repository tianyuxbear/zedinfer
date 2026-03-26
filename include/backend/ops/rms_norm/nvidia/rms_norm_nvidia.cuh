#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void rms_norm(std::byte* output, const std::byte* input, const std::byte* weight, float eps, zedinferDataType_t type,
              size_t seq_len, size_t hidden_size);
}