#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void embedding(std::byte* output, const std::byte* indices, const std::byte* weight, zedinferDataType_t type,
               size_t numel, size_t hidden_size);
}