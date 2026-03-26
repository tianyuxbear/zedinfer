#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, zedinferDataType_t type, size_t numel);
}