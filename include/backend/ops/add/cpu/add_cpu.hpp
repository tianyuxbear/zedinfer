#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void add(std::byte* c, const std::byte* a, const std::byte* b, zedinferDataType_t type, size_t numel);
}