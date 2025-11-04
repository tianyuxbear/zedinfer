#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, zedinferDataType_t type, size_t numel);
}