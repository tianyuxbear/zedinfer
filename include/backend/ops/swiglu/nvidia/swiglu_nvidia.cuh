#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void swiglu(std::byte *output, const std::byte *gate, const std::byte *up, zedinferDataType_t type, size_t numel);
}