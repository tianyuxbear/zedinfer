#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, NeollmDataType_t type, size_t numel);
}