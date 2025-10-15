#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, NeollmDataType_t type, size_t size);
}