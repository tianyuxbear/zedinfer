#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, NeollmDataType_t type, size_t size);
}