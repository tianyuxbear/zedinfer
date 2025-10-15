#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, NeollmDataType_t type, size_t size, size_t len);
}