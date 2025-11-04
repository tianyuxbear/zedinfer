#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, zedinferDataType_t type, size_t size, size_t len);
}