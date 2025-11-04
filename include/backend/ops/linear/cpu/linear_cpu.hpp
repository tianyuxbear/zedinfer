#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, zedinferDataType_t type, size_t nrow, size_t ncol_out, size_t ncol_in);
}