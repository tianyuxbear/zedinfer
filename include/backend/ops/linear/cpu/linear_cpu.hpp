#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, NeollmDataType_t type, size_t nrow, size_t ncol_out, size_t ncol_in);
}