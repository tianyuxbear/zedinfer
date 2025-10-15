#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, NeollmDataType_t type, size_t nrow, size_t ncol);
}