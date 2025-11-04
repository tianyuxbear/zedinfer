#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, zedinferDataType_t type, size_t nrow, size_t ncol);
}