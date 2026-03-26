#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {
void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K);
}