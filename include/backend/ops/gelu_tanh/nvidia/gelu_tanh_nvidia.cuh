#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

void gelu_tanh(std::byte* output, const std::byte* input, zedinferDataType_t dtype, size_t numel);

} // namespace zedinfer::ops::nvidia
