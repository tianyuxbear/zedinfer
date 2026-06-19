#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {

// PyTorch gelu_pytorch_tanh:
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Element-wise; x and y share shape and dtype.
void gelu_tanh(std::byte* output, const std::byte* input, zedinferDataType_t dtype, size_t numel);

} // namespace zedinfer::ops::cpu
