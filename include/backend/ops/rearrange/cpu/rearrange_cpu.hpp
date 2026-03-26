#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <vector>

namespace zedinfer::ops::cpu {
void rearrange(std::byte* out, const std::byte* in, zedinferDataType_t type, size_t numel,
               const std::vector<size_t>& out_shape, const std::vector<ptrdiff_t>& out_strides,
               const std::vector<size_t>& in_shape, const std::vector<ptrdiff_t>& in_strides);
}