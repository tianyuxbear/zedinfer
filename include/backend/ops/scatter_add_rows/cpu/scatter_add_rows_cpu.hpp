#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cstdint>

namespace zedinfer::ops::cpu {

// out[indices[r], :] += weights[r] * src[r, :]
// dtype-aware: accumulation goes through float to preserve precision for F16/BF16.
void scatter_add_rows(std::byte* out, const std::byte* src, const std::int32_t* indices, const float* weights,
                      zedinferDataType_t type, std::size_t num_rows, std::size_t row_elements);

} // namespace zedinfer::ops::cpu
