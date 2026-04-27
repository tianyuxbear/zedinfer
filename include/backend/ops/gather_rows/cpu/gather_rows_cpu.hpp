#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cstdint>

namespace zedinfer::ops::cpu {

// dst[r, :] = src[indices[r], :]
// Bytes-only; dtype-agnostic. Caller computes row_bytes = row_elements * elementSize.
void gather_rows(std::byte* dst, const std::byte* src, const std::int32_t* indices, std::size_t num_rows,
                 std::size_t row_bytes);

} // namespace zedinfer::ops::cpu
