#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cstdint>

namespace zedinfer::ops::nvidia {

void gather_rows(std::byte* dst, const std::byte* src, const std::int32_t* indices_device, std::size_t num_rows,
                 std::size_t row_bytes);

} // namespace zedinfer::ops::nvidia
