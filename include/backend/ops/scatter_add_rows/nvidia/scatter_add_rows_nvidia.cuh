#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cstdint>

namespace zedinfer::ops::nvidia {

void scatter_add_rows(std::byte* out, const std::byte* src, const std::int32_t* indices_device,
                      const float* weights_device, zedinferDataType_t type, std::size_t num_rows,
                      std::size_t row_elements);

} // namespace zedinfer::ops::nvidia
