#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

void add_scaled(std::byte* out, const std::byte* a, float alpha, zedinferDataType_t type, size_t numel);

} // namespace zedinfer::ops::nvidia
