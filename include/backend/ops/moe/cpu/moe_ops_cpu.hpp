#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {

void add_scaled(std::byte* out, const std::byte* a, float alpha, zedinferDataType_t type, size_t numel);
void fill_zero(std::byte* data, size_t size_bytes);

} // namespace zedinfer::ops::cpu
