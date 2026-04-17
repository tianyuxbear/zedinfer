#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {

// out[i] += alpha * a[i] (in-place weighted accumulation)
void add_scaled(std::byte* out, const std::byte* a, float alpha, zedinferDataType_t type, size_t numel);

} // namespace zedinfer::ops::cpu
