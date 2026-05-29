#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

void apply_rotary_emb_inplace(std::byte* q, const std::byte* cos, const std::byte* sin, zedinferDataType_t dtype,
                              int N, int H, int D);

} // namespace zedinfer::ops::nvidia
