#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

// Naive bidirectional vision self-attention. q/k/v/out: [N, H, D] interleaved
// (i.e. element at [n, h, d] = ptr[n * H * D + h * D + d]). All dtypes equal.
void vision_attention(std::byte* out, const std::byte* q, const std::byte* k, const std::byte* v,
                      zedinferDataType_t dtype, int N, int H, int D, float scale);

} // namespace zedinfer::ops::nvidia
