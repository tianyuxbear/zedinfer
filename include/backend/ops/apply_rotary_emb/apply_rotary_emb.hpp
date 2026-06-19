#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// Apply rotary positional embedding in-place to a [N, H, D] tensor using
// pre-computed cos/sin tables of shape [N, D]. Uses HuggingFace's
// non-interleaved layout: the first D/2 dims pair with the second D/2 dims.
//
//   q_new[n, h, d]       = q[n, h, d]       * cos[n, d]       - q[n, h, d + D/2] * sin[n, d]
//   q_new[n, h, d + D/2] = q[n, h, d + D/2] * cos[n, d + D/2] + q[n, h, d]       * sin[n, d + D/2]
//
// All tensors must share dtype and device. D must be even.
void apply_rotary_emb_inplace(tensor_t q, tensor_t cos, tensor_t sin);

} // namespace zedinfer::ops
