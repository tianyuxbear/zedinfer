#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {

// LayerNorm with affine + bias for the ViT path:
//   y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
// Shapes: x/y [seq_len, hidden_size]; weight/bias [hidden_size].
// All tensors share the same dtype (BF16/FP16/FP32). Computation is FP32
// internally for numerical stability of the variance reduction.
void layer_norm_bias(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                     float eps, zedinferDataType_t dtype, size_t seq_len, size_t hidden_size);

} // namespace zedinfer::ops::cpu
