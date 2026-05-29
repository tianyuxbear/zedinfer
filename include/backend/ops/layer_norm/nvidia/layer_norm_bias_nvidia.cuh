#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

// LayerNorm with affine + bias (ViT path). Same semantics as the CPU
// reference; one CTA per row, intra-block reduction.
void layer_norm_bias(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                     float eps, zedinferDataType_t dtype, size_t seq_len, size_t hidden_size);

} // namespace zedinfer::ops::nvidia
