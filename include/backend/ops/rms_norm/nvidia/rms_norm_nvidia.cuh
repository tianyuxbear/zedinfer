#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

void rms_norm(std::byte* output, const std::byte* input, const std::byte* weight, float eps, zedinferDataType_t type,
              size_t seq_len, size_t hidden_size, bool add_one_to_weight = false);

// In-place fused: residual += input; input = rmsnorm(residual) * (weight + weight_bias).
// `add_one_to_weight=true` corresponds to GemmaFusedAddRMSNorm (weight_bias=1.0), matching
// the Qwen3.5/Gemma `output = x * (1.0 + weight)` form.
void fused_add_rms_norm(std::byte* input, std::byte* residual, const std::byte* weight, float eps,
                        zedinferDataType_t type, size_t seq_len, size_t hidden_size,
                        bool add_one_to_weight = false);

} // namespace zedinfer::ops::nvidia