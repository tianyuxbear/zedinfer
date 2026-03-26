#pragma once

#include "backend/ops/attention_params.hpp"
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

void add(tensor_t c, tensor_t a, tensor_t b);
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
void embedding(tensor_t out, tensor_t index, tensor_t weight);
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr);
void linear_quantized(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, tensor_t scale, tensor_t g_idx, int num_bits, int group_size);
void rearrange(tensor_t out, tensor_t in);
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
void swiglu(tensor_t out, tensor_t gate, tensor_t up);

// Unified attention dispatch.
// Mode determined by AttentionParams fields — see attention_params.hpp.
void attention(const AttentionParams& params);

} // namespace zedinfer::ops
