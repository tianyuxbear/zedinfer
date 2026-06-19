#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// Post-attention sigmoid gate used by Qwen3.5 full-attention layers.
// In place: attn := attn * sigmoid(g).
// Both tensors must share the same shape (typically [N, num_heads, head_dim])
// and dtype (bf16 on NVIDIA, f32 on CPU).
void attn_output_gate(tensor_t attn, tensor_t g);

} // namespace zedinfer::ops
