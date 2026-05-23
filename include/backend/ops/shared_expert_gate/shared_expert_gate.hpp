#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// Per-token sigmoid gate used by Qwen3.5 MoE shared-expert path.
// In place: x[n, h] := x[n, h] * sigmoid(gate[n, 0]).
//
// `x` is the shared expert FFN output of shape [N, hidden_size] bf16.
// `gate` is a per-token logit of shape [N, 1] bf16 produced by the
//   layers.{L}.mlp.shared_expert_gate linear projection.
//
// HF Qwen3_5MoeSparseMoeBlock.forward (modeling_qwen3_5_moe.py:809):
//   shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden)) * shared_expert_output
void shared_expert_gate(tensor_t x, tensor_t gate);

} // namespace zedinfer::ops
