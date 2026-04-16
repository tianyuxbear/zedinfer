#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/forward_config.hpp"
#include "zedinfer/activation.hpp"

namespace zedinfer::model {

struct DecodeScratch;

// MoE layer forward: router + expert dispatch + shared expert + accumulation.
// Replaces the dense MLP (gate_proj/up_proj/swiglu/down_proj) for MoE layers.
//
// input: [N, hidden_size] — post-attention-norm hidden states
// output: [N, hidden_size] — MoE layer output (same shape as input)
//
// For N=1 (decode), no token permutation needed — just loops over top-k experts.
// For N>1 (prefill), loops over experts and gathers/scatters tokens.
void moe_layer_forward(const ModelForwardConfig& model, tensor_t output, tensor_t input, int layer_idx,
                       const ExecutorConfig& exec_config, DecodeScratch* scratch);

} // namespace zedinfer::model
