#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/hybrid_forward_config.hpp"

namespace zedinfer {
struct ExecutorConfig;
class InferenceRequest;
} // namespace zedinfer

namespace zedinfer::model {

class PagedForwardContext;
struct DecodeScratch;

// Hybrid forward loop for Qwen3.5-family models.
//
// Architecture (A2 layout decision, design doc §3):
//   - One outer loop over decoder layers L in [0, num_hidden_layers).
//   - Per-layer-kind dispatch (`HybridForwardConfig::layer_kinds[L]`):
//       LayerKind::Linear → forward_linear_attn_layer  (Mamba2 SSU + conv1d)
//       LayerKind::Full   → forward_full_attn_layer    (gated softmax attn + 3D mrope)
//   - Per-MLP dispatch (`is_moe_layer(L)`):
//       moe   → forward_moe_mlp     (delegates to v0.2.0 moe_layer_forward)
//       dense → forward_dense_mlp   (GPTQ Int4 gate/up/swiglu/down)
//
// `input_embeds` (optional): pre-computed vision tower output scattered into
// token positions, replaces embed_tokens lookup. nullptr → text-only path.
tensor_t hybrid_transformer_forward(const HybridForwardConfig& model,
                                      PagedForwardContext& ctx,
                                      InferenceRequest& req,
                                      const ExecutorConfig& exec,
                                      DecodeScratch* scratch = nullptr,
                                      tensor_t input_embeds = nullptr);

} // namespace zedinfer::model
