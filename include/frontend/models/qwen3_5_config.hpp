#pragma once

#include "frontend/models/base.hpp"

#include <utility>
#include <vector>

namespace zedinfer::model {

// Dense Qwen3.5 (hybrid SSM + softmax attention + MRoPE + optional vision tower).
// All hybrid / vision / mrope / linear-attention fields live on the base ModelConfig;
// this subclass only carries the dense-only knobs that have no counterpart in
// Qwen2 / Qwen3 / Qwen3-MoE.
struct Qwen3_5Config : public ModelConfig {
    Qwen3_5Config() = default;
    Qwen3_5Config(ModelConfig base) : ModelConfig(std::move(base)) {}

    // Period (in layers) between full_attention layers in the hybrid stack.
    // Cross-check: must agree with the parsed layer_types pattern.
    int full_attention_interval = 4;
};

// MoE variant of Qwen3.5. Inherits hybrid/vision/mrope from Qwen3_5Config and
// adds the expert-routing fields.
struct Qwen3_5MoEConfig : public Qwen3_5Config {
    Qwen3_5MoEConfig() = default;
    Qwen3_5MoEConfig(ModelConfig base) : Qwen3_5Config(std::move(base)) {}

    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int decoder_sparse_step = 1;
    std::vector<int> mlp_only_layers;
};

} // namespace zedinfer::model
