#pragma once

#include "frontend/models/base.hpp"

#include <string>

namespace zedinfer::model {

/**
 * Static configuration for a model's forward pass.
 * Captures per-model-family differences (bias, Q/K norm, etc.)
 * so the shared transformer loop can be parameterized without virtual dispatch.
 */
struct ModelForwardConfig {
    const ModelConfig &config;
    const ModelWeights &weights;

    // Model-family features
    bool has_qkv_bias = false;   // Qwen2: true, Qwen3/Llama: false
    bool has_qk_norm = false;    // Qwen3: true, Qwen2/Llama: false

    // Weight accessor
    tensor_t W(const std::string &name) const {
        return weights.get_tensor(name);
    }

    // Layer weight prefix
    std::string prefix(int layer) const {
        return "layers." + std::to_string(layer) + ".";
    }

    // QKV bias (nullptr if !has_qkv_bias)
    tensor_t q_bias(const std::string &p) const {
        return has_qkv_bias ? W(p + "self_attn.q_proj.bias") : nullptr;
    }
    tensor_t k_bias(const std::string &p) const {
        return has_qkv_bias ? W(p + "self_attn.k_proj.bias") : nullptr;
    }
    tensor_t v_bias(const std::string &p) const {
        return has_qkv_bias ? W(p + "self_attn.v_proj.bias") : nullptr;
    }
};

// Shared transformer forward loop (defined in transformer_forward.cpp)
class ForwardContext;
struct DecodeScratch;
tensor_t transformer_forward(
    const ModelForwardConfig &model,
    ForwardContext &ctx,
    const ExecutorConfig &exec_config,
    DecodeScratch *scratch = nullptr);

} // namespace zedinfer::model
