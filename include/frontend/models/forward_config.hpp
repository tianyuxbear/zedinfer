#pragma once

#include "frontend/models/base.hpp"

#include <stdexcept>
#include <string>

namespace zedinfer::model {

struct QuantizedLinearRef {
    tensor_t weight = nullptr;
    tensor_t bias = nullptr;
    tensor_t scale = nullptr;
    tensor_t g_idx = nullptr;
    int num_bits = 0;
    int group_size = -1;
};

/**
 * Static configuration for a model's forward pass.
 * Captures per-model-family differences (bias, Q/K norm, etc.)
 * so the shared transformer loop can be parameterized without virtual dispatch.
 */
struct ModelForwardConfig {
    const ModelConfig& config;
    const ModelWeights& weights;

    // Model-family features
    bool has_qkv_bias = false; // Qwen2: true, Qwen3/Llama: false
    bool has_qk_norm = false;  // Qwen3: true, Qwen2/Llama: false

    // Weight accessor
    tensor_t W(const std::string& name) const { return weights.get_tensor(name); }

    bool has_quantized_linear(const std::string& prefix) const {
        return weights.has_tensor(prefix + ".weight_packed") && weights.has_tensor(prefix + ".weight_scale");
    }

    QuantizedLinearRef quant_linear(const std::string& prefix) const {
        if (!has_quantized_linear(prefix)) {
            throw std::runtime_error("Quantized linear not found: " + prefix);
        }

        QuantizedLinearRef ref;
        ref.weight = weights.get_tensor(prefix + ".weight_packed");
        ref.scale = weights.get_tensor(prefix + ".weight_scale");
        if (weights.has_tensor(prefix + ".bias")) {
            ref.bias = weights.get_tensor(prefix + ".bias");
        }
        if (weights.has_tensor(prefix + ".weight_g_idx")) {
            ref.g_idx = weights.get_tensor(prefix + ".weight_g_idx");
        }

        ref.num_bits = config.quant_config.weights.num_bits;
        ref.group_size = config.quant_config.weights.group_size;

        if (ref.num_bits <= 0) {
            throw std::runtime_error("Quantized linear metadata missing num_bits for: " + prefix);
        }

        return ref;
    }

    // Layer weight prefix
    std::string prefix(int layer) const { return "layers." + std::to_string(layer) + "."; }

    // QKV bias (nullptr if !has_qkv_bias)
    tensor_t q_bias(const std::string& p) const { return has_qkv_bias ? W(p + "self_attn.q_proj.bias") : nullptr; }
    tensor_t k_bias(const std::string& p) const { return has_qkv_bias ? W(p + "self_attn.k_proj.bias") : nullptr; }
    tensor_t v_bias(const std::string& p) const { return has_qkv_bias ? W(p + "self_attn.v_proj.bias") : nullptr; }
};

// Shared transformer forward loop (defined in transformer_forward.cpp)
class PagedForwardContext;
struct DecodeScratch;
tensor_t transformer_forward(const ModelForwardConfig& model, PagedForwardContext& ctx,
                             const ExecutorConfig& exec_config, DecodeScratch* scratch = nullptr);

} // namespace zedinfer::model
