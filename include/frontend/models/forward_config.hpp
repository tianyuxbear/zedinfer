#pragma once

#include "backend/ops/ops.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/expert_pool.hpp"

#include <stdexcept>
#include <string>
#include <vector>

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
 * Captures per-model-family differences (bias, Q/K norm, MoE, etc.)
 * so the shared transformer loop can be parameterized without virtual dispatch.
 */
struct ModelForwardConfig {
    // Explicit constructor (prevents -Wmissing-field-initializers warnings from
    // aggregate brace-init when only config+weights are specified).
    ModelForwardConfig(const ModelConfig& cfg, const ModelWeights& w) : config(cfg), weights(w) {}

    const ModelConfig& config;
    const ModelWeights& weights;

    // Model-family features
    bool has_qkv_bias = false; // Qwen2: true, Qwen3/Llama: false
    bool has_qk_norm = false;  // Qwen3: true, Qwen2/Llama: false

    // MoE configuration (all zero/false for dense models).
    // `is_moe` indicates the model has MoE layers; `is_moe_layer(L)` tells whether
    // a specific layer L is sparse — some models (decoder_sparse_step > 1, or explicit
    // mlp_only_layers) mix dense and sparse layers.
    bool is_moe = false;
    size_t num_experts = 0;
    size_t num_experts_per_tok = 0;
    size_t moe_intermediate_size = 0;
    size_t shared_expert_intermediate_size = 0;
    bool norm_topk_prob = false;
    size_t decoder_sparse_step = 1;
    std::vector<int> mlp_only_layers;
    bool has_shared_expert = false;    // uniform across all MoE layers (checked at init)
    ExpertPool* expert_pool = nullptr; // manages GPU residency of expert weights

    // Per-layer router (gate) weight tensor, indexed by layer_idx. Populated by the
    // model's forward_config() factory so compute_router_topk can skip the per-layer
    // `router_weight_name(L) → has_tensor → get_tensor` string-keyed dance. Entry is
    // nullptr for layers whose router uses the quantized path; caller falls back to
    // dispatch_linear by prefix in that case.
    std::vector<tensor_t> router_weights;

    // Cached expert quantization parameters (set once at init from quant_config.weights).
    // Avoids repeated struct-navigation in dispatch_expert_linear's hot path.
    int expert_quant_num_bits = 0;
    int expert_quant_group_size = -1;

    bool is_moe_layer(size_t layer_idx) const {
        if (!is_moe) {
            return false;
        }
        // Dense-override list takes precedence.
        for (int l : mlp_only_layers) {
            if (static_cast<size_t>(l) == layer_idx) {
                return false;
            }
        }
        return decoder_sparse_step > 0 && (layer_idx % decoder_sparse_step == 0);
    }

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

    // Dispatch a linear op: selects quantized path when packed weights exist, else dense.
    // Used by both transformer_forward.cpp (dense path) and moe_forward.cpp (expert FFN).
    void dispatch_linear(tensor_t out, tensor_t in, const std::string& prefix, tensor_t bias = nullptr) const {
        if (has_quantized_linear(prefix)) {
            auto q = quant_linear(prefix);
            if (!bias) {
                bias = q.bias;
            }
            ops::linear_quantized(out, in, q.weight, bias, q.scale, q.g_idx, q.num_bits, q.group_size);
            return;
        }
        ops::linear(out, in, W(prefix + ".weight"), bias);
    }

    // Dispatch an expert FFN projection through the ExpertPool. The pool returns a handle
    // whose tensors are guaranteed on the compute device at return time:
    //   - M1 (ALL_GPU): trivial lookup, no transfer
    //   - M2 (PINNED_LRU, sync): ensure_on_gpu may block on cudaMemcpy
    //   - M3 (PINNED_LRU, async): ensure_on_gpu inserts cudaStreamWaitEvent on compute stream
    void dispatch_expert_linear(tensor_t out, tensor_t in, int layer, int expert_id, ExpertProj proj) const {
        if (!expert_pool) {
            throw std::runtime_error("dispatch_expert_linear called without expert_pool set");
        }
        // ensure_on_gpu returns a stable const reference; binding it locally avoids a
        // 12 × shared_ptr copy of the handle struct on every call. The four `const
        // tensor_t*` aliases below pick the (packed, scale, g_idx, weight) field set
        // for this projection without copying any shared_ptr until linear_quantized /
        // linear actually receives the args.
        const ExpertGpuHandle& h = expert_pool->ensure_on_gpu(layer, expert_id);
        const tensor_t* packed = nullptr;
        const tensor_t* scale = nullptr;
        const tensor_t* g_idx = nullptr;
        const tensor_t* weight = nullptr;
        switch (proj) {
            case ExpertProj::Gate:
                packed = &h.gate_packed;
                scale = &h.gate_scale;
                g_idx = &h.gate_g_idx;
                weight = &h.gate_weight;
                break;
            case ExpertProj::Up:
                packed = &h.up_packed;
                scale = &h.up_scale;
                g_idx = &h.up_g_idx;
                weight = &h.up_weight;
                break;
            case ExpertProj::Down:
                packed = &h.down_packed;
                scale = &h.down_scale;
                g_idx = &h.down_g_idx;
                weight = &h.down_weight;
                break;
        }
        if (*packed) {
            ops::linear_quantized(out, in, *packed, nullptr, *scale, *g_idx, expert_quant_num_bits,
                                  expert_quant_group_size);
        } else if (*weight) {
            ops::linear(out, in, *weight, nullptr);
        } else {
            throw std::runtime_error("Expert FFN weights missing for layer " + std::to_string(layer) + " expert "
                                     + std::to_string(expert_id));
        }
    }

    // Layer weight prefix
    std::string prefix(int layer) const { return "layers." + std::to_string(layer) + "."; }

    // MoE weight prefixes
    std::string expert_prefix(int layer, int expert_id) const {
        return "layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert_id) + ".";
    }

    std::string shared_expert_prefix(int layer) const {
        return "layers." + std::to_string(layer) + ".mlp.shared_expert.";
    }

    std::string router_weight_name(int layer) const { return "layers." + std::to_string(layer) + ".mlp.gate.weight"; }

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
