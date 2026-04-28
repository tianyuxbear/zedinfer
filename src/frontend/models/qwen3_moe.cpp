#include "frontend/models/qwen3_moe.hpp"

#include <plog/Log.h>

namespace zedinfer::model {

namespace {

// Detect the actual expert intermediate size from weight shapes for a given prefix.
// Config fields may not match actual weights across HuggingFace versions
// (e.g. Qwen3-30B-A3B has moe_intermediate_size=768 in config but some docs list 1536).
//
// Assumption: all experts across all layers share the same intermediate size. This holds
// for current Qwen3 MoE models. If a future model uses per-layer sizes, this needs to
// become a per-layer lookup (callers would index by layer_idx).
size_t detect_intermediate_size(const ModelWeights& weights, const std::string& prefix) {
    // GPTQ: gate_proj.weight_packed shape [out_features, K/8]
    std::string packed = prefix + "gate_proj.weight_packed";
    if (weights.has_tensor(packed)) {
        return weights.get_tensor(packed)->shape()[0];
    }
    // Dense: gate_proj.weight shape [out_features, in_features]
    std::string w = prefix + "gate_proj.weight";
    if (weights.has_tensor(w)) {
        return weights.get_tensor(w)->shape()[0];
    }
    return 0;
}

// Try to parse "layers.{L}.mlp.experts.{E}.{proj}.{suffix}" into components.
// Returns true on match. proj is one of: gate_proj, up_proj, down_proj.
bool parse_expert_tensor_name(const std::string& name, size_t& layer, size_t& expert_id, std::string& proj,
                              std::string& suffix) {
    static const std::string kPrefix = "layers.";
    static const std::string kMid = ".mlp.experts.";
    if (name.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    size_t pos = kPrefix.size();
    size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    try {
        layer = std::stoul(name.substr(pos, dot - pos));
    } catch (...) { return false; }
    if (name.compare(dot, kMid.size(), kMid) != 0) {
        return false;
    }
    pos = dot + kMid.size();
    dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    try {
        expert_id = std::stoul(name.substr(pos, dot - pos));
    } catch (...) { return false; }
    pos = dot + 1;
    dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    proj = name.substr(pos, dot - pos);
    if (proj != "gate_proj" && proj != "up_proj" && proj != "down_proj") {
        return false;
    }
    suffix = name.substr(dot + 1);
    return true;
}

// Assign a tensor to the correct slot in ExpertFFN based on proj + suffix.
void assign_expert_tensor(ExpertFFN& ffn, const std::string& proj, const std::string& suffix, tensor_t tensor) {
    tensor_t* target = nullptr;
    if (proj == "gate_proj") {
        if (suffix == "weight_packed") {
            target = &ffn.gate_packed;
        } else if (suffix == "weight_scale") {
            target = &ffn.gate_scale;
        } else if (suffix == "weight_g_idx") {
            target = &ffn.gate_g_idx;
        } else if (suffix == "weight") {
            target = &ffn.gate_weight;
        }
    } else if (proj == "up_proj") {
        if (suffix == "weight_packed") {
            target = &ffn.up_packed;
        } else if (suffix == "weight_scale") {
            target = &ffn.up_scale;
        } else if (suffix == "weight_g_idx") {
            target = &ffn.up_g_idx;
        } else if (suffix == "weight") {
            target = &ffn.up_weight;
        }
    } else if (proj == "down_proj") {
        if (suffix == "weight_packed") {
            target = &ffn.down_packed;
        } else if (suffix == "weight_scale") {
            target = &ffn.down_scale;
        } else if (suffix == "weight_g_idx") {
            target = &ffn.down_g_idx;
        } else if (suffix == "weight") {
            target = &ffn.down_weight;
        }
    }
    if (target) {
        *target = std::move(tensor);
    }
}

// Move all expert tensors out of ModelWeights into a new ExpertWeights.
std::unique_ptr<ExpertWeights> extract_expert_weights(ModelWeights& weights, size_t num_layers,
                                                      size_t num_experts_per_layer) {
    auto experts = std::make_unique<ExpertWeights>(num_layers, num_experts_per_layer);
    std::vector<std::string> to_remove;

    for (const auto& [name, tensor] : weights.get_all_weights()) {
        size_t layer = 0, expert_id = 0;
        std::string proj, suffix;
        if (!parse_expert_tensor_name(name, layer, expert_id, proj, suffix)) {
            continue;
        }
        if (layer >= num_layers || expert_id >= num_experts_per_layer) {
            continue;
        }
        assign_expert_tensor(experts->at(layer, expert_id), proj, suffix, tensor);
        to_remove.push_back(name);
    }

    for (const auto& name : to_remove) { weights.remove_tensor(name); }

    LOGI.printf("[Qwen3MoE] Extracted %zu expert tensors into ExpertWeights [%zu layers × %zu experts]",
                to_remove.size(), num_layers, num_experts_per_layer);
    return experts;
}

} // namespace

Qwen3MoEModel::Qwen3MoEModel(Qwen3MoEConfig& config, std::unique_ptr<ModelWeights> weights,
                             ExpertPoolConfig pool_config)
    : config_(config), weights_(std::move(weights)) {
    auto experts = extract_expert_weights(*weights_, config_.num_hidden_layers, config_.num_experts);
    expert_pool_ = std::make_unique<ExpertPool>(std::move(experts), pool_config);
    num_params_ = calculate_num_parameters();
}

ModelForwardConfig Qwen3MoEModel::forward_config() const {
    ModelForwardConfig cfg{config_, *weights_};
    cfg.has_qkv_bias = false;
    cfg.has_qk_norm = true;
    cfg.is_moe = true;
    cfg.num_experts = config_.num_experts;
    cfg.num_experts_per_tok = config_.num_experts_per_tok;
    cfg.norm_topk_prob = config_.norm_topk_prob;
    cfg.decoder_sparse_step = config_.decoder_sparse_step;
    cfg.mlp_only_layers = config_.mlp_only_layers;
    cfg.expert_pool = expert_pool_.get();

    // Detect actual expert intermediate sizes from weight shapes (via first expert's shape).
    size_t detected_moe = 0;
    size_t detected_shared = detect_intermediate_size(*weights_, "layers.0.mlp.shared_expert.");
    if (expert_pool_ && expert_pool_->num_layers() > 0) {
        // peek_expert is metadata-only: never triggers an H2D transfer under PINNED_LRU.
        const auto& ffn = expert_pool_->peek_expert(0, 0);
        if (ffn.gate_packed) {
            detected_moe = ffn.gate_packed->shape()[0];
        } else if (ffn.gate_weight) {
            detected_moe = ffn.gate_weight->shape()[0];
        }
    }

    cfg.moe_intermediate_size = (detected_moe > 0) ? detected_moe : config_.moe_intermediate_size;
    cfg.shared_expert_intermediate_size
        = (detected_shared > 0) ? detected_shared : config_.shared_expert_intermediate_size;

    // Shared expert is uniform across layers: check layer 0 once.
    cfg.has_shared_expert = weights_->has_tensor("layers.0.mlp.shared_expert.gate_proj.weight")
                         || weights_->has_tensor("layers.0.mlp.shared_expert.gate_proj.weight_packed");

    // Cache per-layer router weight tensors so compute_router_topk can index instead of
    // doing a string-key map lookup every layer × decode step. Null entry = layer's
    // router goes through the quantized path (dispatch_linear by prefix); compute
    // path falls back gracefully.
    cfg.router_weights.resize(config_.num_hidden_layers);
    for (size_t l = 0; l < config_.num_hidden_layers; ++l) {
        std::string name = cfg.router_weight_name(static_cast<int>(l));
        cfg.router_weights[l] = weights_->has_tensor(name) ? weights_->get_tensor(name) : nullptr;
    }

    // Cache quant params used by every dispatch_expert_linear call.
    cfg.expert_quant_num_bits = config_.quant_config.weights.num_bits;
    cfg.expert_quant_group_size = config_.quant_config.weights.group_size;

    if (detected_moe > 0 && detected_moe != config_.moe_intermediate_size) {
        LOGI.printf("[Qwen3MoE] Detected moe_intermediate_size=%zu from weights (config=%zu)", detected_moe,
                    config_.moe_intermediate_size);
    }
    if (detected_shared > 0 && detected_shared != config_.shared_expert_intermediate_size) {
        LOGI.printf("[Qwen3MoE] Detected shared_expert_intermediate_size=%zu from weights (config=%zu)",
                    detected_shared, config_.shared_expert_intermediate_size);
    }

    return cfg;
}

size_t Qwen3MoEModel::calculate_num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_->get_all_weights()) { total += tensor->numel(); }
    // Also count expert tensors (held inside the pool). Use peek_expert so PINNED_LRU
    // doesn't churn the GPU slot arena just to compute numel.
    if (expert_pool_) {
        for (size_t l = 0; l < expert_pool_->num_layers(); ++l) {
            for (size_t e = 0; e < expert_pool_->num_experts_per_layer(); ++e) {
                const auto& ffn = expert_pool_->peek_expert(static_cast<int>(l), static_cast<int>(e));
                for (auto t : {ffn.gate_packed, ffn.gate_scale, ffn.gate_g_idx, ffn.gate_weight, ffn.up_packed,
                               ffn.up_scale, ffn.up_g_idx, ffn.up_weight, ffn.down_packed, ffn.down_scale,
                               ffn.down_g_idx, ffn.down_weight}) {
                    if (t) {
                        total += t->numel();
                    }
                }
            }
        }
    }
    return total;
}

} // namespace zedinfer::model
