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
    } catch (...) {
        return false;
    }
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
    } catch (...) {
        return false;
    }
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
        if (suffix == "weight_packed") target = &ffn.gate_packed;
        else if (suffix == "weight_scale") target = &ffn.gate_scale;
        else if (suffix == "weight_g_idx") target = &ffn.gate_g_idx;
        else if (suffix == "weight") target = &ffn.gate_weight;
    } else if (proj == "up_proj") {
        if (suffix == "weight_packed") target = &ffn.up_packed;
        else if (suffix == "weight_scale") target = &ffn.up_scale;
        else if (suffix == "weight_g_idx") target = &ffn.up_g_idx;
        else if (suffix == "weight") target = &ffn.up_weight;
    } else if (proj == "down_proj") {
        if (suffix == "weight_packed") target = &ffn.down_packed;
        else if (suffix == "weight_scale") target = &ffn.down_scale;
        else if (suffix == "weight_g_idx") target = &ffn.down_g_idx;
        else if (suffix == "weight") target = &ffn.down_weight;
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

    for (const auto& name : to_remove) {
        weights.remove_tensor(name);
    }

    LOGI.printf("[Qwen3MoE] Extracted %zu expert tensors into ExpertWeights [%zu layers × %zu experts]",
                to_remove.size(), num_layers, num_experts_per_layer);
    return experts;
}

} // namespace

Qwen3MoEModel::Qwen3MoEModel(Qwen3MoEConfig& config, std::unique_ptr<ModelWeights> weights)
    : config_(config), weights_(std::move(weights)) {
    expert_weights_ = extract_expert_weights(*weights_, config_.num_hidden_layers, config_.num_experts);
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
    cfg.experts = expert_weights_.get();

    // Detect actual expert intermediate sizes from weight shapes (via first expert's shape).
    size_t detected_moe = 0;
    size_t detected_shared = detect_intermediate_size(*weights_, "layers.0.mlp.shared_expert.");
    if (expert_weights_ && expert_weights_->num_layers() > 0) {
        const auto& first = expert_weights_->at(0, 0);
        if (first.gate_packed) {
            detected_moe = first.gate_packed->shape()[0];
        } else if (first.gate_weight) {
            detected_moe = first.gate_weight->shape()[0];
        }
    }

    cfg.moe_intermediate_size = (detected_moe > 0) ? detected_moe : config_.moe_intermediate_size;
    cfg.shared_expert_intermediate_size
        = (detected_shared > 0) ? detected_shared : config_.shared_expert_intermediate_size;

    // Shared expert is uniform across layers: check layer 0 once.
    cfg.has_shared_expert
        = weights_->has_tensor("layers.0.mlp.shared_expert.gate_proj.weight")
       || weights_->has_tensor("layers.0.mlp.shared_expert.gate_proj.weight_packed");

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
    // Also count expert tensors (moved out of weights_).
    if (expert_weights_) {
        for (size_t l = 0; l < expert_weights_->num_layers(); ++l) {
            for (size_t e = 0; e < expert_weights_->num_experts_per_layer(); ++e) {
                const auto& ffn = expert_weights_->at(l, e);
                for (auto t : {ffn.gate_packed, ffn.gate_scale, ffn.gate_g_idx, ffn.gate_weight, ffn.up_packed,
                                ffn.up_scale, ffn.up_g_idx, ffn.up_weight, ffn.down_packed, ffn.down_scale,
                                ffn.down_g_idx, ffn.down_weight}) {
                    if (t) total += t->numel();
                }
            }
        }
    }
    return total;
}

} // namespace zedinfer::model
