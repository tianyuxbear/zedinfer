#include "frontend/models/qwen3_moe.hpp"

#include <plog/Log.h>

namespace zedinfer::model {

// Detect the actual expert intermediate size from weight shapes.
// The config field moe_intermediate_size may not match the actual weights
// (e.g., some HuggingFace versions differ in what the field means).
static size_t detect_intermediate_size(const ModelWeights& weights, const std::string& prefix) {
    // Try quantized path first (GPTQ): gate_proj.weight_packed shape [out_features, K/8]
    std::string packed_name = prefix + "gate_proj.weight_packed";
    if (weights.has_tensor(packed_name)) {
        return weights.get_tensor(packed_name)->shape()[0];
    }
    // Try regular weight: gate_proj.weight shape [out_features, in_features]
    std::string weight_name = prefix + "gate_proj.weight";
    if (weights.has_tensor(weight_name)) {
        return weights.get_tensor(weight_name)->shape()[0];
    }
    return 0;
}

ModelForwardConfig Qwen3MoEModel::forward_config() const {
    ModelForwardConfig cfg{config_, *weights_, /*has_qkv_bias=*/false, /*has_qk_norm=*/true};
    cfg.is_moe = true;
    cfg.num_experts = config_.num_experts;
    cfg.num_experts_per_tok = config_.num_experts_per_tok;
    cfg.norm_topk_prob = config_.norm_topk_prob;

    // Detect actual expert intermediate sizes from weight shapes
    size_t detected_moe = detect_intermediate_size(*weights_, "layers.0.mlp.experts.0.");
    size_t detected_shared = detect_intermediate_size(*weights_, "layers.0.mlp.shared_expert.");

    cfg.moe_intermediate_size = (detected_moe > 0) ? detected_moe : config_.moe_intermediate_size;
    cfg.shared_expert_intermediate_size = (detected_shared > 0) ? detected_shared : config_.shared_expert_intermediate_size;

    if (detected_moe > 0 && detected_moe != config_.moe_intermediate_size) {
        LOGI.printf("[Qwen3MoE] Detected moe_intermediate_size=%zu from weights (config=%zu)", detected_moe,
                    config_.moe_intermediate_size);
    }
    if (detected_shared > 0 && detected_shared != config_.shared_expert_intermediate_size) {
        LOGI.printf("[Qwen3MoE] Detected shared_expert_intermediate_size=%zu from weights (config=%zu)", detected_shared,
                    config_.shared_expert_intermediate_size);
    }

    return cfg;
}

size_t Qwen3MoEModel::calculate_num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_->get_all_weights()) { total += tensor->numel(); }
    return total;
}

} // namespace zedinfer::model
