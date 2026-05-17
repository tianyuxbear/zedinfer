#include "frontend/models/qwen3_5_moe.hpp"

#include "frontend/models/expert_weights.hpp"

#include <plog/Log.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zedinfer::model {

namespace {

// Same naming convention as v0.2.0's Qwen3MoEModel after the loader strips the
// "model." / "language_model." prefixes — experts live at
//   layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.{weight|weight_packed|weight_scale|weight_g_idx}
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
// Mirrors src/frontend/models/qwen3_moe.cpp:extract_expert_weights.
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

    LOGI.printf("[Qwen3_5MoE] Extracted %zu expert tensors into ExpertWeights [%zu layers x %zu experts]",
                to_remove.size(), num_layers, num_experts_per_layer);
    return experts;
}

} // namespace

Qwen3_5MoeModel::Qwen3_5MoeModel(Qwen3_5MoEConfig config, std::unique_ptr<ModelWeights> weights,
                                 const ExecutorConfig& exec, int max_concurrent, ExpertPoolConfig pool_cfg,
                                 const std::string& model_path)
    // Forward to parent: Qwen3_5Model takes ownership of the weights and exposes them through the
    // protected weights_ member. The parent ctor runs SSM pool sizing + optional VisionTower
    // verification first; expert extraction below mutates *weights_ in-place after that. The
    // base-slice copy is intentional — Qwen3_5Config / Qwen3_5MoEConfig are plain structs with
    // no virtual functions, so slicing the MoE-only fields off the parent copy is safe.
    : Qwen3_5Model(static_cast<const Qwen3_5Config&>(config), std::move(weights), exec, max_concurrent, model_path),
      moe_config_(std::move(config)) {
    if (!weights_) {
        throw std::runtime_error("[Qwen3_5MoeModel] parent ctor did not retain ModelWeights");
    }
    // Early guard: a missing or zero num_experts in config silently produces an empty
    // ExpertPool, which is only detectable later via an "experts not found" failure deep
    // inside the M1 forward path. Catch it at construction time so misconfigured models
    // (e.g. text_config.num_experts absent from config.json) fail fast and clearly.
    if (moe_config_.num_experts <= 0 || moe_config_.num_experts_per_tok <= 0) {
        throw std::runtime_error(
            "[Qwen3_5MoeModel] invalid MoE config: num_experts="
            + std::to_string(moe_config_.num_experts)
            + ", num_experts_per_tok=" + std::to_string(moe_config_.num_experts_per_tok)
            + " (both must be > 0; check config.json text_config)");
    }
    auto experts = extract_expert_weights(*weights_, moe_config_.num_hidden_layers,
                                          static_cast<size_t>(moe_config_.num_experts));
    expert_pool_ = std::make_unique<ExpertPool>(std::move(experts), pool_cfg);

    LOGI.printf("[Qwen3_5MoeModel] constructed: experts=%d top_k=%d shared_expert_size=%d",
                moe_config_.num_experts, moe_config_.num_experts_per_tok,
                moe_config_.shared_expert_intermediate_size);
}

Qwen3_5MoeModel::~Qwen3_5MoeModel() = default;

size_t Qwen3_5MoeModel::num_parameters() const {
    // Count tensors still held by *weights_* (embeddings, attention, shared experts, LM head, ...).
    size_t total = 0;
    for (const auto& [name, tensor] : weights_->get_all_weights()) {
        (void)name;
        if (tensor) {
            total += tensor->numel();
        }
    }
    // Plus the per-(layer, expert) FFN tensors that the pool now owns. Use peek_expert so
    // PINNED_LRU doesn't churn the GPU slot arena just to enumerate sizes.
    if (expert_pool_) {
        for (size_t l = 0; l < expert_pool_->num_layers(); ++l) {
            for (size_t e = 0; e < expert_pool_->num_experts_per_layer(); ++e) {
                const auto& ffn = expert_pool_->peek_expert(static_cast<int>(l), static_cast<int>(e));
                for (auto t :
                     {ffn.gate_packed, ffn.gate_scale, ffn.gate_g_idx, ffn.gate_weight, ffn.up_packed, ffn.up_scale,
                      ffn.up_g_idx, ffn.up_weight, ffn.down_packed, ffn.down_scale, ffn.down_g_idx, ffn.down_weight}) {
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
