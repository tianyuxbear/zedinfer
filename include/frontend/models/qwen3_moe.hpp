#pragma once

#include "frontend/models/base.hpp"
#include "frontend/models/expert_weights.hpp"
#include "frontend/models/forward_config.hpp"

#include <memory>
#include <vector>

namespace zedinfer::model {

struct Qwen3MoEConfig : public ModelConfig {
    Qwen3MoEConfig(ModelConfig config) : ModelConfig(config) {}

    // MoE architecture fields
    size_t num_experts = 128;
    size_t num_experts_per_tok = 8;
    size_t moe_intermediate_size = 1536;
    size_t shared_expert_intermediate_size = 4096;
    size_t decoder_sparse_step = 1;   // every Nth layer is sparse (MoE); default 1 = all MoE
    std::vector<int> mlp_only_layers; // layers forced to dense MLP (override sparse_step)
    bool norm_topk_prob = true;

    // Attention fields (shared with Qwen3)
    int max_window_layers = 94;
    bool use_sliding_window = false;
};

class Qwen3MoEModel : public Model {
public:
    Qwen3MoEModel(Qwen3MoEConfig& config, std::unique_ptr<ModelWeights> weights);

    const Qwen3MoEConfig& config() const override { return config_; }
    const ModelWeights& weights() const override { return *weights_; }
    const ExpertWeights& expert_weights() const { return *expert_weights_; }
    std::string model_type() const override { return "qwen3_moe"; }
    size_t num_parameters() const override { return num_params_; }

    ModelForwardConfig forward_config() const override;

private:
    size_t calculate_num_parameters() const;

    Qwen3MoEConfig config_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<ExpertWeights> expert_weights_;
    size_t num_params_;
};

} // namespace zedinfer::model
