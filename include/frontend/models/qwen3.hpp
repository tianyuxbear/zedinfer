#pragma once

#include "frontend/models/base.hpp"
#include "frontend/models/forward_config.hpp"

#include <memory>

namespace zedinfer::model {

struct Qwen3Config : public ModelConfig {
    Qwen3Config(ModelConfig config) : ModelConfig(config) {}
    int sliding_window;
    int max_window_layers;
    bool use_sliding_window;
};

class Qwen3Model : public Model {
public:
    Qwen3Model(Qwen3Config &config, std::unique_ptr<ModelWeights> weights)
        : config_(config), weights_(std::move(weights)) {
        num_params_ = calculate_num_parameters();
    }

    const Qwen3Config &config() const override { return config_; }
    const ModelWeights &weights() const override { return *weights_; }
    std::string model_type() const override { return "qwen3"; }
    size_t num_parameters() const override { return num_params_; }

    ModelForwardConfig forward_config() const override {
        return {config_, *weights_, /*has_qkv_bias=*/false, /*has_qk_norm=*/true};
    }

private:
    size_t calculate_num_parameters() const;

    Qwen3Config config_;
    std::unique_ptr<ModelWeights> weights_;
    size_t num_params_;
};

} // namespace zedinfer::model
