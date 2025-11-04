#pragma once

#include "frontend/models/base.hpp"

#include <memory>

namespace zedinfer::model {

// Qwen2-specific configuration extending base model config.
struct Qwen2Config : public ModelConfig {
    Qwen2Config(ModelConfig config) : ModelConfig(config) {}
    int sliding_window;      // Sliding window size for attention
    int max_window_layers;   // Number of layers using sliding window
    bool use_sliding_window; // Whether sliding window attention is enabled
};

// Concrete implementation of the Qwen2 model.
class Qwen2Model : public Model {
public:
    Qwen2Model(Qwen2Config &config, std::unique_ptr<ModelWeights> weights) : config_(config), weights_(std::move(weights)) {
        num_params_ = calculate_num_parameters();
    };

    const Qwen2Config &config() const override {
        return config_;
    }
    const ModelWeights &weights() const override {
        return *weights_;
    };

    // Weight name accessors for inference graph construction
    std::string get_embedding_weight_name() const override;
    std::string get_output_norm_weight_name() const override;
    std::string get_output_weight_name() const override;
    std::vector<std::string> get_layer_weight_names(int layer_idx) const override;

    std::string model_type() const override { return "qwen2"; };
    size_t num_parameters() const override { return num_params_; };

private:
    size_t calculate_num_parameters() const;

    Qwen2Config config_;
    std::unique_ptr<ModelWeights> weights_;
    size_t num_params_;
};

} // namespace zedinfer::model