#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace zedinfer {
struct ExecutorConfig;
}

namespace zedinfer::model {

using json = nlohmann::json;

struct ModelForwardConfig;

struct QuantParam {
    int num_bits = 0;
    int group_size = -1;
    bool symmetric = false;
    bool dynamic = false;
    bool act_order = false;
    std::string strategy;
    std::string observer;
    std::string value_type;
    json block_structure;
    json observer_kwargs;
    json raw;
};

struct QuantizationConfig {
    bool enabled = false;
    std::string quant_method;
    std::string format;
    std::string quantization_status;
    json global_compression_ratio;
    json kv_cache_scheme;

    QuantParam weights;
    QuantParam activations;
    QuantParam output_activations;

    std::vector<std::string> ignored_layers;
    std::vector<std::string> target_modules;
    json config_groups_raw;
    json raw;

    bool is_layer_ignored(const std::string& layer_name) const {
        for (const auto& name : ignored_layers) {
            if (layer_name.find(name) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

struct ModelConfig {
    std::vector<std::string> architectures;
    std::string model_type;
    std::string hidden_act;
    std::string torch_dtype;

    size_t bos_token_id;
    size_t eos_token_id;
    std::vector<int> eos_token_ids;

    size_t hidden_size;
    size_t intermediate_size;
    size_t vocab_size;
    size_t max_position_embeddings;

    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t head_dim = 0; // 0 = derive from hidden_size/num_attention_heads

    float rms_norm_eps;
    float rope_theta;
    bool tie_word_embeddings;

    QuantizationConfig quant_config;

    virtual ~ModelConfig() = default;
};

class ModelWeights {
public:
    void add_tensor(const std::string& name, tensor_t tensor) { weights_[name] = tensor; }

    tensor_t get_tensor(const std::string& name) const {
        auto it = weights_.find(name);
        if (it == weights_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    bool has_tensor(const std::string& name) const { return weights_.find(name) != weights_.end(); }

    // Remove a tensor from the map (used when extracting experts into ExpertWeights).
    void remove_tensor(const std::string& name) { weights_.erase(name); }

    const auto& get_all_weights() const { return weights_; }

    void retain_resource(std::shared_ptr<void> resource) { resources_.push_back(std::move(resource)); }

private:
    std::unordered_map<std::string, tensor_t> weights_;
    std::vector<std::shared_ptr<void>> resources_;
};

/**
 * Abstract base class for large language models.
 * After R3: Model is a data holder (config + weights) with a forward_config()
 * factory. The forward logic lives in transformer_forward() parameterized by
 * ModelForwardConfig + ForwardContext.
 */
class Model {
public:
    virtual ~Model() = default;

    virtual const ModelConfig& config() const = 0;
    virtual const ModelWeights& weights() const = 0;
    virtual std::string model_type() const = 0;
    virtual size_t num_parameters() const = 0;

    // Produce the forward config for this model family
    virtual ModelForwardConfig forward_config() const = 0;

    // Static factory
    static std::shared_ptr<Model> parse(const std::string& model_path, zedinferDeviceType_t target_device);
    static void load_base_config(ModelConfig& config, const json& j);
    static std::unique_ptr<ModelConfig> load_config(const std::string& config_path);
    static std::unique_ptr<ModelWeights> load_weights(const std::string& model_path, zedinferDeviceType_t target_device,
                                                      const ModelConfig& config);
    static std::string map_weight_name(const std::string& raw_name);

private:
    std::string model_path_;
};

} // namespace zedinfer::model
