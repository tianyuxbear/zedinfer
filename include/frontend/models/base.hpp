#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace zedinfer {
struct ExecutorConfig;
namespace kvcache {
class KVCache;
}
} // namespace zedinfer

namespace zedinfer::model {

using json = nlohmann::json;

// Configuration parameters for a neural language model.
struct ModelConfig {
    std::vector<std::string> architectures; // Model architecture identifiers
    std::string model_type;                 // Type of the model (e.g., "llama")
    std::string hidden_act;                 // Activation function used in hidden layers
    std::string torch_dtype;                // Data type used in original PyTorch weights

    size_t bos_token_id; // Beginning-of-sequence token ID
    size_t eos_token_id; // End-of-sequence token size_t

    size_t hidden_size;             // Dimensionality of hidden states
    size_t intermediate_size;       // Size of feed-forward network
    size_t vocab_size;              // Size of vocabulary
    size_t max_position_embeddings; // Maximum sequence length supportsize_t

    size_t num_hidden_layers;   // Number of transformer layers
    size_t num_attention_heads; // Number of attention heads
    size_t num_key_value_heads; // Number of key/value heads (for GQA)

    float rms_norm_eps;       // Epsilon for RMS normalization
    float rope_theta;         // Base for rotary position embeddings
    bool tie_word_embeddings; // Whether input/output embeddings are tied

    virtual ~ModelConfig() = default;
};

// Container for model weights, mapping tensor names to tensors.
class ModelWeights {
public:
    void add_tensor(const std::string &name, tensor_t tensor) {
        weights_[name] = tensor;
    }

    tensor_t get_tensor(const std::string &name) const {
        auto it = weights_.find(name);
        if (it == weights_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    bool has_tensor(const std::string &name) const {
        return weights_.find(name) != weights_.end();
    }

    const auto &get_all_weights() const { return weights_; }

private:
    std::unordered_map<std::string, tensor_t> weights_;
};

// Abstract base class for large language models.
class Model {
public:
    virtual ~Model() = default;

    virtual const ModelConfig &config() const = 0;
    virtual const ModelWeights &weights() const = 0;

    // Weight name accessors for graph construction
    virtual std::string get_embedding_weight_name() const = 0;
    virtual std::string get_output_norm_weight_name() const = 0;
    virtual std::string get_output_weight_name() const = 0;
    virtual std::vector<std::string> get_layer_weight_names(int layer_idx) const = 0;

    // Model metadata
    virtual std::string model_type() const = 0;
    virtual size_t num_parameters() const = 0;

    /**
     * Direct forward pass.
     * @param input_ids   Token IDs (prefill: many tokens, decode: 1 token)
     * @param past_len    Number of tokens already in kvcache
     * @param kvcache     Session KV cache (read history, write new K/V)
     * @param exec_config Device, dtype, max_seq_len
     * @return Logits tensor [seq_len, vocab_size]
     */
    virtual tensor_t forward(
        const std::vector<int> &input_ids,
        int past_len,
        kvcache::KVCache &kvcache,
        const ExecutorConfig &exec_config) = 0;

private:
    std::string model_path_;

public:
    // Parses model directory and returns a concrete Model instance.
    static std::shared_ptr<Model> parse(const std::string &model_path, zedinferDeviceType_t target_device);

    // Populates ModelConfig from parsed JSON.
    static void load_base_config(ModelConfig &config, const json &j);

    // Loads and parses config.json into a ModelConfig.
    static std::unique_ptr<ModelConfig> load_config(const std::string &config_path);

    // Loads model weights from files in the given path.
    static std::unique_ptr<ModelWeights> load_weights(const std::string &model_path, zedinferDeviceType_t target_device);

    // Normalizes raw weight names (e.g., strips "model." prefix).
    static std::string map_weight_name(const std::string &raw_name);
};

} // namespace zedinfer::model