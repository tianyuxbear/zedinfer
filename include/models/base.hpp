#pragma once

#include "backend/tensor/tensor.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace neollm::model {

// Configuration parameters for a neural language model.
struct ModelConfig {
    std::vector<std::string> architectures; // Model architecture identifiers
    std::string model_type;                 // Type of the model (e.g., "llama")
    std::string hidden_act;                 // Activation function used in hidden layers
    std::string torch_dtype;                // Data type used in original PyTorch weights

    int bos_token_id; // Beginning-of-sequence token ID
    int eos_token_id; // End-of-sequence token ID

    int hidden_size;             // Dimensionality of hidden states
    int intermediate_size;       // Size of feed-forward network
    int vocab_size;              // Size of vocabulary
    int max_position_embeddings; // Maximum sequence length supported

    int num_hidden_layers;   // Number of transformer layers
    int num_attention_heads; // Number of attention heads
    int num_key_value_heads; // Number of key/value heads (for GQA)

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
};

} // namespace neollm::model