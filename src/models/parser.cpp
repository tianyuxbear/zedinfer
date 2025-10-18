#include "models/parser.hpp"
#include "frontend/loader/safetensors.hpp"
#include "models/qwen2.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace neollm::model {

std::unique_ptr<loader::IModelLoader> ModelParser::loader;

// Parse model config and weights, then instantiate the corresponding model.
std::unique_ptr<Model> ModelParser::parse(const std::string &model_path) {
    // Load model configuration
    std::string config_path = (fs::path(model_path) / "config.json").string();
    auto config = load_config(config_path);

    // Load model weights from SafeTensors files
    auto weights = load_weights(model_path);

    // Construct and return the model instance
    if (config->model_type == "qwen2") {
        auto *qwen2_config = dynamic_cast<Qwen2Config *>(config.get());
        if (!qwen2_config) {
            throw std::logic_error("Config is not Qwen2Config");
        }
        return std::make_unique<Qwen2Model>(*qwen2_config, std::move(weights));
    }

    throw std::runtime_error("Unsupported model type: " + config->model_type);
}

// Populate common config fields from JSON.
void ModelParser::load_base_config(ModelConfig &config, const json &j) {
    config.model_type = j.value("model_type", "unknown");
    config.hidden_act = j.value("hidden_act", "silu");
    config.torch_dtype = j.value("torch_dtype", "bfloat16");

    config.bos_token_id = j.value("bos_token_id", 151643);
    config.eos_token_id = j.value("eos_token_id", 151643);

    config.hidden_size = j["hidden_size"];
    config.intermediate_size = j["intermediate_size"];
    config.vocab_size = j["vocab_size"];
    config.max_position_embeddings = j["max_position_embeddings"];

    config.num_hidden_layers = j["num_hidden_layers"];
    config.num_attention_heads = j["num_attention_heads"];
    config.num_key_value_heads = j.value("num_key_value_heads",
                                         config.num_attention_heads);

    config.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    config.rope_theta = j.value("rope_theta", 10000.0f);
    config.tie_word_embeddings = j.value("tie_word_embeddings", false);
}

// Load and parse config.json into a model-specific config object.
std::unique_ptr<ModelConfig> ModelParser::load_config(const std::string &config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    json j = json::parse(f);
    std::string model_type = j.value("model_type", "unknown");

    ModelConfig base_config;
    load_base_config(base_config, j);

    if (model_type == "qwen2") {
        auto qwen2_config = std::make_unique<Qwen2Config>(base_config);
        qwen2_config->sliding_window = j.value("sliding_window", 4096);
        qwen2_config->max_window_layers = j.value("max_window_layers", 21);
        qwen2_config->use_sliding_window = j.value("use_sliding_window", false);
        return qwen2_config;
    }

    throw std::runtime_error("Unsupported model type: " + model_type);
}

// Load model weights using memory-mapped SafeTensors.
std::unique_ptr<ModelWeights> ModelParser::load_weights(const std::string &model_path) {
    loader = neollm::loader::SafeTensorsLoader::create(model_path);
    auto weights = std::make_unique<ModelWeights>();

    for (const auto &raw_name : loader->get_all_tensor_names()) {
        std::string mapped_name = map_weight_name(raw_name);

        auto *info = loader->get_tensor_info(raw_name);
        if (!info) {
            throw std::runtime_error("Failed to get tensor info: " + raw_name);
        }

        const void *data_ptr = loader->get_tensor_data(raw_name);
        if (!data_ptr) {
            throw std::runtime_error("Failed to get tensor data: " + raw_name);
        }

        // Create CPU-resident, mmap-backed tensor
        auto tensor = Tensor::create(
            info->shape,
            info->dtype,
            NEOLLM_DEVICE_CPU,
            0,
            true, // is_mmap
            const_cast<std::byte *>(static_cast<const std::byte *>(data_ptr)));

        weights->add_tensor(mapped_name, tensor);
    }

    return weights;
}

// Normalize weight names by stripping common prefixes (e.g., "model.").
std::string ModelParser::map_weight_name(const std::string &raw_name) {
    if (raw_name.substr(0, 6) == "model.") {
        return raw_name.substr(6);
    }
    return raw_name;
}

} // namespace neollm::model