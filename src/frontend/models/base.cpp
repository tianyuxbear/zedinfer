#include "frontend/loader/safetensors.hpp"
#include "frontend/models/qwen2.hpp"
#include "frontend/models/qwen3.hpp"
#include "zedinfer.h"
#ifdef DEBUG
#include "utils/system_info.hpp"
#endif

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <plog/Log.h>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace zedinfer::model {

// Parse model config and weights, then instantiate the corresponding model.
std::shared_ptr<Model> Model::parse(const std::string &model_path, zedinferDeviceType_t target_device) {
    // Load model configuration
    std::string config_path = (fs::path(model_path) / "config.json").string();
    auto config = load_config(config_path);

    // Load model weights from SafeTensors files
    auto weights = load_weights(model_path, target_device);

    if (target_device == ZEDINFER_DEVICE_CPU) {
        config->torch_dtype = "float32";
    }

    // Handle tied word embeddings: reuse embed_tokens.weight as lm_head.weight
    if (config->tie_word_embeddings && !weights->has_tensor("lm_head.weight")) {
        if (weights->has_tensor("embed_tokens.weight")) {
            weights->add_tensor("lm_head.weight", weights->get_tensor("embed_tokens.weight"));
            LOGI << "[Model] tie_word_embeddings=true: aliased embed_tokens.weight -> lm_head.weight";
        }
    }

    // Construct and return the model instance
    if (config->model_type == "qwen2") {
        auto *qwen2_config = dynamic_cast<Qwen2Config *>(config.get());
        if (!qwen2_config) {
            throw std::logic_error("Config is not Qwen2Config");
        }
        return std::make_shared<Qwen2Model>(*qwen2_config, std::move(weights));
    } else if (config->model_type == "qwen3") {
        auto *qwen3_config = dynamic_cast<Qwen3Config *>(config.get());
        if (!qwen3_config) {
            throw std::logic_error("Config is not Qwen3Config");
        }
        return std::make_shared<Qwen3Model>(*qwen3_config, std::move(weights));
    }

    throw std::runtime_error("Unsupported model type: " + config->model_type);
}

// Safe string reader: returns default if key missing, null, or non-string type.
static std::string safe_string(const json &j, const std::string &key, const std::string &def) {
    if (!j.contains(key) || !j[key].is_string()) return def;
    return j[key].get<std::string>();
}

// Populate common config fields from JSON.
void Model::load_base_config(ModelConfig &config, const json &j) {
    config.model_type = safe_string(j, "model_type", "unknown");
    config.hidden_act = safe_string(j, "hidden_act", "silu");
    config.torch_dtype = safe_string(j, "torch_dtype", "bfloat16");

    config.bos_token_id = j.value("bos_token_id", 151643);

    // eos_token_id can be int or array of ints in config.json
    if (j.contains("eos_token_id")) {
        if (j["eos_token_id"].is_array()) {
            for (const auto &id : j["eos_token_id"]) {
                config.eos_token_ids.push_back(id.get<int>());
            }
        } else {
            config.eos_token_ids.push_back(j["eos_token_id"].get<int>());
        }
    } else {
        config.eos_token_ids.push_back(151643);
    }
    config.eos_token_id = config.eos_token_ids.empty() ? 151643 : config.eos_token_ids[0];

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
std::unique_ptr<ModelConfig> Model::load_config(const std::string &config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    json j = json::parse(f);
    std::string model_type = safe_string(j, "model_type", "unknown");

    ModelConfig base_config;
    load_base_config(base_config, j);

    if (model_type == "qwen2") {
        auto qwen2_config = std::make_unique<Qwen2Config>(base_config);
        qwen2_config->sliding_window = (j.contains("sliding_window") && j["sliding_window"].is_number())
            ? j["sliding_window"].get<int>() : 4096;
        qwen2_config->max_window_layers = j.value("max_window_layers", 21);
        qwen2_config->use_sliding_window = j.value("use_sliding_window", false);
        return qwen2_config;
    } else if (model_type == "qwen3") {
        auto qwen3_config = std::make_unique<Qwen3Config>(base_config);
        qwen3_config->sliding_window = 0;
        qwen3_config->max_window_layers = j.value("max_window_layers", 36);
        qwen3_config->use_sliding_window = j.value("use_sliding_window", false);
        return qwen3_config;
    }

    throw std::runtime_error("Unsupported model type: " + model_type);
}

// Load model weights using memory-mapped SafeTensors.
std::unique_ptr<ModelWeights> Model::load_weights(const std::string &model_path, zedinferDeviceType_t target_device) {
    auto load_start = std::chrono::high_resolution_clock::now();

    auto loader = zedinfer::loader::SafeTensorsLoader::create(model_path);

    auto mmap_end = std::chrono::high_resolution_clock::now();
    auto mmap_time = std::chrono::duration<double>(mmap_end - load_start).count();
    LOGI.printf("⏱️  Mmap time: %.4fs", mmap_time);

    auto weights = std::make_unique<ModelWeights>();

    size_t converted_count = 0;
    auto convert_start = std::chrono::high_resolution_clock::now();

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
            ZEDINFER_DEVICE_CPU,
            0,
            true, // is_mmap
            const_cast<std::byte *>(static_cast<const std::byte *>(data_ptr)));

        if (target_device == ZEDINFER_DEVICE_CPU) {
            tensor = tensor->to(ZEDINFER_DTYPE_F32);
            converted_count++;
        } else {
            tensor = tensor->to(target_device, 0);
        }

        weights->add_tensor(mapped_name, tensor);
    }
    auto convert_end = std::chrono::high_resolution_clock::now();
    auto convert_time = std::chrono::duration<double>(convert_end - convert_start).count();
    LOGI.printf("⏱️  Conversion time: %.4fs (%zu tensors)", convert_time, converted_count);

#ifdef DEBUG
    LOGD << utils::get_numa_maps_info();
#endif

    return weights;
}

// Normalize weight names by stripping common prefixes (e.g., "model.").
std::string Model::map_weight_name(const std::string &raw_name) {
    if (raw_name.substr(0, 6) == "model.") {
        return raw_name.substr(6);
    }
    return raw_name;
}

} // namespace zedinfer::model