#include "frontend/loader/safetensors.hpp"
#include "frontend/models/qwen2.hpp"
#include "frontend/models/qwen3.hpp"
#include "zedinfer.h"
#ifdef DEBUG
#include "utils/system_info.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <plog/Log.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace zedinfer::model {

// Parse model config and weights, then instantiate the corresponding model.
std::shared_ptr<Model> Model::parse(const std::string& model_path, zedinferDeviceType_t target_device) {
    // Load model configuration
    std::string config_path = (fs::path(model_path) / "config.json").string();
    auto config = load_config(config_path);

    // Load model weights from SafeTensors files
    auto weights = load_weights(model_path, target_device, *config);

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
        auto* qwen2_config = dynamic_cast<Qwen2Config*>(config.get());
        if (!qwen2_config) {
            throw std::logic_error("Config is not Qwen2Config");
        }
        return std::make_shared<Qwen2Model>(*qwen2_config, std::move(weights));
    } else if (config->model_type == "qwen3") {
        auto* qwen3_config = dynamic_cast<Qwen3Config*>(config.get());
        if (!qwen3_config) {
            throw std::logic_error("Config is not Qwen3Config");
        }
        return std::make_shared<Qwen3Model>(*qwen3_config, std::move(weights));
    }

    throw std::runtime_error("Unsupported model type: " + config->model_type);
}

// Safe string reader: returns default if key missing, null, or non-string type.
static std::string safe_string(const json& j, const std::string& key, const std::string& def) {
    if (!j.contains(key) || !j[key].is_string()) {
        return def;
    }
    return j[key].get<std::string>();
}

static bool parse_json_bool(const json &value, bool default_value = false) {
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int>() != 0;
    }
    if (value.is_string()) {
        const std::string flag = value.get<std::string>();
        return (flag == "true" || flag == "yes" || flag == "1" || flag == "group");
    }
    return default_value;
}

static std::string parse_json_string(const json &obj, const char *key, const std::string &default_value = "") {
    if (!obj.contains(key) || obj[key].is_null()) {
        return default_value;
    }
    if (obj[key].is_string()) {
        return obj[key].get<std::string>();
    }
    return default_value;
}

static int get_env_int(const char *name, int default_value = -1) {
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }

    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

static bool is_valid_nvidia_int8_act_group_size(int group_size, size_t K) {
    return group_size > 0 &&
           group_size < static_cast<int>(K) &&
           (K % static_cast<size_t>(group_size)) == 0 &&
           (group_size % 64) == 0;
}

static std::string normalize_quant_strategy(std::string strategy) {
    for (char &c : strategy) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return strategy;
}

static int select_default_nvidia_int8_act_group_size(size_t K) {
    constexpr int candidates[] = {256, 128};
    for (int group_size : candidates) {
        if (is_valid_nvidia_int8_act_group_size(group_size, K)) {
            return group_size;
        }
    }
    return -1;
}

static int resolve_nvidia_int8_act_group_size_from_config(const QuantizationConfig &quant_config, size_t K) {
    const QuantParam &act = quant_config.activations;
    if (act.raw.is_null() || act.num_bits != 8) {
        return 0;
    }

    const std::string strategy = normalize_quant_strategy(act.strategy);
    if (act.group_size > 0) {
        static bool warned_invalid_config_group = false;
        if (!is_valid_nvidia_int8_act_group_size(act.group_size, K)) {
            if (!warned_invalid_config_group) {
                LOGW << "[Model] Ignoring quantization_config.input_activations.group_size="
                     << act.group_size
                     << " for NVIDIA INT8 activation grouping; require 0 < group_size < K, "
                        "K % group_size == 0, and group_size % 64 == 0";
                warned_invalid_config_group = true;
            }
            return 0;
        }
        return act.group_size;
    }

    if (strategy == "token" || strategy == "tensor" || strategy == "row") {
        return -1;
    }

    return 0;
}

static int resolve_nvidia_int8_act_group_size(const QuantizationConfig &quant_config, size_t K) {
    constexpr const char *kEnvName = "ZEDINFER_NVIDIA_INT8_ACT_GROUP_SIZE";
    int group_size = get_env_int(kEnvName, 0);
    if (group_size > 0) {
        static bool warned_invalid = false;
        if (!is_valid_nvidia_int8_act_group_size(group_size, K)) {
            if (!warned_invalid) {
                LOGW << "[Model] Ignoring " << kEnvName << "=" << group_size
                     << " for INT8 GPU grouped activation quantization; require 0 < group_size < K, "
                        "K % group_size == 0, and group_size % 64 == 0";
                warned_invalid = true;
            }
            return select_default_nvidia_int8_act_group_size(K);
        }
        return group_size;
    }

    int config_group_size =
        resolve_nvidia_int8_act_group_size_from_config(quant_config, K);
    if (config_group_size != 0) {
        return config_group_size;
    }

    return select_default_nvidia_int8_act_group_size(K);
}

static int get_nvidia_int8_act_group_size(const QuantizationConfig &quant_config, size_t K) {
    int group_size = resolve_nvidia_int8_act_group_size(quant_config, K);
    if (group_size <= 0) {
        return -1;
    }
    return group_size;
}

static tensor_t expand_rowwise_scale_tensor(tensor_t scale, size_t rows, int num_groups) {
    if (!scale || num_groups <= 1) {
        return scale;
    }

    if (scale->numel() != rows) {
        return scale;
    }

    auto expanded = Tensor::create(
        {rows, static_cast<size_t>(num_groups)},
        scale->dtype(),
        ZEDINFER_DEVICE_CPU);
    const size_t elem_size = scale->elementSize();
    const std::byte *src = scale->data();
    std::byte *dst = expanded->data();

    for (size_t row = 0; row < rows; ++row) {
        const std::byte *row_scale = src + row * elem_size;
        for (int g = 0; g < num_groups; ++g) {
            std::memcpy(
                dst + (row * static_cast<size_t>(num_groups) + static_cast<size_t>(g)) * elem_size,
                row_scale,
                elem_size);
        }
    }

    return expanded;
}

static void parse_quant_param(const json &j, QuantParam &param) {
    if (j.is_null()) {
        return;
    }

    param.raw = j;
    param.num_bits = j.value("num_bits", 0);
    param.symmetric = j.value("symmetric", false);
    param.dynamic = j.value("dynamic", false);
    param.strategy = parse_json_string(j, "strategy", "unknown");
    param.observer = parse_json_string(j, "observer", "");
    param.value_type = parse_json_string(j, "type", "");

    if (j.contains("block_structure")) {
        param.block_structure = j["block_structure"];
    }
    if (j.contains("observer_kwargs")) {
        param.observer_kwargs = j["observer_kwargs"];
    }

    if (j.contains("group_size") && !j["group_size"].is_null()) {
        param.group_size = j["group_size"].get<int>();
    } else {
        param.group_size = -1;
    }

    bool act_order_assigned = false;
    if (j.contains("actorder") && !j["actorder"].is_null()) {
        param.act_order = parse_json_bool(j["actorder"], false);
        act_order_assigned = true;
    }
    if (!act_order_assigned &&
        j.contains("desc_act") &&
        !j["desc_act"].is_null()) {
        param.act_order = parse_json_bool(j["desc_act"], false);
    }
}

static bool find_primary_group(const json &groups, json &target_group) {
    if (!groups.is_object()) {
        return false;
    }

    if (groups.contains("group_0") && groups["group_0"].is_object()) {
        target_group = groups["group_0"];
        return true;
    }

    for (const auto &[key, val] : groups.items()) {
        (void)key;
        if (!val.is_object()) {
            continue;
        }
        if (val.contains("weights") ||
            val.contains("input_activations") ||
            val.contains("output_activations")) {
            target_group = val;
            return true;
        }
    }

    return false;
}

// Populate common config fields from JSON.
void Model::load_base_config(ModelConfig& config, const json& j) {
    config.architectures = j.value("architectures", std::vector<std::string>{});
    config.model_type = safe_string(j, "model_type", "unknown");
    config.hidden_act = safe_string(j, "hidden_act", "silu");
    config.torch_dtype = safe_string(j, "torch_dtype", "bfloat16");

    config.bos_token_id = j.value("bos_token_id", 151643);

    // eos_token_id can be int or array of ints in config.json
    if (j.contains("eos_token_id")) {
        if (j["eos_token_id"].is_array()) {
            for (const auto& id : j["eos_token_id"]) { config.eos_token_ids.push_back(id.get<int>()); }
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
    config.num_key_value_heads = j.value("num_key_value_heads", config.num_attention_heads);

    config.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    config.rope_theta = j.value("rope_theta", 10000.0f);
    config.tie_word_embeddings = j.value("tie_word_embeddings", false);
}

// Load and parse config.json into a model-specific config object.
std::unique_ptr<ModelConfig> Model::load_config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    json j = json::parse(f);
    std::string model_type = safe_string(j, "model_type", "unknown");

    ModelConfig base_config;
    load_base_config(base_config, j);

    if (j.contains("quantization_config") && j["quantization_config"].is_object()) {
        auto &q_json = j["quantization_config"];
        base_config.quant_config.enabled = true;
        base_config.quant_config.raw = q_json;
        base_config.quant_config.quant_method =
            parse_json_string(q_json, "quant_method", "unknown");
        base_config.quant_config.format =
            parse_json_string(q_json, "format", "unknown");
        base_config.quant_config.quantization_status =
            parse_json_string(q_json, "quantization_status", "");

        if (q_json.contains("global_compression_ratio")) {
            base_config.quant_config.global_compression_ratio =
                q_json["global_compression_ratio"];
        }
        if (q_json.contains("kv_cache_scheme")) {
            base_config.quant_config.kv_cache_scheme = q_json["kv_cache_scheme"];
        }
        if (q_json.contains("ignore") && q_json["ignore"].is_array()) {
            base_config.quant_config.ignored_layers =
                q_json["ignore"].get<std::vector<std::string>>();
        }
        if (q_json.contains("config_groups")) {
            base_config.quant_config.config_groups_raw = q_json["config_groups"];
            json target_group;
            bool found =
                find_primary_group(q_json["config_groups"], target_group);
            if (found) {
                if (target_group.contains("weights")) {
                    parse_quant_param(
                        target_group["weights"],
                        base_config.quant_config.weights);
                }
                if (target_group.contains("input_activations")) {
                    parse_quant_param(
                        target_group["input_activations"],
                        base_config.quant_config.activations);
                }
                if (target_group.contains("output_activations")) {
                    parse_quant_param(
                        target_group["output_activations"],
                        base_config.quant_config.output_activations);
                }
                if (target_group.contains("targets")) {
                    base_config.quant_config.target_modules =
                        target_group["targets"].get<std::vector<std::string>>();
                }
            }
        }
    }

    if (model_type == "qwen2") {
        auto qwen2_config = std::make_unique<Qwen2Config>(base_config);
        qwen2_config->sliding_window
            = (j.contains("sliding_window") && j["sliding_window"].is_number()) ? j["sliding_window"].get<int>() : 4096;
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

struct LayerGroup {
    std::string prefix;
    std::string packed_name;
    std::string weight_name;
    std::string scale_name;
    std::string g_idx_name;
    std::string bias_name;

    tensor_t t_packed = nullptr;
    tensor_t t_weight = nullptr;
    tensor_t t_scale = nullptr;
    tensor_t t_g_idx = nullptr;
    tensor_t t_bias = nullptr;
};

static void unpack_4bit_row(const int32_t *packed, int8_t *unpacked, size_t K) {
    for (size_t k_blk = 0; k_blk < K / 8; ++k_blk) {
        int32_t val = packed[k_blk];
        for (int i = 0; i < 8; ++i) {
            unpacked[k_blk * 8 + static_cast<size_t>(i)] =
                static_cast<int8_t>((val >> (i * 4)) & 0xF);
        }
    }
}

static void repack_4bit_row(const int8_t *unpacked, int32_t *packed, size_t K) {
    std::memset(packed, 0, (K / 8) * sizeof(int32_t));
    for (size_t k_blk = 0; k_blk < K / 8; ++k_blk) {
        int32_t val = 0;
        for (int i = 0; i < 8; ++i) {
            val |= (static_cast<int32_t>(
                        unpacked[k_blk * 8 + static_cast<size_t>(i)] & 0xF)
                    << (i * 4));
        }
        packed[k_blk] = val;
    }
}

// Load model weights using memory-mapped SafeTensors.
std::unique_ptr<ModelWeights> Model::load_weights(const std::string& model_path,
                                                  zedinferDeviceType_t target_device,
                                                  const ModelConfig& config) {
    auto load_start = std::chrono::high_resolution_clock::now();

    auto loader_unique = zedinfer::loader::SafeTensorsLoader::create(model_path);
    std::shared_ptr<zedinfer::loader::IModelLoader> loader(std::move(loader_unique));

    auto mmap_end = std::chrono::high_resolution_clock::now();
    auto mmap_time = std::chrono::duration<double>(mmap_end - load_start).count();
    LOGI.printf("⏱️  Mmap time: %.4fs", mmap_time);

    auto weights = std::make_unique<ModelWeights>();
    weights->retain_resource(loader);
    std::map<std::string, LayerGroup> layer_groups;
    std::vector<std::pair<std::string, tensor_t>> standalone_tensors;
    auto convert_start = std::chrono::high_resolution_clock::now();

    for (const auto& raw_name : loader->get_all_tensor_names()) {
        std::string mapped_name = map_weight_name(raw_name);
        if (mapped_name.find("_shape") != std::string::npos) {
            continue;
        }

        auto* info = loader->get_tensor_info(raw_name);
        if (!info) {
            throw std::runtime_error("Failed to get tensor info: " + raw_name);
        }

        const void* data_ptr = loader->get_tensor_data(raw_name);
        if (!data_ptr) {
            throw std::runtime_error("Failed to get tensor data: " + raw_name);
        }

        // Create CPU-resident, mmap-backed tensor
        auto tensor = Tensor::create(info->shape, info->dtype, ZEDINFER_DEVICE_CPU, 0,
                                     true, // is_mmap
                                     const_cast<std::byte*>(static_cast<const std::byte*>(data_ptr)));

        std::string prefix;
        if (mapped_name.size() > 14 &&
            mapped_name.substr(mapped_name.size() - 14) == ".weight_packed") {
            prefix = mapped_name.substr(0, mapped_name.size() - 14);
            layer_groups[prefix].packed_name = mapped_name;
            layer_groups[prefix].t_packed = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 7 &&
                   mapped_name.substr(mapped_name.size() - 7) == ".weight") {
            prefix = mapped_name.substr(0, mapped_name.size() - 7);
            layer_groups[prefix].weight_name = mapped_name;
            layer_groups[prefix].t_weight = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 13 &&
                   mapped_name.substr(mapped_name.size() - 13) == ".weight_scale") {
            prefix = mapped_name.substr(0, mapped_name.size() - 13);
            layer_groups[prefix].scale_name = mapped_name;
            layer_groups[prefix].t_scale = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 13 &&
                   mapped_name.substr(mapped_name.size() - 13) == ".weight_g_idx") {
            prefix = mapped_name.substr(0, mapped_name.size() - 13);
            layer_groups[prefix].g_idx_name = mapped_name;
            layer_groups[prefix].t_g_idx = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 5 &&
                   mapped_name.substr(mapped_name.size() - 5) == ".bias") {
            prefix = mapped_name.substr(0, mapped_name.size() - 5);
            layer_groups[prefix].bias_name = mapped_name;
            layer_groups[prefix].t_bias = tensor;
            layer_groups[prefix].prefix = prefix;
        } else {
            standalone_tensors.push_back({mapped_name, tensor});
        }
    }

    size_t converted_count = 0;

    for (auto &pair : standalone_tensors) {
        tensor_t tensor = pair.second;
        if (target_device == ZEDINFER_DEVICE_CPU &&
            (tensor->dtype() == ZEDINFER_DTYPE_BF16 ||
             tensor->dtype() == ZEDINFER_DTYPE_F16)) {
            tensor = tensor->to(ZEDINFER_DTYPE_F32);
            converted_count++;
        } else if (target_device != ZEDINFER_DEVICE_CPU) {
            tensor = tensor->to(target_device, 0);
        }
        weights->add_tensor(pair.first, tensor);
    }

    for (auto &[prefix, group] : layer_groups) {
        tensor_t current_weight = group.t_packed ? group.t_packed : group.t_weight;

        if (current_weight && group.t_g_idx) {
            if (!group.t_packed) {
                throw std::runtime_error(
                    "Act-order reordering requires packed weights: " + prefix);
            }

            LOGI.printf("🔄 Reordering weights for %s (Act-Order detected)", prefix.c_str());

            const int32_t *g_idx_ptr =
                reinterpret_cast<const int32_t *>(group.t_g_idx->data());
            const int32_t *packed_ptr =
                reinterpret_cast<const int32_t *>(group.t_packed->data());

            size_t N = group.t_packed->shape()[0];
            size_t K_packed = group.t_packed->shape()[1];
            size_t K = group.t_g_idx->shape()[0];

            if (K != K_packed * 8) {
                throw std::runtime_error("Shape mismatch in packing: " + prefix);
            }

            std::vector<int32_t> perm(K);
            std::iota(perm.begin(), perm.end(), 0);
            std::stable_sort(
                perm.begin(),
                perm.end(),
                [&](int32_t a, int32_t b) { return g_idx_ptr[a] < g_idx_ptr[b]; });

            std::vector<int32_t> new_packed_data(N * (K / 8));
            std::vector<int8_t> row_unpacked(K);
            std::vector<int8_t> row_permuted(K);

            for (size_t n = 0; n < N; ++n) {
                const int32_t *src_row = packed_ptr + n * (K / 8);
                int32_t *dst_row = new_packed_data.data() + n * (K / 8);
                unpack_4bit_row(src_row, row_unpacked.data(), K);
                for (size_t k = 0; k < K; ++k) {
                    row_permuted[k] = row_unpacked[static_cast<size_t>(perm[k])];
                }
                repack_4bit_row(row_permuted.data(), dst_row, K);
            }

            auto t_packed_new = Tensor::create(
                group.t_packed->shape(),
                ZEDINFER_DTYPE_I32,
                ZEDINFER_DEVICE_CPU);
            std::memcpy(
                t_packed_new->data(),
                new_packed_data.data(),
                new_packed_data.size() * sizeof(int32_t));
            group.t_packed = t_packed_new;

            auto t_perm = Tensor::create({K}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU);
            std::memcpy(
                t_perm->data(),
                perm.data(),
                perm.size() * sizeof(int32_t));
            group.t_g_idx = t_perm;
        }

        if (current_weight &&
            config.quant_config.enabled &&
            config.quant_config.weights.num_bits == 8 &&
            config.quant_config.weights.value_type == "int") {
            if (current_weight->dtype() == ZEDINFER_DTYPE_I8 ||
                current_weight->dtype() == ZEDINFER_DTYPE_U8) {
                size_t N = current_weight->shape()[0];
                size_t K = current_weight->shape()[1];
                auto t_packed_new = Tensor::create(
                    {N, K / 4},
                    ZEDINFER_DTYPE_I32,
                    ZEDINFER_DEVICE_CPU);
                std::memcpy(t_packed_new->data(), current_weight->data(), N * K);
                group.t_packed = t_packed_new;
                group.packed_name = prefix + ".weight_packed";
                group.t_weight = nullptr;
            } else if (current_weight->dtype() == ZEDINFER_DTYPE_I32) {
                size_t N = current_weight->shape()[0];
                size_t K_packed = current_weight->shape()[1];
                auto t_packed_new = Tensor::create(
                    {N, K_packed},
                    ZEDINFER_DTYPE_I32,
                    ZEDINFER_DEVICE_CPU);

                const uint8_t *src =
                    reinterpret_cast<const uint8_t *>(current_weight->data());
                uint8_t *dst =
                    reinterpret_cast<uint8_t *>(t_packed_new->data());
                const size_t total_bytes = N * K_packed * sizeof(int32_t);

                for (size_t i = 0; i < total_bytes; ++i) {
                    dst[i] = src[i] ^ 0x80;
                }

                group.t_packed = t_packed_new;
                group.packed_name = prefix + ".weight_packed";
                group.t_weight = nullptr;
            }
        }

        if (target_device == ZEDINFER_DEVICE_NVIDIA &&
            current_weight &&
            group.t_scale &&
            config.quant_config.enabled &&
            config.quant_config.weights.num_bits == 8 &&
            config.quant_config.weights.group_size < 0) {
            size_t logical_k = 0;
            if (current_weight->dtype() == ZEDINFER_DTYPE_I8 ||
                current_weight->dtype() == ZEDINFER_DTYPE_U8) {
                logical_k = current_weight->shape()[1];
            } else if (current_weight->dtype() == ZEDINFER_DTYPE_I32) {
                logical_k = current_weight->shape()[1] * 4;
            }

            int act_group_size =
                get_nvidia_int8_act_group_size(config.quant_config, logical_k);
            if (act_group_size > 0) {
                const size_t num_groups =
                    logical_k / static_cast<size_t>(act_group_size);
                group.t_scale = expand_rowwise_scale_tensor(
                    group.t_scale,
                    current_weight->shape()[0],
                    static_cast<int>(num_groups));
            }
        }

        auto process_and_add = [&](tensor_t &tensor, const std::string &name) {
            if (!tensor) {
                return;
            }
            if (target_device == ZEDINFER_DEVICE_CPU &&
                (tensor->dtype() == ZEDINFER_DTYPE_BF16 ||
                 tensor->dtype() == ZEDINFER_DTYPE_F16)) {
                tensor = tensor->to(ZEDINFER_DTYPE_F32);
                converted_count++;
            } else if (target_device != ZEDINFER_DEVICE_CPU) {
                tensor = tensor->to(target_device, 0);
            }
            weights->add_tensor(name, tensor);
        };

        process_and_add(group.t_weight, group.weight_name);
        process_and_add(group.t_packed, group.packed_name);
        process_and_add(group.t_scale, group.scale_name);
        process_and_add(group.t_bias, group.bias_name);
        if (group.t_g_idx) {
            process_and_add(group.t_g_idx, group.g_idx_name);
        }
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
std::string Model::map_weight_name(const std::string& raw_name) {
    if (raw_name.substr(0, 6) == "model.") {
        return raw_name.substr(6);
    }
    return raw_name;
}

} // namespace zedinfer::model
