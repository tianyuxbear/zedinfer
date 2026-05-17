#include "backend/device/runtime_api.hpp"
#include "frontend/loader/safetensors.hpp"
#include "frontend/models/qwen2.hpp"
#include "frontend/models/qwen3.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include "frontend/models/qwen3_moe.hpp"
#include "zedinfer/activation.hpp"
#include "utils/types.hpp"
#include "zedinfer.h"
#ifdef ZEDINFER_USE_TERMBAR
#include "termbar.h"
#endif
#ifdef DEBUG
#include "utils/system_info.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <plog/Log.h>
#include <string>
#include <vector>

#ifdef ZEDINFER_USE_TERMBAR
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace zedinfer::model {

namespace {

// Rough per-expert FFN byte size from config — quant-aware. Used as a heuristic input to
// auto-sizing; precision is not critical because the pool's slot arena allocates from the
// actual loaded tensor shapes, not from this estimate.
size_t estimate_per_expert_bytes(const Qwen3MoEConfig& cfg) {
    const size_t hidden = cfg.hidden_size;
    const size_t inter = cfg.moe_intermediate_size;

    if (cfg.quant_config.enabled && cfg.quant_config.weights.num_bits > 0) {
        const int bits = cfg.quant_config.weights.num_bits;
        const int gs = cfg.quant_config.weights.group_size > 0 ? cfg.quant_config.weights.group_size : 128;
        // Linear [out, in]: packed weight bytes + fp16 group scales. g_idx / qzeros are small.
        auto linear_bytes = [bits, gs](size_t out, size_t in) {
            size_t weight = (out * in * static_cast<size_t>(bits)) / 8;
            size_t scale = out * ((in + static_cast<size_t>(gs) - 1) / static_cast<size_t>(gs)) * 2;
            return weight + scale;
        };
        return linear_bytes(inter, hidden)  // gate_proj
             + linear_bytes(inter, hidden)  // up_proj
             + linear_bytes(hidden, inter); // down_proj
    }

    // Dense: 3 × hidden × inter weights at torch_dtype size.
    const size_t elem_bytes = utils::dsize(utils::str_to_dtype(cfg.torch_dtype));
    return 3 * hidden * inter * elem_bytes;
}

// Decide ExpertPoolConfig for a Qwen3MoE model. Order:
//  1. If ZEDINFER_MOE_GPU_SLOTS is set: parse and respect it.
//  2. Else: auto-size. Apply gpu_memory_utilization the same way init_block_pool does
//     (allowed = total × util; headroom = allowed − used), then carve out a fraction
//     of headroom for experts. Falls back to ALL_GPU when experts fit.
//
// Using the same util-aware formula as init_block_pool prevents the loader from picking
// ALL_GPU on a budget the engine will then refuse to honor (e.g. user passes
// --gpu-memory-utilization 0.5 to fit alongside other workloads).
ExpertPoolConfig compute_moe_pool_config(const Qwen3MoEConfig& cfg, zedinferDeviceType_t device,
                                         float gpu_memory_utilization) {
    ExpertPoolConfig pool; // defaults: ALL_GPU

    if (device == ZEDINFER_DEVICE_CPU) {
        return pool;
    }

    const char* env = std::getenv("ZEDINFER_MOE_GPU_SLOTS");
    if (env != nullptr && *env != '\0') {
        int n = 0;
        try {
            size_t consumed = 0;
            n = std::stoi(env, &consumed);
            if (consumed != std::string(env).size()) {
                throw std::invalid_argument("trailing characters");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("ZEDINFER_MOE_GPU_SLOTS: invalid integer '") + env + "' (" + e.what()
                                     + ")");
        }
        if (n <= 0) {
            throw std::runtime_error("ZEDINFER_MOE_GPU_SLOTS must be positive, got " + std::to_string(n));
        }
        if (static_cast<size_t>(n) >= cfg.num_experts) {
            LOGI.printf("[Model] ZEDINFER_MOE_GPU_SLOTS=%d >= num_experts=%zu; ALL_GPU", n, cfg.num_experts);
            return pool;
        }
        pool.strategy = ExpertPoolStrategy::PINNED_LRU;
        pool.num_gpu_slots = n;
        LOGI.printf("[Model] ZEDINFER_MOE_GPU_SLOTS=%d; PINNED_LRU N=%d of %zu experts/layer", n, n, cfg.num_experts);
        return pool;
    }

    // Auto path. Query VRAM before any allocation.
    auto api = device::getRuntimeAPI(device);
    size_t free_bytes = 0, total_bytes = 0;
    api->get_memory_info(&free_bytes, &total_bytes);

    // Mirror init_block_pool's accounting: "allowed" caps total zedinfer footprint;
    // "headroom" is what's actually available after other processes' usage.
    const size_t used_bytes = (total_bytes > free_bytes) ? (total_bytes - free_bytes) : 0;
    const size_t allowed_bytes = static_cast<size_t>(static_cast<double>(total_bytes) * gpu_memory_utilization);
    const size_t headroom_bytes = (allowed_bytes > used_bytes) ? (allowed_bytes - used_bytes) : 0;

    const size_t per_expert = estimate_per_expert_bytes(cfg);
    const size_t total_expert_bytes = cfg.num_experts * cfg.num_hidden_layers * per_expert;

    // Of the headroom, experts may take up to kExpertVramFraction; the rest covers
    // KV cache + non-expert weights + activations + scratch.
    constexpr double kExpertVramFraction = 0.70;
    const size_t expert_budget = static_cast<size_t>(static_cast<double>(headroom_bytes) * kExpertVramFraction);

    LOGI.printf("[Model] Auto MoE sizing: total=%zu MB, free=%zu MB, allowed=%zu MB (util=%.2f), "
                "headroom=%zu MB, per_expert≈%zu KB, total_experts=%zu MB, budget=%zu MB",
                total_bytes / (1024 * 1024), free_bytes / (1024 * 1024), allowed_bytes / (1024 * 1024),
                gpu_memory_utilization, headroom_bytes / (1024 * 1024), per_expert / 1024,
                total_expert_bytes / (1024 * 1024), expert_budget / (1024 * 1024));

    if (total_expert_bytes <= expert_budget) {
        LOGI.printf("[Model] Auto: experts fit in budget; ALL_GPU");
        return pool;
    }

    // Compute per-layer slot count. Floor at num_experts_per_tok so a single layer's
    // selected experts always fit without eviction within that layer.
    size_t denom = cfg.num_hidden_layers * per_expert;
    int n = denom > 0 ? static_cast<int>(expert_budget / denom) : 0;
    if (n < static_cast<int>(cfg.num_experts_per_tok)) {
        n = static_cast<int>(cfg.num_experts_per_tok);
    }
    if (static_cast<size_t>(n) >= cfg.num_experts) {
        LOGI.printf("[Model] Auto: budget allows all experts; ALL_GPU");
        return pool;
    }

    pool.strategy = ExpertPoolStrategy::PINNED_LRU;
    pool.num_gpu_slots = n;
    LOGI.printf("[Model] Auto: PINNED_LRU N=%d of %zu experts/layer (budget exceeded by %zu MB)", n, cfg.num_experts,
                (total_expert_bytes - expert_budget) / (1024 * 1024));
    return pool;
}
enum class LoadProgressColor {
    Blue,
};

#ifdef ZEDINFER_USE_TERMBAR

static size_t clamp_progress_value(size_t value) {
    return std::min(value, static_cast<size_t>(std::numeric_limits<int>::max()));
}

static bool is_stdout_interactive() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

static bool should_render_load_progress() {
    if (std::getenv("ZEDINFER_DISABLE_LOAD_PROGRESS") != nullptr) {
        return false;
    }
    const char* term = std::getenv("TERM");
    if (term && std::strcmp(term, "dumb") == 0) {
        return false;
    }
    return is_stdout_interactive();
}

static termbar::Color to_termbar_color(LoadProgressColor color) {
    switch (color) {
        case LoadProgressColor::Blue:
        default:
            return termbar::Color::Blue;
    }
}

class ModelLoadProgress {
public:
    ModelLoadProgress() : enabled_(should_render_load_progress()) {}

    void begin_stage(const std::string& label, size_t total_steps, LoadProgressColor color) {
        finish_stage();

        stage_total_ = clamp_progress_value(std::max<size_t>(total_steps, 1));
        current_step_ = 0;
        if (!enabled_) {
            return;
        }

        std::cout << label << std::endl;
        bar_ = std::make_unique<termbar::ProgressBar>(static_cast<int>(stage_total_), to_termbar_color(color));
        bar_->update(0);
    }

    void update(size_t step) {
        current_step_ = stage_total_ == 0 ? 0 : std::min(clamp_progress_value(step), stage_total_);
        if (enabled_ && bar_) {
            bar_->update(static_cast<int>(current_step_));
        }
    }

    void advance() { update(current_step_ + 1); }

    void finish_stage() {
        if (bar_) {
            bar_->finish();
            bar_.reset();
        }
        stage_total_ = 0;
        current_step_ = 0;
    }

private:
    bool enabled_ = false;
    size_t stage_total_ = 0;
    size_t current_step_ = 0;
    std::unique_ptr<termbar::ProgressBar> bar_;
};

#else

class ModelLoadProgress {
public:
    void begin_stage(const std::string&, size_t, LoadProgressColor) {}
    void update(size_t) {}
    void advance() {}
    void finish_stage() {}
};

#endif
} // namespace

// Parse model config and weights, then instantiate the corresponding model.
std::shared_ptr<Model> Model::parse(const std::string& model_path, zedinferDeviceType_t target_device,
                                    float gpu_memory_utilization) {
    // Load model configuration
    std::string config_path = (fs::path(model_path) / "config.json").string();
    auto config = load_config(config_path);

    // For MoE models, decide the ExpertPool strategy up front so the loader's CPU-pinned
    // routing predicate and the pool strategy stay coupled to the same single decision
    // (env var override or auto-sized fallback).
    ExpertPoolConfig moe_pool_config; // defaults: ALL_GPU; only meaningful for qwen3_moe.
    std::function<bool(const std::string&)> route = [](const std::string&) { return false; };
    if (config->model_type == "qwen3_moe" && target_device != ZEDINFER_DEVICE_CPU) {
        auto* moe_config = dynamic_cast<const Qwen3MoEConfig*>(config.get());
        if (moe_config) {
            moe_pool_config = compute_moe_pool_config(*moe_config, target_device, gpu_memory_utilization);
            if (moe_pool_config.strategy == ExpertPoolStrategy::PINNED_LRU) {
                LOGI << "[Model] Routing MoE expert tensors to CPU pinned memory";
                route = [](const std::string& name) { return name.find(".mlp.experts.") != std::string::npos; };
            }
        }
    }

    // Load model weights from SafeTensors files
    auto weights = load_weights(model_path, target_device, *config, route);

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
    } else if (config->model_type == "qwen3_moe") {
        auto* moe_config = dynamic_cast<Qwen3MoEConfig*>(config.get());
        if (!moe_config) {
            throw std::logic_error("Config is not Qwen3MoEConfig");
        }
        return std::make_shared<Qwen3MoEModel>(*moe_config, std::move(weights), moe_pool_config);
    } else if (config->model_type == "qwen3_5") {
        auto* qcfg = dynamic_cast<Qwen3_5Config*>(config.get());
        if (!qcfg) {
            throw std::logic_error("Config is not Qwen3_5Config");
        }
        // M0 default; M5 will revisit this to plumb through SchedulerConfig.
        const int max_concurrent = 1;
        ExecutorConfig exec(target_device, 0, ZEDINFER_DTYPE_BF16);
        return std::make_shared<Qwen3_5Model>(*qcfg, std::move(weights), exec, max_concurrent);
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

static bool parse_json_bool(const json& value, bool default_value = false) {
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

static std::string parse_json_string(const json& obj, const char* key, const std::string& default_value = "") {
    if (!obj.contains(key) || obj[key].is_null()) {
        return default_value;
    }
    if (obj[key].is_string()) {
        return obj[key].get<std::string>();
    }
    return default_value;
}

static int get_env_int(const char* name, int default_value = -1) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

static bool is_valid_nvidia_int8_act_group_size(int group_size, size_t K) {
    return group_size > 0 && group_size < static_cast<int>(K) && (K % static_cast<size_t>(group_size)) == 0
        && (group_size % 64) == 0;
}

static std::string normalize_quant_strategy(std::string strategy) {
    for (char& c : strategy) {
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

static int resolve_nvidia_int8_act_group_size_from_config(const QuantizationConfig& quant_config, size_t K) {
    const QuantParam& act = quant_config.activations;
    if (act.raw.is_null() || act.num_bits != 8) {
        return 0;
    }

    const std::string strategy = normalize_quant_strategy(act.strategy);
    if (act.group_size > 0) {
        static bool warned_invalid_config_group = false;
        if (!is_valid_nvidia_int8_act_group_size(act.group_size, K)) {
            if (!warned_invalid_config_group) {
                LOGW << "[Model] Ignoring quantization_config.input_activations.group_size=" << act.group_size
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

static int resolve_nvidia_int8_act_group_size(const QuantizationConfig& quant_config, size_t K) {
    constexpr const char* kEnvName = "ZEDINFER_NVIDIA_INT8_ACT_GROUP_SIZE";
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

    int config_group_size = resolve_nvidia_int8_act_group_size_from_config(quant_config, K);
    if (config_group_size != 0) {
        return config_group_size;
    }

    return select_default_nvidia_int8_act_group_size(K);
}

static int get_nvidia_int8_act_group_size(const QuantizationConfig& quant_config, size_t K) {
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

    auto expanded = Tensor::create({rows, static_cast<size_t>(num_groups)}, scale->dtype(), ZEDINFER_DEVICE_CPU);
    const size_t elem_size = scale->elementSize();
    const std::byte* src = scale->data();
    std::byte* dst = expanded->data();

    for (size_t row = 0; row < rows; ++row) {
        const std::byte* row_scale = src + row * elem_size;
        for (int g = 0; g < num_groups; ++g) {
            std::memcpy(dst + (row * static_cast<size_t>(num_groups) + static_cast<size_t>(g)) * elem_size, row_scale,
                        elem_size);
        }
    }

    return expanded;
}

static void parse_quant_param(const json& j, QuantParam& param) {
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
    if (!act_order_assigned && j.contains("desc_act") && !j["desc_act"].is_null()) {
        param.act_order = parse_json_bool(j["desc_act"], false);
    }
}

static bool find_primary_group(const json& groups, json& target_group) {
    if (!groups.is_object()) {
        return false;
    }

    if (groups.contains("group_0") && groups["group_0"].is_object()) {
        target_group = groups["group_0"];
        return true;
    }

    for (const auto& [key, val] : groups.items()) {
        (void)key;
        if (!val.is_object()) {
            continue;
        }
        if (val.contains("weights") || val.contains("input_activations") || val.contains("output_activations")) {
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
    // Explicit head_dim from config takes priority; fallback to hidden_size/num_heads
    config.head_dim = j.value("head_dim", static_cast<size_t>(0));
    if (config.head_dim == 0) {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }

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
    // Qwen3.5 nests text-model fields under "text_config" (top-level only carries
    // multimodal token IDs + vision_config + quantization_config). Route the base
    // load through that nested object so hidden_size/num_hidden_layers/etc. are
    // populated from the right slot. The qwen3_5 dispatcher branch below relies on
    // the result and additionally populates hybrid/vision/mrope fields.
    const bool is_qwen3_5 = (model_type == "qwen3_5" || model_type == "qwen3_5_moe");
    if (is_qwen3_5 && j.contains("text_config") && j["text_config"].is_object()) {
        // Make a mutable copy so we can inject safe defaults for fields the dense
        // base loader requires unconditionally (e.g. intermediate_size, which the
        // Qwen3.5-MoE text_config omits because every layer uses experts).
        json text_for_base = j["text_config"];
        if (!text_for_base.contains("intermediate_size")) {
            text_for_base["intermediate_size"]
                = text_for_base.value("moe_intermediate_size", static_cast<size_t>(0));
        }
        load_base_config(base_config, text_for_base);
        // load_base_config copies model_type out of the nested JSON (e.g. "qwen3_5_text"
        // or "qwen3_5_moe_text"); restore the canonical top-level value so the
        // dispatcher and downstream model code see "qwen3_5" / "qwen3_5_moe".
        base_config.model_type = model_type;
    } else {
        load_base_config(base_config, j);
    }

    if (j.contains("quantization_config") && j["quantization_config"].is_object()) {
        auto& q_json = j["quantization_config"];
        base_config.quant_config.enabled = true;
        base_config.quant_config.raw = q_json;
        base_config.quant_config.quant_method = parse_json_string(q_json, "quant_method", "unknown");
        base_config.quant_config.format = parse_json_string(q_json, "format", "unknown");
        base_config.quant_config.quantization_status = parse_json_string(q_json, "quantization_status", "");

        if (q_json.contains("global_compression_ratio")) {
            base_config.quant_config.global_compression_ratio = q_json["global_compression_ratio"];
        }
        if (q_json.contains("kv_cache_scheme")) {
            base_config.quant_config.kv_cache_scheme = q_json["kv_cache_scheme"];
        }
        if (q_json.contains("ignore") && q_json["ignore"].is_array()) {
            base_config.quant_config.ignored_layers = q_json["ignore"].get<std::vector<std::string>>();
        }
        if (q_json.contains("config_groups")) {
            base_config.quant_config.config_groups_raw = q_json["config_groups"];
            json target_group;
            bool found = find_primary_group(q_json["config_groups"], target_group);
            if (found) {
                if (target_group.contains("weights")) {
                    parse_quant_param(target_group["weights"], base_config.quant_config.weights);
                }
                if (target_group.contains("input_activations")) {
                    parse_quant_param(target_group["input_activations"], base_config.quant_config.activations);
                }
                if (target_group.contains("output_activations")) {
                    parse_quant_param(target_group["output_activations"], base_config.quant_config.output_activations);
                }
                if (target_group.contains("targets")) {
                    base_config.quant_config.target_modules = target_group["targets"].get<std::vector<std::string>>();
                }
            }
        }

        // Handle GPTQ flat format (bits/group_size/sym/desc_act)
        if (base_config.quant_config.quant_method == "gptq" && !q_json.contains("config_groups")) {
            auto& wp = base_config.quant_config.weights;
            wp.num_bits = q_json.value("bits", 0);
            wp.group_size = q_json.value("group_size", -1);
            wp.symmetric = q_json.value("sym", false);
            if (q_json.contains("desc_act")) {
                wp.act_order = parse_json_bool(q_json["desc_act"], false);
            }
            LOGI.printf("[Model] GPTQ config: bits=%d, group_size=%d, sym=%s, desc_act=%s", wp.num_bits, wp.group_size,
                        wp.symmetric ? "true" : "false", wp.act_order ? "true" : "false");
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
    } else if (model_type == "qwen3_moe") {
        auto moe_config = std::make_unique<Qwen3MoEConfig>(base_config);
        moe_config->num_experts = j.value("num_experts", 128);
        moe_config->num_experts_per_tok = j.value("num_experts_per_tok", 8);
        moe_config->moe_intermediate_size = j.value("moe_intermediate_size", 1536);
        moe_config->shared_expert_intermediate_size = j.value("shared_expert_intermediate_size", 4096);
        moe_config->decoder_sparse_step = j.value("decoder_sparse_step", 1);
        if (j.contains("mlp_only_layers") && j["mlp_only_layers"].is_array()) {
            moe_config->mlp_only_layers = j["mlp_only_layers"].get<std::vector<int>>();
        }
        moe_config->norm_topk_prob = j.value("norm_topk_prob", true);
        moe_config->max_window_layers = j.value("max_window_layers", 94);
        moe_config->use_sliding_window = j.value("use_sliding_window", false);

        LOGI.printf("[Model] Qwen3MoE: %zu layers, %zu experts (top-%zu), moe_inter=%zu, shared_inter=%zu",
                    moe_config->num_hidden_layers, moe_config->num_experts, moe_config->num_experts_per_tok,
                    moe_config->moe_intermediate_size, moe_config->shared_expert_intermediate_size);
        return moe_config;
    } else if (model_type == "qwen3_5" || model_type == "qwen3_5_moe") {
        // base_config has already been populated from text_config above (see is_qwen3_5
        // dispatch). Here we layer on the hybrid / vision / mrope / linear-attn fields.
        const json& text_json = j.contains("text_config") && j["text_config"].is_object()
                                    ? j["text_config"]
                                    : j;
        const json vision_json = j.value("vision_config", json::object());

        // Hybrid attention layout.
        if (text_json.contains("layer_types") && text_json["layer_types"].is_array()) {
            base_config.layer_types.clear();
            base_config.layer_types.reserve(text_json["layer_types"].size());
            for (const auto& el : text_json["layer_types"]) {
                base_config.layer_types.push_back(el.get<std::string>());
            }
        }
        base_config.attn_output_gate = text_json.value("attn_output_gate", false);
        base_config.mtp_num_hidden_layers = text_json.value("mtp_num_hidden_layers", 0);

        // RoPE parameters (Qwen3.5 nests them under text_config.rope_parameters).
        if (text_json.contains("rope_parameters") && text_json["rope_parameters"].is_object()) {
            const auto& rp = text_json["rope_parameters"];
            base_config.partial_rotary_factor = rp.value("partial_rotary_factor", 1.0f);
            base_config.mrope_interleaved = rp.value("mrope_interleaved", false);
            if (rp.contains("mrope_section") && rp["mrope_section"].is_array()
                && rp["mrope_section"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    base_config.mrope_section[i] = rp["mrope_section"][i].get<int>();
                }
            }
            base_config.rope_theta = rp.value("rope_theta", base_config.rope_theta);
        }

        // Linear attention sub-config (mamba-style state-space layers).
        base_config.linear_attn.num_v_heads = text_json.value("linear_num_value_heads", 0);
        base_config.linear_attn.value_head_dim = text_json.value("linear_value_head_dim", 0);
        base_config.linear_attn.num_k_heads = text_json.value("linear_num_key_heads", 0);
        base_config.linear_attn.key_head_dim = text_json.value("linear_key_head_dim", 0);
        base_config.linear_attn.conv_kernel_dim = text_json.value("linear_conv_kernel_dim", 4);
        base_config.linear_attn.state_dtype = text_json.value("mamba_ssm_dtype", std::string("bfloat16"));
        // d_state defaults to value_head_dim; the loader can override it from the actual
        // in_proj_b tensor shape once weights are read.
        base_config.linear_attn.d_state = base_config.linear_attn.value_head_dim;

        // Vision sub-config (top-level "vision_config", not nested in text_config).
        if (!vision_json.empty() && vision_json.is_object()) {
            base_config.has_vision = true;
            base_config.vision.depth = vision_json.value("depth", 0);
            base_config.vision.hidden_size = vision_json.value("hidden_size", 0);
            base_config.vision.out_hidden_size = vision_json.value("out_hidden_size", 0);
            base_config.vision.num_heads = vision_json.value("num_heads", 0);
            base_config.vision.patch_size = vision_json.value("patch_size", 16);
            base_config.vision.temporal_patch_size = vision_json.value("temporal_patch_size", 2);
            base_config.vision.spatial_merge_size = vision_json.value("spatial_merge_size", 2);
            base_config.vision.num_position_embeddings = vision_json.value("num_position_embeddings", 0);
            base_config.vision.intermediate_size = vision_json.value("intermediate_size", 0);
        }

        // Special-token IDs live at the *top* of the config, not under text_config.
        base_config.image_token_id = j.value("image_token_id", -1);
        base_config.video_token_id = j.value("video_token_id", -1);
        base_config.vision_start_token_id = j.value("vision_start_token_id", -1);
        base_config.vision_end_token_id = j.value("vision_end_token_id", -1);

        if (model_type == "qwen3_5_moe") {
            auto moe_cfg = std::make_unique<Qwen3_5MoEConfig>(std::move(base_config));
            moe_cfg->num_experts = text_json.value("num_experts", 0);
            moe_cfg->num_experts_per_tok = text_json.value("num_experts_per_tok", 0);
            moe_cfg->moe_intermediate_size = text_json.value("moe_intermediate_size", 0);
            moe_cfg->shared_expert_intermediate_size = text_json.value("shared_expert_intermediate_size", 0);
            moe_cfg->decoder_sparse_step = text_json.value("decoder_sparse_step", 1);
            if (text_json.contains("mlp_only_layers") && text_json["mlp_only_layers"].is_array()) {
                moe_cfg->mlp_only_layers = text_json["mlp_only_layers"].get<std::vector<int>>();
            }
            LOGI.printf("[Model] Qwen3.5-MoE: %zu layers, %d experts (top-%d), moe_inter=%d, shared_inter=%d",
                        moe_cfg->num_hidden_layers, moe_cfg->num_experts, moe_cfg->num_experts_per_tok,
                        moe_cfg->moe_intermediate_size, moe_cfg->shared_expert_intermediate_size);
            return moe_cfg;
        }

        auto dense_cfg = std::make_unique<Qwen3_5Config>(std::move(base_config));
        LOGI.printf("[Model] Qwen3.5: %zu layers (hybrid, linear_v_heads=%d), vision=%s",
                    dense_cfg->num_hidden_layers, dense_cfg->linear_attn.num_v_heads,
                    dense_cfg->has_vision ? "true" : "false");
        return dense_cfg;
    }

    throw std::runtime_error("Unsupported model type: " + model_type);
}

struct LayerGroup {
    std::string prefix;
    std::string packed_name;
    std::string weight_name;
    std::string scale_name;
    std::string g_idx_name;
    std::string zeros_name;
    std::string bias_name;

    tensor_t t_packed = nullptr;
    tensor_t t_weight = nullptr;
    tensor_t t_scale = nullptr;
    tensor_t t_g_idx = nullptr;
    tensor_t t_zeros = nullptr;
    tensor_t t_bias = nullptr;
};

// Detect the actual GPTQ zero-point from qzeros.
// AutoGPTQ stores qzeros with a -1 offset (e.g. 7 for actual zp=8); other tools store it directly.
// Env var ZEDINFER_GPTQ_ZEROPOINT overrides auto-detection.
static int detect_gptq_zero_point(const LayerGroup& group, int num_bits) {
    int actual_zero_point = 1 << (num_bits - 1); // symmetric default (8 for INT4)
    if (!group.t_zeros || num_bits != 4) {
        return actual_zero_point;
    }

    int32_t first_qzero = *reinterpret_cast<const int32_t*>(group.t_zeros->data());
    int stored_zp = first_qzero & 0xF;

    const char* zp_env = std::getenv("ZEDINFER_GPTQ_ZEROPOINT");
    if (zp_env) {
        return std::atoi(zp_env);
    }
    // AutoGPTQ convention: stored = actual - 1
    return stored_zp + 1;
}

// Transpose GPTQ qweight from [K/pack, N] to [N, K/pack] and pre-adjust packed nibbles
// so that (adjusted - 8) matches the actual dequant (kernel hardcodes zero_point=8).
static tensor_t transpose_gptq_qweight(tensor_t packed, int zp_adjust) {
    size_t rows = packed->shape()[0];
    size_t cols = packed->shape()[1];
    auto transposed = Tensor::create({cols, rows}, packed->dtype(), ZEDINFER_DEVICE_CPU);
    const int32_t* src = reinterpret_cast<const int32_t*>(packed->data());
    int32_t* dst = reinterpret_cast<int32_t*>(transposed->data());

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            int32_t val = src[r * cols + c];
            if (zp_adjust != 0) {
                int32_t adjusted = 0;
                for (int i = 0; i < 8; ++i) {
                    int nibble = (val >> (i * 4)) & 0xF;
                    nibble = std::max(0, std::min(15, nibble + zp_adjust));
                    adjusted |= (nibble << (i * 4));
                }
                val = adjusted;
            }
            dst[c * rows + r] = val;
        }
    }
    return transposed;
}

// Transpose GPTQ scales from [K/group, N] to [N, K/group] and convert dtype if it doesn't
// match the model activation dtype (kernel checks scale/activation dtype equality).
static tensor_t transpose_gptq_scale(tensor_t scale, zedinferDataType_t model_dtype) {
    size_t rows = scale->shape()[0];
    size_t cols = scale->shape()[1];
    zedinferDataType_t scale_dtype = scale->dtype();
    bool need_convert
        = (scale_dtype != model_dtype) && (model_dtype == ZEDINFER_DTYPE_BF16 || model_dtype == ZEDINFER_DTYPE_F16);

    auto transposed = Tensor::create({cols, rows}, need_convert ? model_dtype : scale_dtype, ZEDINFER_DEVICE_CPU);
    const std::byte* src = scale->data();
    std::byte* dst = transposed->data();
    size_t src_elem = scale->elementSize();
    size_t dst_elem = transposed->elementSize();

    if (!need_convert) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                std::memcpy(dst + (c * rows + r) * src_elem, src + (r * cols + c) * src_elem, src_elem);
            }
        }
        return transposed;
    }

    // Transpose + convert (FP16 ↔ BF16 via FP32 intermediate).
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const std::byte* s = src + (r * cols + c) * src_elem;
            float val = 0.0f;
            if (scale_dtype == ZEDINFER_DTYPE_F16) {
                val = utils::fp16_to_fp32_f16c(*reinterpret_cast<const fp16_t*>(s));
            } else if (scale_dtype == ZEDINFER_DTYPE_BF16) {
                val = utils::_bf16_to_f32(*reinterpret_cast<const bf16_t*>(s));
            }
            std::byte* d = dst + (c * rows + r) * dst_elem;
            if (model_dtype == ZEDINFER_DTYPE_BF16) {
                *reinterpret_cast<bf16_t*>(d) = utils::_f32_to_bf16(val);
            } else if (model_dtype == ZEDINFER_DTYPE_F16) {
                *reinterpret_cast<fp16_t*>(d) = utils::fp32_to_fp16_f16c(val);
            }
        }
    }
    return transposed;
}

// Process a GPTQ quantized layer group in-place: discard unused g_idx, transpose qweight
// with zero-point adjustment, and transpose/convert scales.
static void process_gptq_group(LayerGroup& group, const ModelConfig& config, tensor_t& current_weight) {
    // Discard g_idx when desc_act=false (AutoGPTQ may still ship it).
    if (current_weight && group.t_g_idx && !config.quant_config.weights.act_order) {
        group.t_g_idx = nullptr;
    }

    if (group.t_packed && group.t_packed->shape().size() == 2) {
        int actual_zero_point = detect_gptq_zero_point(group, config.quant_config.weights.num_bits);
        int zp_adjust = 8 - actual_zero_point;
        group.t_packed = transpose_gptq_qweight(group.t_packed, zp_adjust);
        current_weight = group.t_packed;

        static bool zp_logged = false;
        if (!zp_logged) {
            zp_logged = true;
            LOGI.printf("[GPTQ] Zero-point: actual_zp=%d, kernel_zp=8, nibble_adjust=%+d", actual_zero_point,
                        zp_adjust);
        }
    }

    if (group.t_scale && group.t_scale->shape().size() == 2) {
        zedinferDataType_t model_dtype = utils::str_to_dtype(config.torch_dtype);
        group.t_scale = transpose_gptq_scale(group.t_scale, model_dtype);
    }
}

static void unpack_4bit_row(const int32_t* packed, int8_t* unpacked, size_t K) {
    for (size_t k_blk = 0; k_blk < K / 8; ++k_blk) {
        int32_t val = packed[k_blk];
        for (int i = 0; i < 8; ++i) {
            unpacked[k_blk * 8 + static_cast<size_t>(i)] = static_cast<int8_t>((val >> (i * 4)) & 0xF);
        }
    }
}

static void repack_4bit_row(const int8_t* unpacked, int32_t* packed, size_t K) {
    std::memset(packed, 0, (K / 8) * sizeof(int32_t));
    for (size_t k_blk = 0; k_blk < K / 8; ++k_blk) {
        int32_t val = 0;
        for (int i = 0; i < 8; ++i) {
            val |= (static_cast<int32_t>(unpacked[k_blk * 8 + static_cast<size_t>(i)] & 0xF) << (i * 4));
        }
        packed[k_blk] = val;
    }
}

// Load model weights using memory-mapped SafeTensors.
std::unique_ptr<ModelWeights> Model::load_weights(const std::string& model_path, zedinferDeviceType_t target_device,
                                                  const ModelConfig& config,
                                                  std::function<bool(const std::string&)> to_cpu_pinned) {
    auto load_start = std::chrono::high_resolution_clock::now();
    ModelLoadProgress progress;

    size_t shard_total = 0;
    auto loader_unique = zedinfer::loader::SafeTensorsLoader::create(model_path, [&](size_t current, size_t total) {
        if (total == 0) {
            return;
        }
        if (shard_total != total) {
            shard_total = total;
            progress.begin_stage("Loading safetensor shards...", total, LoadProgressColor::Blue);
        }
        progress.update(current);
    });
    std::shared_ptr<zedinfer::loader::IModelLoader> loader(std::move(loader_unique));

    auto mmap_end = std::chrono::high_resolution_clock::now();
    progress.finish_stage();
    auto mmap_time = std::chrono::duration<double>(mmap_end - load_start).count();
    LOGI.printf("⏱️  Mmap time: %.4fs", mmap_time);

    auto weights = std::make_unique<ModelWeights>();
    weights->retain_resource(loader);
    std::map<std::string, LayerGroup> layer_groups;
    std::vector<std::pair<std::string, tensor_t>> standalone_tensors;
    auto convert_start = std::chrono::high_resolution_clock::now();
    auto tensor_names = loader->get_all_tensor_names();

    progress.begin_stage("Indexing tensors...", tensor_names.size(), LoadProgressColor::Blue);

    for (const auto& raw_name : tensor_names) {
        std::string mapped_name = map_weight_name(raw_name);
        if (mapped_name.find("_shape") != std::string::npos) {
            progress.advance();
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
        if (mapped_name.size() > 14 && mapped_name.substr(mapped_name.size() - 14) == ".weight_packed") {
            prefix = mapped_name.substr(0, mapped_name.size() - 14);
            layer_groups[prefix].packed_name = mapped_name;
            layer_groups[prefix].t_packed = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 7 && mapped_name.substr(mapped_name.size() - 7) == ".weight") {
            prefix = mapped_name.substr(0, mapped_name.size() - 7);
            layer_groups[prefix].weight_name = mapped_name;
            layer_groups[prefix].t_weight = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 13 && mapped_name.substr(mapped_name.size() - 13) == ".weight_scale") {
            prefix = mapped_name.substr(0, mapped_name.size() - 13);
            layer_groups[prefix].scale_name = mapped_name;
            layer_groups[prefix].t_scale = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 13 && mapped_name.substr(mapped_name.size() - 13) == ".weight_g_idx") {
            prefix = mapped_name.substr(0, mapped_name.size() - 13);
            layer_groups[prefix].g_idx_name = mapped_name;
            layer_groups[prefix].t_g_idx = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 13 && mapped_name.substr(mapped_name.size() - 13) == ".weight_zeros") {
            prefix = mapped_name.substr(0, mapped_name.size() - 13);
            layer_groups[prefix].zeros_name = mapped_name;
            layer_groups[prefix].t_zeros = tensor;
            layer_groups[prefix].prefix = prefix;
        } else if (mapped_name.size() > 5 && mapped_name.substr(mapped_name.size() - 5) == ".bias") {
            prefix = mapped_name.substr(0, mapped_name.size() - 5);
            layer_groups[prefix].bias_name = mapped_name;
            layer_groups[prefix].t_bias = tensor;
            layer_groups[prefix].prefix = prefix;
        } else {
            standalone_tensors.push_back({mapped_name, tensor});
        }

        progress.advance();
    }
    progress.finish_stage();

    size_t converted_count = 0;
    size_t cpu_pinned_count = 0;
    size_t materialize_total = standalone_tensors.size();
    for (const auto& [_, group] : layer_groups) {
        if (group.t_weight) {
            materialize_total++;
        }
        if (group.t_packed) {
            materialize_total++;
        }
        if (group.t_scale) {
            materialize_total++;
        }
        if (group.t_bias) {
            materialize_total++;
        }
        if (group.t_g_idx) {
            materialize_total++;
        }
        if (group.t_zeros) {
            materialize_total++;
        }
    }

    progress.begin_stage("Preparing model weights...", materialize_total, LoadProgressColor::Blue);

    // Materialize a tensor in CPU pinned memory by copying from its mmap source.
    // On NVIDIA runtime, Tensor::create(..., CPU, 0) goes through cudaMallocHost, so
    // the destination buffer is eligible for cudaMemcpyAsync with full PCIe bandwidth.
    auto route_to_cpu_pinned = [&](tensor_t src) -> tensor_t {
        auto pinned = Tensor::create(src->shape(), src->dtype(), ZEDINFER_DEVICE_CPU, 0);
        std::memcpy(pinned->data(), src->data(), src->numel() * src->elementSize());
        ++cpu_pinned_count;
        return pinned;
    };

    for (auto& pair : standalone_tensors) {
        tensor_t tensor = pair.second;
        if (to_cpu_pinned(pair.first)) {
            tensor = route_to_cpu_pinned(tensor);
        } else if (target_device == ZEDINFER_DEVICE_CPU
                   && (tensor->dtype() == ZEDINFER_DTYPE_BF16 || tensor->dtype() == ZEDINFER_DTYPE_F16)) {
            tensor = tensor->to(ZEDINFER_DTYPE_F32);
            converted_count++;
        } else if (target_device != ZEDINFER_DEVICE_CPU) {
            tensor = tensor->to(target_device, 0);
        }
        weights->add_tensor(pair.first, tensor);
        progress.advance();
    }

    for (auto& [prefix, group] : layer_groups) {
        tensor_t current_weight = group.t_packed ? group.t_packed : group.t_weight;

        // GPTQ format: transpose qweight/scales to our kernel's expected layout and
        // adjust zero-points. Leaves other quant formats (SparseML INT8) unchanged.
        if (current_weight && config.quant_config.quant_method == "gptq") {
            process_gptq_group(group, config, current_weight);
        }

        if (current_weight && group.t_g_idx) {
            if (!group.t_packed) {
                throw std::runtime_error("Act-order reordering requires packed weights: " + prefix);
            }

            LOGI.printf("🔄 Reordering weights for %s (Act-Order detected)", prefix.c_str());

            const int32_t* g_idx_ptr = reinterpret_cast<const int32_t*>(group.t_g_idx->data());
            const int32_t* packed_ptr = reinterpret_cast<const int32_t*>(group.t_packed->data());

            size_t N = group.t_packed->shape()[0];
            size_t K_packed = group.t_packed->shape()[1];
            size_t K = group.t_g_idx->shape()[0];

            if (K != K_packed * 8) {
                throw std::runtime_error("Shape mismatch in packing: " + prefix);
            }

            std::vector<int32_t> perm(K);
            std::iota(perm.begin(), perm.end(), 0);
            std::stable_sort(perm.begin(), perm.end(),
                             [&](int32_t a, int32_t b) { return g_idx_ptr[a] < g_idx_ptr[b]; });

            std::vector<int32_t> new_packed_data(N * (K / 8));
            std::vector<int8_t> row_unpacked(K);
            std::vector<int8_t> row_permuted(K);

            for (size_t n = 0; n < N; ++n) {
                const int32_t* src_row = packed_ptr + n * (K / 8);
                int32_t* dst_row = new_packed_data.data() + n * (K / 8);
                unpack_4bit_row(src_row, row_unpacked.data(), K);
                for (size_t k = 0; k < K; ++k) { row_permuted[k] = row_unpacked[static_cast<size_t>(perm[k])]; }
                repack_4bit_row(row_permuted.data(), dst_row, K);
            }

            auto t_packed_new = Tensor::create(group.t_packed->shape(), ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU);
            std::memcpy(t_packed_new->data(), new_packed_data.data(), new_packed_data.size() * sizeof(int32_t));
            group.t_packed = t_packed_new;

            auto t_perm = Tensor::create({K}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU);
            std::memcpy(t_perm->data(), perm.data(), perm.size() * sizeof(int32_t));
            group.t_g_idx = t_perm;
        }

        if (current_weight && config.quant_config.enabled && config.quant_config.weights.num_bits == 8
            && config.quant_config.weights.value_type == "int") {
            if (current_weight->dtype() == ZEDINFER_DTYPE_I8 || current_weight->dtype() == ZEDINFER_DTYPE_U8) {
                size_t N = current_weight->shape()[0];
                size_t K = current_weight->shape()[1];
                auto t_packed_new = Tensor::create({N, K / 4}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU);
                std::memcpy(t_packed_new->data(), current_weight->data(), N * K);
                group.t_packed = t_packed_new;
                group.packed_name = prefix + ".weight_packed";
                group.t_weight = nullptr;
            } else if (current_weight->dtype() == ZEDINFER_DTYPE_I32) {
                size_t N = current_weight->shape()[0];
                size_t K_packed = current_weight->shape()[1];
                auto t_packed_new = Tensor::create({N, K_packed}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU);

                const uint8_t* src = reinterpret_cast<const uint8_t*>(current_weight->data());
                uint8_t* dst = reinterpret_cast<uint8_t*>(t_packed_new->data());
                const size_t total_bytes = N * K_packed * sizeof(int32_t);

                for (size_t i = 0; i < total_bytes; ++i) { dst[i] = src[i] ^ 0x80; }

                group.t_packed = t_packed_new;
                group.packed_name = prefix + ".weight_packed";
                group.t_weight = nullptr;
            }
        }

        if (target_device == ZEDINFER_DEVICE_NVIDIA && current_weight && group.t_scale && config.quant_config.enabled
            && config.quant_config.weights.num_bits == 8 && config.quant_config.weights.group_size < 0) {
            size_t logical_k = 0;
            if (current_weight->dtype() == ZEDINFER_DTYPE_I8 || current_weight->dtype() == ZEDINFER_DTYPE_U8) {
                logical_k = current_weight->shape()[1];
            } else if (current_weight->dtype() == ZEDINFER_DTYPE_I32) {
                logical_k = current_weight->shape()[1] * 4;
            }

            int act_group_size = get_nvidia_int8_act_group_size(config.quant_config, logical_k);
            if (act_group_size > 0) {
                const size_t num_groups = logical_k / static_cast<size_t>(act_group_size);
                group.t_scale = expand_rowwise_scale_tensor(group.t_scale, current_weight->shape()[0],
                                                            static_cast<int>(num_groups));
            }
        }

        auto process_and_add = [&](tensor_t& tensor, const std::string& name) {
            if (!tensor) {
                return;
            }
            if (to_cpu_pinned(name)) {
                tensor = route_to_cpu_pinned(tensor);
            } else if (target_device == ZEDINFER_DEVICE_CPU
                       && (tensor->dtype() == ZEDINFER_DTYPE_BF16 || tensor->dtype() == ZEDINFER_DTYPE_F16)) {
                tensor = tensor->to(ZEDINFER_DTYPE_F32);
                converted_count++;
            } else if (target_device != ZEDINFER_DEVICE_CPU) {
                tensor = tensor->to(target_device, 0);
            }
            weights->add_tensor(name, tensor);
            progress.advance();
        };

        process_and_add(group.t_weight, group.weight_name);
        process_and_add(group.t_packed, group.packed_name);
        process_and_add(group.t_scale, group.scale_name);
        process_and_add(group.t_bias, group.bias_name);
        if (group.t_g_idx) {
            process_and_add(group.t_g_idx, group.g_idx_name);
        }
        if (group.t_zeros) {
            process_and_add(group.t_zeros, group.zeros_name);
        }
    }
    auto convert_end = std::chrono::high_resolution_clock::now();
    progress.finish_stage();
    auto convert_time = std::chrono::duration<double>(convert_end - convert_start).count();
    LOGI.printf("⏱️  Conversion time: %.4fs (%zu tensors)", convert_time, converted_count);
    if (cpu_pinned_count > 0) {
        LOGI.printf("[Loader] Routed %zu tensors to CPU pinned memory via predicate", cpu_pinned_count);
    }

#ifdef DEBUG
    LOGD << utils::get_numa_maps_info();
#endif

    return weights;
}

// Normalize weight names by stripping common prefixes (e.g., "model.")
// and remapping GPTQ-specific suffixes to the internal naming convention.
std::string Model::map_weight_name(const std::string& raw_name) {
    std::string name = raw_name;

    // Strip "model." prefix
    if (name.size() > 6 && name.substr(0, 6) == "model.") {
        name = name.substr(6);
    }
    // Then strip "language_model." prefix (Qwen3.5)
    if (name.size() > 15 && name.substr(0, 15) == "language_model.") {
        name = name.substr(15);
    }

    // Map GPTQ suffixes to internal naming convention
    if (name.size() > 8 && name.substr(name.size() - 8) == ".qweight") {
        name = name.substr(0, name.size() - 8) + ".weight_packed";
    } else if (name.size() > 7 && name.substr(name.size() - 7) == ".scales") {
        name = name.substr(0, name.size() - 7) + ".weight_scale";
    } else if (name.size() > 6 && name.substr(name.size() - 6) == ".g_idx") {
        name = name.substr(0, name.size() - 6) + ".weight_g_idx";
    } else if (name.size() > 7 && name.substr(name.size() - 7) == ".qzeros") {
        name = name.substr(0, name.size() - 7) + ".weight_zeros";
    }

    return name;
}

} // namespace zedinfer::model
