#include "frontend/models/qwen3_5.hpp"

#include "frontend/models/vision_tower.hpp"
#include "zedinfer.h"
#include "zedinfer/chat_template_jinja.hpp"

#include <plog/Log.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace zedinfer::model {

Qwen3_5Model::Qwen3_5Model(Qwen3_5Config config, std::unique_ptr<ModelWeights> weights, const ExecutorConfig& exec,
                           int max_concurrent, const std::string& model_path)
    : config_(std::move(config)), weights_(std::move(weights)) {
    // Count linear-attention layers and compute the QKV concat width used by
    // the SSM conv-state buffer. Width matches the in_proj_b output layout:
    //   2 × Hk × Dk  (q & k packed together) + Hv × Dv  (v).
    int num_linear = 0;
    for (const auto& t : config_.layer_types) {
        if (t == "linear_attention") {
            ++num_linear;
        }
    }
    const int qkv_dim = 2 * config_.linear_attn.num_k_heads * config_.linear_attn.key_head_dim
                      + config_.linear_attn.num_v_heads * config_.linear_attn.value_head_dim;

    SSMStatePoolConfig pool_cfg;
    pool_cfg.num_linear_layers = num_linear;
    pool_cfg.num_v_heads       = config_.linear_attn.num_v_heads;
    pool_cfg.value_head_dim    = config_.linear_attn.value_head_dim;
    pool_cfg.d_state           = config_.linear_attn.d_state;
    pool_cfg.conv_kernel_dim   = config_.linear_attn.conv_kernel_dim;
    pool_cfg.qkv_dim           = qkv_dim;
    pool_cfg.max_concurrent    = std::max(1, max_concurrent);
    // Map the parsed state_dtype string (set in load_config) to the runtime
    // dtype enum the pool allocates with. Default to BF16 to match the model
    // dtype when the string is empty / unrecognised.
    if (config_.linear_attn.state_dtype == "float32") {
        pool_cfg.state_dtype = ZEDINFER_DTYPE_F32;
    } else if (config_.linear_attn.state_dtype == "float16") {
        pool_cfg.state_dtype = ZEDINFER_DTYPE_F16;
    } else {
        pool_cfg.state_dtype = ZEDINFER_DTYPE_BF16;
    }

    ssm_pool_ = std::make_unique<SSMStatePool>(pool_cfg, exec);

    // Optional vision tower. M0 ctor only verifies expected weights exist; the
    // actual forward arrives in M3. Construction happens after weights are
    // loaded so missing-tensor errors surface here, not deep in forward().
    if (config_.has_vision) {
        vision_ = std::make_unique<VisionTower>(config_.vision, *weights_, exec);
    }

    const int num_full = static_cast<int>(config_.num_hidden_layers) - num_linear;
    LOGI.printf("[Qwen3_5Model] constructed: %d linear-attn layers, %d full-attn layers, vision=%s", num_linear,
                num_full, vision_ ? "yes" : "no");

    // Optional: load chat_template.jinja for Qwen3.5's multimodal ChatML
    // dialect. The file is shipped alongside the safetensors in HF releases;
    // when absent (e.g. minimal test fixtures) we silently leave the member
    // null so server / smoke-test code paths can fall back to the data-driven
    // zedinfer::ChatTemplate as needed.
    if (!model_path.empty()) {
        fs::path tpl_path = fs::path(model_path) / "chat_template.jinja";
        if (fs::exists(tpl_path)) {
            try {
                chat_template_ = std::make_shared<ChatTemplateJinja>(
                    ChatTemplateJinja::load(tpl_path.string()));
                LOGI << "[Qwen3_5Model] Loaded chat_template.jinja from " << tpl_path.string();
            } catch (const std::exception& e) {
                LOGW << "[Qwen3_5Model] Failed to compile chat_template.jinja at " << tpl_path.string()
                     << ": " << e.what();
            }
        } else {
            LOGI << "[Qwen3_5Model] No chat_template.jinja in " << model_path << "; skipping Jinja loader";
        }
    }
}

Qwen3_5Model::~Qwen3_5Model() = default;

size_t Qwen3_5Model::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_->get_all_weights()) {
        (void)name;
        if (tensor) {
            total += tensor->numel();
        }
    }
    return total;
}

ModelForwardConfig Qwen3_5Model::forward_config() const {
    // M0: forward pipeline lives in M1. The smoke load test (T13) checks for the
    // exact substring "not implemented until M1" so it can recognize the stub
    // and exit gracefully without erroring out the suite.
    throw std::runtime_error("Qwen3_5Model::forward_config: not implemented until M1");
}

HybridForwardConfig Qwen3_5Model::hybrid_forward_config() const {
    HybridForwardConfig h(config_, *weights_);
    h.layer_kinds.reserve(config_.layer_types.size());
    for (const auto& t : config_.layer_types) {
        h.layer_kinds.push_back(t == "linear_attention" ? LayerKind::Linear : LayerKind::Full);
    }
    h.linear_attn          = config_.linear_attn;
    h.mrope.interleaved    = config_.mrope_interleaved;
    h.mrope.section        = config_.mrope_section;
    h.mrope.partial_factor = config_.partial_rotary_factor;
    h.mrope.theta          = config_.rope_theta;
    h.attn_output_gate     = config_.attn_output_gate;
    h.ssm_pool             = ssm_pool_.get();
    // Precompute O(1) layer-index lookup tables for the M1 decode hot path so
    // dispatch sites do not rescan layer_kinds on every layer call.
    h.rebuild_layer_index_tables();
    return h;
}

} // namespace zedinfer::model
