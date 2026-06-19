#include "frontend/models/qwen3_5.hpp"

#include "backend/device/runtime_api.hpp"
#include "frontend/models/mtp_module.hpp"
#include "frontend/models/vision_tower.hpp"
#include "zedinfer.h"
#include "zedinfer/chat_template_jinja.hpp"

#include <plog/Log.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace zedinfer::model {

namespace {

// Reorder full-attention `q_proj.weight` so that the first Hq*Dh rows are all
// query channels and the next Hq*Dh rows are all gate channels.
//
// HF Qwen3_5MoeAttention.forward (line 682-684) splits q_proj output via
//   torch.chunk(q_proj(x).view(..., num_heads, head_dim*2), 2, dim=-1)
// which interleaves query and gate per head along the output axis:
//   raw_out[..., h, 0..Dh-1]    = head h query
//   raw_out[..., h, Dh..2*Dh-1] = head h gate
// In the stored weight (shape [Hq*Dh*2, hidden] row-major) this means
// rows are grouped as [q_h0, g_h0, q_h1, g_h1, ...] in Dh-row chunks.
//
// forward_full_attn_layer (in hybrid_transformer_forward.cpp) instead expects
// the first Hq*Dh rows to be the full query block and the next Hq*Dh rows to
// be the full gate block — i.e. it slices the weight as [0, q_dim) for query
// and [q_dim, 2*q_dim) for gate. Reordering the stored weight once at load
// time keeps that simple slice correct and avoids a per-call permute.
void reorder_q_proj_weight(tensor_t w, int Hq, int Dh, int hidden) {
    if (!w) {
        return;
    }
    if (w->dtype() != ZEDINFER_DTYPE_BF16) {
        throw std::runtime_error("[Qwen3_5Model] reorder_q_proj_weight: only bf16 supported");
    }
    const size_t total_rows = static_cast<size_t>(Hq) * 2 * static_cast<size_t>(Dh);
    const size_t row_w = static_cast<size_t>(hidden);
    const size_t N = total_rows * row_w;
    auto* api = device::getRuntimeAPI(w->deviceType());

    std::vector<uint16_t> host(N);
    api->memcpy_sync(host.data(), w->data(), N * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);

    std::vector<uint16_t> reord(N);
    const size_t q_block_rows = static_cast<size_t>(Hq) * static_cast<size_t>(Dh);
    for (size_t h = 0; h < static_cast<size_t>(Hq); ++h) {
        for (size_t k = 0; k < 2; ++k) {
            const uint16_t* src = host.data() + (h * 2 * static_cast<size_t>(Dh) + k * static_cast<size_t>(Dh)) * row_w;
            uint16_t* dst = reord.data() + (k * q_block_rows + h * static_cast<size_t>(Dh)) * row_w;
            std::memcpy(dst, src, static_cast<size_t>(Dh) * row_w * sizeof(uint16_t));
        }
    }

    api->memcpy_sync(w->data(), reord.data(), N * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
}

static float bf16_bits_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Replace an fp32 weight tensor with a freshly-allocated bf16 tensor of the
// same shape, holding the truncate-to-bf16 conversion of the original values.
// The new tensor is registered back into ModelWeights under the same key so
// every later W(name) lookup transparently sees bf16.
void cast_fp32_weight_to_bf16(ModelWeights& weights, const std::string& name) {
    if (!weights.has_tensor(name)) {
        return;
    }
    tensor_t orig = weights.get_tensor(name);
    if (!orig || orig->dtype() != ZEDINFER_DTYPE_F32) {
        return;
    }

    const size_t W = orig->numel();
    auto* api = device::getRuntimeAPI(orig->deviceType());

    std::vector<float> host_f32(W);
    std::vector<uint16_t> host_bf16(W);
    api->memcpy_sync(host_f32.data(), orig->data(), W * sizeof(float), ZEDINFER_MEMCPY_D2H);
    for (size_t i = 0; i < W; ++i) {
        uint32_t u = 0;
        std::memcpy(&u, &host_f32[i], sizeof(u));
        host_bf16[i] = static_cast<uint16_t>(u >> 16); // truncate-to-bf16
    }

    auto replacement = Tensor::create(orig->shape(), ZEDINFER_DTYPE_BF16, orig->deviceType(), orig->deviceId());
    api->memcpy_sync(replacement->data(), host_bf16.data(), W * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
    weights.add_tensor(name, replacement);
}

// Replace a bf16 weight tensor with a freshly-allocated fp32 tensor. GDN kernels
// consume A_log as float* because the decay exponent is sensitive and older
// Qwen3.5 releases stored this weight as fp32. Qwen3.6 bf16 releases store A_log
// as BF16, so normalize the storage at load time.
void cast_bf16_weight_to_f32(ModelWeights& weights, const std::string& name) {
    if (!weights.has_tensor(name)) {
        return;
    }
    tensor_t orig = weights.get_tensor(name);
    if (!orig || orig->dtype() != ZEDINFER_DTYPE_BF16) {
        return;
    }

    const size_t W = orig->numel();
    auto* api = device::getRuntimeAPI(orig->deviceType());

    std::vector<uint16_t> host_bf16(W);
    std::vector<float> host_f32(W);
    api->memcpy_sync(host_bf16.data(), orig->data(), W * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    for (size_t i = 0; i < W; ++i) { host_f32[i] = bf16_bits_to_f32(host_bf16[i]); }

    auto replacement = Tensor::create(orig->shape(), ZEDINFER_DTYPE_F32, orig->deviceType(), orig->deviceId());
    api->memcpy_sync(replacement->data(), host_f32.data(), W * sizeof(float), ZEDINFER_MEMCPY_H2D);
    weights.add_tensor(name, replacement);
}

// Walk linear-attention layers and convert A_log from bf16 to fp32 when needed.
// The GDN kernels read this tensor as float*, so leaving a bf16 checkpoint tensor
// in place corrupts the per-head decay and destabilizes long generation.
void fixup_qwen3_5_linear_attn_A_log_weights(ModelWeights& weights, const Qwen3_5Config& cfg) {
    int converted = 0;
    int already_f32 = 0;
    for (size_t L = 0; L < cfg.layer_types.size(); ++L) {
        if (cfg.layer_types[L] != "linear_attention") {
            continue;
        }
        const std::string name = "layers." + std::to_string(L) + ".linear_attn.A_log";
        if (!weights.has_tensor(name)) {
            continue;
        }
        tensor_t w = weights.get_tensor(name);
        if (w->dtype() == ZEDINFER_DTYPE_BF16) {
            cast_bf16_weight_to_f32(weights, name);
            ++converted;
        } else if (w->dtype() == ZEDINFER_DTYPE_F32) {
            ++already_f32;
        } else {
            throw std::runtime_error("[Qwen3_5Model] " + name + " must be bf16 or fp32, got dtype "
                                     + std::to_string(static_cast<int>(w->dtype())));
        }
    }
    LOGI.printf("[Qwen3_5Model] Normalized linear_attn.A_log tensors to fp32: converted=%d already_f32=%d", converted,
                already_f32);
}

// Walk linear-attention layers and convert their .norm.weight tensors from
// fp32 to bf16 once when needed. ops::rms_norm requires the weight to match the
// activation dtype (bf16); some releases already store this tensor as bf16.
void fixup_qwen3_5_linear_attn_norm_weights(ModelWeights& weights, const Qwen3_5Config& cfg) {
    int count = 0;
    for (size_t L = 0; L < cfg.layer_types.size(); ++L) {
        if (cfg.layer_types[L] != "linear_attention") {
            continue;
        }
        const std::string name = "layers." + std::to_string(L) + ".linear_attn.norm.weight";
        if (weights.has_tensor(name) && weights.get_tensor(name)->dtype() == ZEDINFER_DTYPE_F32) {
            cast_fp32_weight_to_bf16(weights, name);
            ++count;
        }
    }
    LOGI.printf("[Qwen3_5Model] Pre-cast %d linear_attn.norm.weight tensors from fp32 to bf16", count);
}

// Walk full-attention layers and reorder their q_proj weights.
void fixup_qwen3_5_q_proj_weights(ModelWeights& weights, const Qwen3_5Config& cfg) {
    const int Hq = static_cast<int>(cfg.num_attention_heads);
    const int Dh = cfg.head_dim > 0 ? cfg.head_dim : static_cast<int>(cfg.hidden_size / cfg.num_attention_heads);
    const int hidden = static_cast<int>(cfg.hidden_size);

    int count = 0;
    for (size_t L = 0; L < cfg.layer_types.size(); ++L) {
        if (cfg.layer_types[L] != "full_attention") {
            continue;
        }
        const std::string name = "layers." + std::to_string(L) + ".self_attn.q_proj.weight";
        if (weights.has_tensor(name)) {
            reorder_q_proj_weight(weights.get_tensor(name), Hq, Dh, hidden);
            ++count;
        }
    }
    LOGI.printf("[Qwen3_5Model] Reordered q_proj.weight (interleaved q|gate -> contiguous q|gate) on %d "
                "full-attention layers",
                count);
}

} // namespace

Qwen3_5Model::Qwen3_5Model(Qwen3_5Config config, std::unique_ptr<ModelWeights> weights, const ExecutorConfig& exec,
                           int max_concurrent, const std::string& model_path)
    : config_(std::move(config)), weights_(std::move(weights)) {
    // Note: Qwen3.5's (1 + weight) RMSNorm convention is applied at kernel time
    // (ops::rms_norm with add_one_to_weight=true), not pre-baked at load time —
    // pre-baking into bf16 lost precision around 1.0 and drifted from HF.

    // De-interleave full-attention q_proj.weight so it matches the layout the
    // forward path assumes (contiguous [query | gate] blocks instead of the
    // HF [q_h0, g_h0, q_h1, g_h1, ...] per-head interleaving).
    fixup_qwen3_5_q_proj_weights(*weights_, config_);

    // GDN kernels consume A_log as fp32. Qwen3.6 bf16 checkpoints store it as
    // BF16, so promote it once at load to preserve the kernel contract.
    fixup_qwen3_5_linear_attn_A_log_weights(*weights_, config_);

    // Pre-cast linear_attn.norm.weight to bf16 when the checkpoint stores it as
    // fp32, so the rms_norm dtype check inside forward_linear_attn_layer is
    // satisfied without a per-call cast.
    fixup_qwen3_5_linear_attn_norm_weights(*weights_, config_);

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
    pool_cfg.num_v_heads = config_.linear_attn.num_v_heads;
    pool_cfg.value_head_dim = config_.linear_attn.value_head_dim;
    pool_cfg.d_state = config_.linear_attn.d_state;
    pool_cfg.conv_kernel_dim = config_.linear_attn.conv_kernel_dim;
    pool_cfg.qkv_dim = qkv_dim;
    pool_cfg.max_concurrent = std::max(1, max_concurrent);
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
                chat_template_ = std::make_shared<ChatTemplateJinja>(ChatTemplateJinja::load(tpl_path.string()));
                LOGI << "[Qwen3_5Model] Loaded chat_template.jinja from " << tpl_path.string();
            } catch (const std::exception& e) {
                LOGW << "[Qwen3_5Model] Failed to compile chat_template.jinja at " << tpl_path.string() << ": "
                     << e.what();
            }
        } else {
            LOGI << "[Qwen3_5Model] No chat_template.jinja in " << model_path << "; skipping Jinja loader";
        }
    }

    // Optional MTP speculative-decode head. Build it here for DENSE MTP layers
    // (Qwen3.5/3.6-27B: mtp.layers.0.mlp.{gate,up,down}_proj). The MoE subclass
    // (Qwen3_5MoeModel) builds the MoE-expert MTP head itself after this ctor
    // using moe_config_, so skip when the MoE router weight is present.
    if (weights_->has_tensor("mtp.fc.weight") && !weights_->has_tensor("mtp.layers.0.mlp.gate.weight")) {
        mtp_ = std::make_unique<MTPModule>(config_, *weights_, exec);
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
    // M1: forward_config returns the hybrid config as its base slice. Callers
    // that walk dispatch through ServingLoop/Profiler check for the hybrid
    // subclass via dynamic_cast (`Qwen3_5Model*`) and route to
    // hybrid_transformer_forward(); callers that only need ModelConfig fields
    // (DecodeScratch sizing, KV pool init, etc.) get the slice and work
    // unchanged.
    return hybrid_forward_config();
}

HybridForwardConfig Qwen3_5Model::hybrid_forward_config() const {
    HybridForwardConfig h(config_, *weights_);
    h.layer_kinds.reserve(config_.layer_types.size());
    for (const auto& t : config_.layer_types) {
        h.layer_kinds.push_back(t == "linear_attention" ? LayerKind::Linear : LayerKind::Full);
    }
    h.linear_attn = config_.linear_attn;
    h.mrope.interleaved = config_.mrope_interleaved;
    h.mrope.section = config_.mrope_section;
    h.mrope.partial_factor = config_.partial_rotary_factor;
    h.mrope.theta = config_.rope_theta;
    h.attn_output_gate = config_.attn_output_gate;
    h.ssm_pool = ssm_pool_.get();
    // Precompute O(1) layer-index lookup tables for the M1 decode hot path so
    // dispatch sites do not rescan layer_kinds on every layer call.
    h.rebuild_layer_index_tables();
    return h;
}

} // namespace zedinfer::model
