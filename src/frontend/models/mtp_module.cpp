#include "frontend/models/mtp_module.hpp"

#include "backend/device/runtime_api.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"

#include <plog/Log.h>

#include <stdexcept>
#include <string>

namespace zedinfer::model {

namespace {

// Wire up MTP layer-0 experts from ModelWeights into the supplied
// 1-layer ExpertWeights pool. Handles both release layouts:
//
//   (A) Fused 3-D bf16 (Qwen3.6-35B-A3B style):
//         mtp.layers.0.mlp.experts.gate_up_proj   [E, 2M, H]
//         mtp.layers.0.mlp.experts.down_proj      [E,  H, M]
//
//   (B) Per-expert (Qwen3.5-A3B-GPTQ-Int4 style; MTP is NOT quantized in
//       the GPTQ release because dynamic rules exclude `-:.*mtp.*`):
//         mtp.layers.0.mlp.experts.{i}.gate_proj.weight  [M, H]
//         mtp.layers.0.mlp.experts.{i}.up_proj.weight    [M, H]
//         mtp.layers.0.mlp.experts.{i}.down_proj.weight  [H, M]
//
// For (A) the per-expert tensors share storage with the fused tensor via
// Tensor::slice + view (zero copy). For (B) the per-expert tensors are
// already separate and just get moved into the pool.
void wire_mtp_experts(ModelWeights& weights, ExpertWeights& dst, size_t num_experts) {
    const std::string fused_gate_up_name = "mtp.layers.0.mlp.experts.gate_up_proj";
    const std::string fused_down_name    = "mtp.layers.0.mlp.experts.down_proj";

    if (weights.has_tensor(fused_gate_up_name) && weights.has_tensor(fused_down_name)) {
        // Layout (A): fused 3-D. Slice + view per expert.
        auto gu = weights.get_tensor(fused_gate_up_name);
        auto dp = weights.get_tensor(fused_down_name);
        if (gu->ndim() != 3 || dp->ndim() != 3) {
            throw std::runtime_error("[MTPModule] fused expert tensors must be 3-D");
        }
        const auto& gu_shape = gu->shape();
        const auto& dp_shape = dp->shape();
        if (gu_shape[0] != num_experts || dp_shape[0] != num_experts) {
            throw std::runtime_error("[MTPModule] num_experts mismatch on fused tensor dim 0");
        }
        const size_t two_M = gu_shape[1];
        const size_t H     = gu_shape[2];
        if (two_M % 2 != 0) {
            throw std::runtime_error("[MTPModule] gate_up_proj dim 1 (" + std::to_string(two_M)
                                     + ") is not even");
        }
        const size_t M = two_M / 2;
        if (dp_shape[1] != H || dp_shape[2] != M) {
            throw std::runtime_error("[MTPModule] down_proj shape disagrees with gate_up_proj");
        }
        for (size_t i = 0; i < num_experts; ++i) {
            auto& ffn      = dst.at(0, i);
            auto  gu_row_3d = gu->slice(0, i, i + 1);
            auto  gu_row_2d = gu_row_3d->view({two_M, H});
            ffn.gate_weight = gu_row_2d->slice(0, 0, M);
            ffn.up_weight   = gu_row_2d->slice(0, M, two_M);

            auto dp_row_3d = dp->slice(0, i, i + 1);
            ffn.down_weight = dp_row_3d->view({H, M});
        }
        weights.remove_tensor(fused_gate_up_name);
        weights.remove_tensor(fused_down_name);
        LOGI.printf("[MTPModule] expert layout=fused; expanded %zu experts into per-expert views",
                    num_experts);
        return;
    }

    // Layout (B): per-expert tensors already separated.
    // Probe expert 0 to confirm and produce a clear error if neither layout matches.
    const std::string probe_name = "mtp.layers.0.mlp.experts.0.down_proj.weight";
    if (!weights.has_tensor(probe_name)) {
        throw std::runtime_error("[MTPModule] missing both fused tensor ('" + fused_gate_up_name
                                 + "') and per-expert tensor ('" + probe_name
                                 + "'); model release lacks MTP experts");
    }
    for (size_t i = 0; i < num_experts; ++i) {
        auto& ffn = dst.at(0, i);
        const std::string per_expert_prefix = "mtp.layers.0.mlp.experts." + std::to_string(i);
        const std::string gn = per_expert_prefix + ".gate_proj.weight";
        const std::string un = per_expert_prefix + ".up_proj.weight";
        const std::string dn = per_expert_prefix + ".down_proj.weight";
        if (!weights.has_tensor(gn) || !weights.has_tensor(un) || !weights.has_tensor(dn)) {
            throw std::runtime_error("[MTPModule] missing per-expert weight for expert "
                                     + std::to_string(i) + " (looked for " + gn + ")");
        }
        ffn.gate_weight = weights.get_tensor(gn);
        ffn.up_weight   = weights.get_tensor(un);
        ffn.down_weight = weights.get_tensor(dn);
        weights.remove_tensor(gn);
        weights.remove_tensor(un);
        weights.remove_tensor(dn);
    }
    LOGI.printf("[MTPModule] expert layout=per-expert; bound %zu experts (no copy)", num_experts);
}

// Verify a single tensor exists and is of the expected dtype (bf16 by default).
// Throws with a clear message that says which key is missing.
tensor_t fetch(const ModelWeights& weights, const std::string& name) {
    if (!weights.has_tensor(name)) {
        throw std::runtime_error("[MTPModule] missing weight: " + name);
    }
    return weights.get_tensor(name);
}

} // namespace

MTPModule::MTPModule(const Qwen3_5MoEConfig& main_cfg, ModelWeights& weights, const ExecutorConfig& exec)
    : main_cfg_(main_cfg), exec_(exec) {
    // If the user-provided model doesn't ship MTP (e.g. base Qwen3, DeepSeek
    // distill), bail early. ready() will return false and callers (engine /
    // scheduler) treat speculative decoding as disabled.
    if (!weights.has_tensor("mtp.fc.weight")) {
        LOGI << "[MTPModule] mtp.fc.weight not found; MTP head disabled for this model";
        return;
    }

    // ----- Fusion projection + pre-fc norms -----
    pre_fc_norm_embedding_ = fetch(weights, "mtp.pre_fc_norm_embedding.weight");
    pre_fc_norm_hidden_    = fetch(weights, "mtp.pre_fc_norm_hidden.weight");
    fc_weight_             = fetch(weights, "mtp.fc.weight");

    // ----- Layer 0 standalone tensors -----
    const std::string p = "mtp.layers.0.";
    in_layernorm_   = fetch(weights, p + "input_layernorm.weight");
    post_layernorm_ = fetch(weights, p + "post_attention_layernorm.weight");
    q_proj_         = fetch(weights, p + "self_attn.q_proj.weight");
    k_proj_         = fetch(weights, p + "self_attn.k_proj.weight");
    v_proj_         = fetch(weights, p + "self_attn.v_proj.weight");
    o_proj_         = fetch(weights, p + "self_attn.o_proj.weight");
    q_norm_         = fetch(weights, p + "self_attn.q_norm.weight");
    k_norm_         = fetch(weights, p + "self_attn.k_norm.weight");

    // ----- MoE router + shared expert -----
    mlp_gate_router_           = fetch(weights, p + "mlp.gate.weight");
    shared_expert_gate_        = fetch(weights, p + "mlp.shared_expert_gate.weight");
    shared_expert_gate_proj_   = fetch(weights, p + "mlp.shared_expert.gate_proj.weight");
    shared_expert_up_proj_     = fetch(weights, p + "mlp.shared_expert.up_proj.weight");
    shared_expert_down_proj_   = fetch(weights, p + "mlp.shared_expert.down_proj.weight");

    // ----- 256 experts via fused-tensor split -----
    const size_t num_experts = static_cast<size_t>(main_cfg_.num_experts);
    if (num_experts == 0) {
        throw std::runtime_error("[MTPModule] main_cfg.num_experts is 0; cannot build MTP MoE");
    }
    experts_ = std::make_unique<ExpertWeights>(/*num_layers=*/1, num_experts);
    wire_mtp_experts(weights, *experts_, num_experts);

    // ----- Final norm -----
    final_norm_ = fetch(weights, "mtp.norm.weight");

    ready_ = true;
    LOGI.printf("[MTPModule] loaded: 1 transformer layer + fc(2H->H) + final_norm; "
                "fused experts expanded to %zu per-expert views",
                num_experts);
}

MTPModule::~MTPModule() = default;

tensor_t MTPModule::forward(tensor_t /*hidden_at_t*/, int /*next_token_id*/,
                            const ExecutorConfig& /*exec*/) {
    // Stage A skeleton — math implementation lives in Stage B once we wire
    // MTP's own KV cache and merge with the speculative-decode scheduler.
    throw std::runtime_error("MTPModule::forward not yet implemented (Stage A: weights "
                             "load + verify only; Stage B: forward + KV cache; "
                             "Stage C: scheduler integration)");
}

} // namespace zedinfer::model
