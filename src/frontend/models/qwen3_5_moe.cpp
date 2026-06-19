#include "frontend/models/qwen3_5_moe.hpp"

#include "frontend/models/expert_weights.hpp"
#include "frontend/models/mtp_module.hpp"

#include <plog/Log.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zedinfer::model {

namespace {

// parse_expert_tensor_name / assign_expert_tensor are shared with the Qwen3 MoE
// loader and live in expert_weights.hpp — both use the identical per-expert
// naming convention after the loader strips the "model." / "language_model."
// prefixes.

// Parse a fused expert tensor name produced by newer Qwen3.x bf16 releases
// (e.g. Qwen3.6-35B-A3B), where all experts within a layer share one 3-D
// tensor instead of per-expert per-projection tensors.
//
//   layers.{L}.mlp.experts.gate_up_proj  shape=[num_experts, 2*moe_inter, hidden]
//   layers.{L}.mlp.experts.down_proj     shape=[num_experts,   hidden, moe_inter]
//
// Returns true and fills `layer` + `proj` if the name matches one of those
// two patterns. The MTP head's fused tensors (`mtp.layers.*.mlp.experts.*`)
// deliberately don't match — they're not consumed by the main forward path.
bool parse_fused_expert_tensor_name(const std::string& name, size_t& layer, std::string& proj) {
    static const std::string kPrefix = "layers.";
    static const std::string kMid = ".mlp.experts.";
    if (name.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    size_t pos = kPrefix.size();
    size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    try {
        layer = std::stoul(name.substr(pos, dot - pos));
    } catch (...) { return false; }
    if (name.compare(dot, kMid.size(), kMid) != 0) {
        return false;
    }
    pos = dot + kMid.size();
    proj = name.substr(pos);
    return proj == "gate_up_proj" || proj == "down_proj";
}

// Expand fused 3-D expert tensors (newer bf16 release format) into per-expert
// 2-D views so the rest of the loader (which is built around the GPTQ-Int4
// per-expert naming convention) can consume them uniformly.
//
// The per-expert tensors share storage with the fused tensors via Tensor::slice
// + Tensor::view (both share `_storage` shared_ptr). After expansion the
// original fused tensor is removed from ModelWeights; its storage stays alive
// as long as any of the per-expert views are held. Zero extra device memory.
//
// Layout (matches HF Qwen3_5MoeExperts param init):
//   gate_up_proj[i, :moe_inter,   :]  -> experts.{i}.gate_proj.weight  [M, H]
//   gate_up_proj[i, moe_inter:,   :]  -> experts.{i}.up_proj.weight    [M, H]
//   down_proj   [i,   :,          :]  -> experts.{i}.down_proj.weight  [H, M]
//
// No-op for models already in per-expert format (Qwen3 / Qwen3.5 GPTQ-Int4).
void expand_fused_expert_weights(ModelWeights& weights, size_t num_layers, size_t num_experts_per_layer) {
    // First pass: collect the fused tensor handles so the iteration is not
    // invalidated by add_tensor / remove_tensor mutations below.
    struct Hit {
        std::string name;
        tensor_t tensor;
        size_t layer;
        std::string proj;
    };
    std::vector<Hit> hits;
    for (const auto& [name, tensor] : weights.get_all_weights()) {
        size_t layer = 0;
        std::string proj;
        if (parse_fused_expert_tensor_name(name, layer, proj) && layer < num_layers) {
            hits.push_back({name, tensor, layer, proj});
        }
    }
    if (hits.empty()) {
        return;
    }

    int n_views = 0;
    for (const auto& hit : hits) {
        const std::string prefix = "layers." + std::to_string(hit.layer) + ".mlp.experts.";
        const auto& shape = hit.tensor->shape();
        if (shape.size() != 3) {
            throw std::runtime_error("[Qwen3_5MoE] expected 3-D fused expert tensor for " + hit.name + " but got "
                                     + std::to_string(shape.size()) + "-D");
        }
        const size_t E = shape[0];
        const size_t want_E = num_experts_per_layer;
        if (E != want_E) {
            throw std::runtime_error("[Qwen3_5MoE] " + hit.name + " has " + std::to_string(E)
                                     + " experts on dim 0 but config says num_experts=" + std::to_string(want_E));
        }
        if (hit.proj == "gate_up_proj") {
            const size_t two_M = shape[1];
            const size_t H = shape[2];
            if (two_M % 2 != 0) {
                throw std::runtime_error("[Qwen3_5MoE] " + hit.name + " dim 1 = " + std::to_string(two_M)
                                         + " is not even; cannot split gate/up");
            }
            const size_t M = two_M / 2;
            for (size_t i = 0; i < E; ++i) {
                // [E, 2M, H] slice(0, i, i+1) -> [1, 2M, H] (contiguous because
                // dim-0 slice preserves the inner contiguous layout). view to
                // 2-D then slice the 2M axis into gate (first half) and up
                // (second half), matching HF's gate-then-up param init.
                auto row_3d = hit.tensor->slice(0, i, i + 1);
                auto row_2d = row_3d->view({two_M, H});
                auto gate = row_2d->slice(0, 0, M);
                auto up = row_2d->slice(0, M, two_M);
                weights.add_tensor(prefix + std::to_string(i) + ".gate_proj.weight", gate);
                weights.add_tensor(prefix + std::to_string(i) + ".up_proj.weight", up);
            }
            n_views += static_cast<int>(E * 2);
        } else { // down_proj
            const size_t H = shape[1];
            const size_t M = shape[2];
            for (size_t i = 0; i < E; ++i) {
                auto row_3d = hit.tensor->slice(0, i, i + 1);
                auto down = row_3d->view({H, M});
                weights.add_tensor(prefix + std::to_string(i) + ".down_proj.weight", down);
            }
            n_views += static_cast<int>(E);
        }
        weights.remove_tensor(hit.name);
    }

    LOGI.printf("[Qwen3_5MoE] Expanded %zu fused expert tensors into %d per-expert views "
                "(zero-copy slice; storage shared with originals)",
                hits.size(), n_views);
}

// Move all expert tensors out of ModelWeights into a new ExpertWeights.
// Mirrors src/frontend/models/qwen3_moe.cpp:extract_expert_weights.
std::unique_ptr<ExpertWeights> extract_expert_weights(ModelWeights& weights, size_t num_layers,
                                                      size_t num_experts_per_layer) {
    auto experts = std::make_unique<ExpertWeights>(num_layers, num_experts_per_layer);
    std::vector<std::string> to_remove;

    for (const auto& [name, tensor] : weights.get_all_weights()) {
        size_t layer = 0, expert_id = 0;
        std::string proj, suffix;
        if (!parse_expert_tensor_name(name, layer, expert_id, proj, suffix)) {
            continue;
        }
        if (layer >= num_layers || expert_id >= num_experts_per_layer) {
            continue;
        }
        assign_expert_tensor(experts->at(layer, expert_id), proj, suffix, tensor);
        to_remove.push_back(name);
    }

    for (const auto& name : to_remove) { weights.remove_tensor(name); }

    LOGI.printf("[Qwen3_5MoE] Extracted %zu expert tensors into ExpertWeights [%zu layers x %zu experts]",
                to_remove.size(), num_layers, num_experts_per_layer);
    return experts;
}

} // namespace

Qwen3_5MoeModel::Qwen3_5MoeModel(Qwen3_5MoEConfig config, std::unique_ptr<ModelWeights> weights,
                                 const ExecutorConfig& exec, int max_concurrent, ExpertPoolConfig pool_cfg,
                                 const std::string& model_path)
    // Forward to parent: Qwen3_5Model takes ownership of the weights and exposes them through the
    // protected weights_ member. The parent ctor runs SSM pool sizing + optional VisionTower
    // verification first; expert extraction below mutates *weights_ in-place after that. The
    // base-slice copy is intentional — Qwen3_5Config / Qwen3_5MoEConfig are plain structs with
    // no virtual functions, so slicing the MoE-only fields off the parent copy is safe.
    : Qwen3_5Model(static_cast<const Qwen3_5Config&>(config), std::move(weights), exec, max_concurrent, model_path),
      moe_config_(std::move(config)) {
    if (!weights_) {
        throw std::runtime_error("[Qwen3_5MoeModel] parent ctor did not retain ModelWeights");
    }
    // Early guard: a missing or zero num_experts in config silently produces an empty
    // ExpertPool, which is only detectable later via an "experts not found" failure deep
    // inside the M1 forward path. Catch it at construction time so misconfigured models
    // (e.g. text_config.num_experts absent from config.json) fail fast and clearly.
    if (moe_config_.num_experts <= 0 || moe_config_.num_experts_per_tok <= 0) {
        throw std::runtime_error("[Qwen3_5MoeModel] invalid MoE config: num_experts="
                                 + std::to_string(moe_config_.num_experts)
                                 + ", num_experts_per_tok=" + std::to_string(moe_config_.num_experts_per_tok)
                                 + " (both must be > 0; check config.json text_config)");
    }
    // Some bf16 releases (Qwen3.6-35B-A3B) store all experts of a layer in one
    // 3-D tensor (experts.gate_up_proj / experts.down_proj). The GPTQ-Int4
    // release we originally bring-up'd against uses per-expert tensors. Detect
    // the fused layout and expand it into per-expert views first; the rest of
    // the loader is then format-agnostic.
    expand_fused_expert_weights(*weights_, moe_config_.num_hidden_layers, static_cast<size_t>(moe_config_.num_experts));
    auto experts = extract_expert_weights(*weights_, moe_config_.num_hidden_layers,
                                          static_cast<size_t>(moe_config_.num_experts));
    expert_pool_ = std::make_unique<ExpertPool>(std::move(experts), pool_cfg);

    // Optional MTP head. Constructor probes for mtp.fc.weight and silently
    // disables itself (ready()==false) when the release doesn't ship MTP.
    // We have to construct it AFTER the main expert extraction above so the
    // expert iterator doesn't accidentally pull in mtp.layers.0.mlp.experts.*
    // (extract_expert_weights only looks at the "layers.{L}." prefix; MTP
    // tensors live under "mtp.layers.{L}." and are skipped).
    mtp_ = std::make_unique<MTPModule>(moe_config_, *weights_, exec);

    LOGI.printf("[Qwen3_5MoeModel] constructed: experts=%d top_k=%d shared_expert_size=%d mtp=%s",
                moe_config_.num_experts, moe_config_.num_experts_per_tok, moe_config_.shared_expert_intermediate_size,
                (mtp_ && mtp_->ready()) ? "yes" : "no");
}

Qwen3_5MoeModel::~Qwen3_5MoeModel() = default;

HybridForwardConfig Qwen3_5MoeModel::hybrid_forward_config_moe() const {
    auto h = Qwen3_5Model::hybrid_forward_config();

    h.is_moe = true;
    h.num_experts = static_cast<size_t>(moe_config_.num_experts);
    h.num_experts_per_tok = static_cast<size_t>(moe_config_.num_experts_per_tok);
    h.moe_intermediate_size = static_cast<size_t>(moe_config_.moe_intermediate_size);
    h.shared_expert_intermediate_size = static_cast<size_t>(moe_config_.shared_expert_intermediate_size);
    h.decoder_sparse_step = static_cast<size_t>(moe_config_.decoder_sparse_step);
    h.mlp_only_layers = moe_config_.mlp_only_layers;
    // Qwen3_5MoeTopKRouter ALWAYS renormalizes the top-k probabilities so they
    // sum to 1 (modeling_qwen3_5_moe.py:788). Equivalent to norm_topk_prob=true
    // in the Qwen3 family. Without this, expert contributions are scaled down
    // by the sum-of-top-k probability mass and the residual ends up dominating
    // the MoE output, yielding context-independent degenerate generation.
    h.norm_topk_prob = true;
    h.has_shared_expert = h.shared_expert_intermediate_size > 0
                       && (weights_->has_tensor("layers.0.mlp.shared_expert.gate_proj.weight")
                           || weights_->has_tensor("layers.0.mlp.shared_expert.gate_proj.weight_packed"));
    h.expert_pool = expert_pool_.get();

    // Cache per-layer router weights for moe_layer_forward's compute_router_topk
    // hot path (avoids string-keyed lookups per layer per step).
    h.router_weights.resize(moe_config_.num_hidden_layers);
    for (size_t l = 0; l < moe_config_.num_hidden_layers; ++l) {
        const std::string name = h.router_weight_name(static_cast<int>(l));
        h.router_weights[l] = weights_->has_tensor(name) ? weights_->get_tensor(name) : nullptr;
    }

    // Cache quant params used in every dispatch_expert_linear call.
    h.expert_quant_num_bits = moe_config_.quant_config.weights.num_bits;
    h.expert_quant_group_size = moe_config_.quant_config.weights.group_size;
    return h;
}

size_t Qwen3_5MoeModel::num_parameters() const {
    // Count tensors still held by *weights_* (embeddings, attention, shared experts, LM head, ...).
    size_t total = 0;
    for (const auto& [name, tensor] : weights_->get_all_weights()) {
        (void)name;
        if (tensor) {
            total += tensor->numel();
        }
    }
    // Plus the per-(layer, expert) FFN tensors that the pool now owns. Use peek_expert so
    // PINNED_LRU doesn't churn the GPU slot arena just to enumerate sizes.
    if (expert_pool_) {
        for (size_t l = 0; l < expert_pool_->num_layers(); ++l) {
            for (size_t e = 0; e < expert_pool_->num_experts_per_layer(); ++e) {
                const auto& ffn = expert_pool_->peek_expert(static_cast<int>(l), static_cast<int>(e));
                for (auto t :
                     {ffn.gate_packed, ffn.gate_scale, ffn.gate_g_idx, ffn.gate_weight, ffn.up_packed, ffn.up_scale,
                      ffn.up_g_idx, ffn.up_weight, ffn.down_packed, ffn.down_scale, ffn.down_g_idx, ffn.down_weight}) {
                    if (t) {
                        total += t->numel();
                    }
                }
            }
        }
    }
    return total;
}

} // namespace zedinfer::model
