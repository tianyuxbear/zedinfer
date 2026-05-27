#include "frontend/models/mtp_module.hpp"

#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "backend/ops/ops.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "utils/types.hpp"

#include <plog/Log.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

// SIMPLIFIED single-token attention for MTP with empty past KV.
//
// For one token at the start of a fresh KV cache, the softmax over q@k.T
// degenerates to a single weight of 1.0, so the attention output is just
// v at the (sole) attended position. We bypass the full paged-attention
// kernel entirely and assemble attn_out[q_head, d] = v_self[q_head//rep, d]
// with the GQA replication factor rep = Hq / Hkv. mrope at position 0 is
// a no-op (angle = pos * freq = 0), so we also skip the rotation.
//
// Then apply attn_output_gate (sigmoid(gate) * attn) and o_proj as usual.
// Returns o_proj output [1, hidden_size].
//
// h_in:    [1, hidden]
// q_proj:  [2*Hq*Dh, hidden]  (Qwen3.5 q+gate doubled)
// k_proj:  [Hkv*Dh, hidden]
// v_proj:  [Hkv*Dh, hidden]
// o_proj:  [hidden, Hq*Dh]
// q_norm/k_norm: [Dh]   (1+w)
tensor_t mtp_attention_one_token(tensor_t h_in,
                                 tensor_t q_proj, tensor_t k_proj, tensor_t v_proj,
                                 tensor_t o_proj, tensor_t q_norm, tensor_t k_norm,
                                 const Qwen3_5MoEConfig& cfg, const ExecutorConfig& exec) {
    const size_t Hq  = cfg.num_attention_heads;
    const size_t Hkv = cfg.num_key_value_heads;
    const size_t Dh  = cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / Hq);
    const size_t q_dim  = Hq  * Dh;
    const size_t kv_dim = Hkv * Dh;
    const size_t rep = Hq / Hkv;

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // q_proj is reordered like the main full-attn path: rows [0, q_dim)
    // are query, rows [q_dim, 2*q_dim) are output gate (see
    // reorder_q_proj_weight in qwen3_5.cpp). Slice once at call time.
    auto w_q    = q_proj->slice(0, 0, q_dim);
    auto w_gate = q_proj->slice(0, q_dim, 2 * q_dim);

    auto q_raw = make({1, q_dim});
    auto gate  = make({1, q_dim});
    auto k_raw = make({1, kv_dim});
    auto v     = make({1, kv_dim});
    ops::linear(q_raw, h_in, w_q);
    ops::linear(gate,  h_in, w_gate);
    ops::linear(k_raw, h_in, k_proj);
    ops::linear(v,     h_in, v_proj);

    // Per-head RMSNorm on q and k (Qwen3_5MoeRMSNorm with (1+w) at kernel time).
    auto q_normed = make({1, q_dim});
    auto k_normed = make({1, kv_dim});
    ops::rms_norm(q_normed->view({Hq,  Dh}), q_raw->view({Hq,  Dh}),
                  q_norm, cfg.rms_norm_eps, /*add_one_to_weight=*/true);
    ops::rms_norm(k_normed->view({Hkv, Dh}), k_raw->view({Hkv, Dh}),
                  k_norm, cfg.rms_norm_eps, /*add_one_to_weight=*/true);

    // (mrope skipped — position 0 = identity rotation)
    // (attention skipped — single token attends only to itself, output = v)
    //
    // Build attn = v replicated across query heads. Layout per token:
    //   attn[1, Hq, Dh] where attn[h, d] = v[h / rep, d]
    auto attn = make({1, Hq, Dh});
    auto* api = device::getRuntimeAPI(exec.device_type);
    const size_t elt = utils::dsize(exec.data_type);
    for (size_t h = 0; h < Hq; ++h) {
        const size_t kv_head = h / rep;
        // dest: attn->data() + (h * Dh) * elt
        // src:  v->data()    + (kv_head * Dh) * elt
        api->memcpy_sync(static_cast<std::byte*>(attn->data()) + h * Dh * elt,
                         static_cast<std::byte*>(v->data())    + kv_head * Dh * elt,
                         Dh * elt, ZEDINFER_MEMCPY_D2D);
    }

    // attn := sigmoid(gate) * attn   (in place on attn)
    ops::attn_output_gate(attn, gate);

    // o_proj: view attn as [1, q_dim] (head-major flatten matches the main path)
    auto out = make({1, cfg.hidden_size});
    ops::linear(out, attn->view({1, q_dim}), o_proj);
    return out;
}

// Inline MoE for MTP layer 0 — same router → top-k → expert FFN → shared
// expert pattern as moe_layer_forward in moe_forward.cpp, but consumes the
// MTPModule's own ExpertWeights (1 layer, no ExpertPool wrapper) and the
// MTP shared-expert tensors directly. N is always 1 in Stage B.1.
tensor_t mtp_moe_one_token(tensor_t h_post,
                           tensor_t router_w, tensor_t shared_gate_w,
                           tensor_t shared_gate_proj, tensor_t shared_up_proj,
                           tensor_t shared_down_proj,
                           const ExpertWeights& experts,
                           const Qwen3_5MoEConfig& cfg, const ExecutorConfig& exec) {
    const size_t H        = cfg.hidden_size;
    const size_t num_e    = static_cast<size_t>(cfg.num_experts);
    const size_t top_k    = static_cast<size_t>(cfg.num_experts_per_tok);
    const size_t moe_inter = static_cast<size_t>(cfg.moe_intermediate_size);
    const size_t shared_inter = static_cast<size_t>(cfg.shared_expert_intermediate_size);

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };
    auto* api = device::getRuntimeAPI(exec.device_type);

    // 1. Router: router_logits = h_post @ router_w.T
    auto router_logits = make({1, num_e});
    ops::linear(router_logits, h_post, router_w);

    // 2. D2H + bf16->fp32 + global softmax + top-k + renorm.
    //    num_e is small (256 for Qwen3.5), so do it on host.
    //    Qwen3_5MoeTopKRouter ALWAYS renormalizes (modeling_qwen3_5_moe.py:788),
    //    so pass true here too.
    std::vector<uint16_t> rl_bf16(num_e);
    api->memcpy_sync(rl_bf16.data(), router_logits->data(),
                     num_e * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    std::vector<float> rl_f32(num_e);
    for (size_t i = 0; i < num_e; ++i) {
        uint32_t u = static_cast<uint32_t>(rl_bf16[i]) << 16;
        std::memcpy(&rl_f32[i], &u, sizeof(float));
    }
    auto topk = ops::moe::topk_softmax(rl_f32.data(), /*N=*/1, num_e, top_k,
                                       /*norm_topk_prob=*/true);

    // 3. Output accumulator (fp32 on device for stable accumulation).
    auto out_bf16 = make({1, H});
    ops::fill_zero(out_bf16);

    // 4. Loop top-k experts. Each expert contributes
    //    delta = down(swiglu(gate(h), up(h))) and we accumulate
    //    out += topk_weights[k] * delta.
    auto gate_buf = make({1, moe_inter});
    auto up_buf   = make({1, moe_inter});
    auto act_buf  = make({1, moe_inter});
    auto down_buf = make({1, H});
    for (size_t k = 0; k < top_k; ++k) {
        const int    e_id = topk.expert_ids[k];
        const float  w_k  = topk.expert_weights[k];
        const auto& ffn = experts.at(/*layer=*/0, static_cast<size_t>(e_id));
        if (!ffn.gate_weight || !ffn.up_weight || !ffn.down_weight) {
            throw std::runtime_error("[MTPModule] expert " + std::to_string(e_id)
                                     + " missing bf16 weights (quantized MTP not supported yet)");
        }
        ops::linear(gate_buf, h_post, ffn.gate_weight);
        ops::linear(up_buf,   h_post, ffn.up_weight);
        ops::swiglu(act_buf, gate_buf, up_buf);
        ops::linear(down_buf, act_buf, ffn.down_weight);
        ops::add_scaled(out_bf16, down_buf, w_k);
    }

    // 5. Shared expert.
    auto sh_gate = make({1, shared_inter});
    auto sh_up   = make({1, shared_inter});
    auto sh_act  = make({1, shared_inter});
    auto sh_down = make({1, H});
    ops::linear(sh_gate, h_post, shared_gate_proj);
    ops::linear(sh_up,   h_post, shared_up_proj);
    ops::swiglu(sh_act, sh_gate, sh_up);
    ops::linear(sh_down, sh_act, shared_down_proj);

    // 6. Shared expert gate (sigmoid scalar from linear [1,1]).
    auto sh_gate_logit = make({1, 1});
    ops::linear(sh_gate_logit, h_post, shared_gate_w);
    uint16_t sgl_bf16 = 0;
    api->memcpy_sync(&sgl_bf16, sh_gate_logit->data(), sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    uint32_t u_sgl = static_cast<uint32_t>(sgl_bf16) << 16;
    float    sgl_f32 = 0.0f;
    std::memcpy(&sgl_f32, &u_sgl, sizeof(float));
    const float sh_gate_val = 1.0f / (1.0f + std::exp(-sgl_f32));

    // 7. out += sh_gate_val * sh_down
    ops::add_scaled(out_bf16, sh_down, sh_gate_val);

    return out_bf16;
}

} // namespace

tensor_t MTPModule::forward(tensor_t hidden_at_t, int next_token_id,
                            tensor_t embed_tokens_w, tensor_t lm_head_w,
                            const ExecutorConfig& exec) const {
    if (!ready_) {
        throw std::runtime_error("[MTPModule] forward called but module is not ready "
                                 "(model did not ship MTP weights)");
    }
    if (!hidden_at_t || hidden_at_t->ndim() != 2
        || hidden_at_t->shape()[0] != 1
        || hidden_at_t->shape()[1] != main_cfg_.hidden_size) {
        throw std::runtime_error("[MTPModule] hidden_at_t must be shape [1, hidden_size]");
    }

    const size_t H = main_cfg_.hidden_size;

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };
    auto* api = device::getRuntimeAPI(exec.device_type);
    const size_t elt = utils::dsize(exec.data_type);

    // 1. Embed lookup for the just-sampled main token. ops::embedding takes a
    //    1-D int32 index tensor and an [vocab, hidden] weight; we wrap the
    //    single id in a one-element device tensor.
    auto id_tensor = Tensor::create({1}, ZEDINFER_DTYPE_I32, exec.device_type, exec.device_id);
    const int32_t id32 = next_token_id;
    api->memcpy_sync(id_tensor->data(), &id32, sizeof(int32_t), ZEDINFER_MEMCPY_H2D);
    auto emb = make({1, H});
    ops::embedding(emb, id_tensor, embed_tokens_w);

    // 2. Pre-fc norms: (1+w) RMSNorm on emb and on hidden_at_t.
    auto norm_e = make({1, H});
    auto norm_h = make({1, H});
    ops::rms_norm(norm_e, emb,          pre_fc_norm_embedding_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);
    ops::rms_norm(norm_h, hidden_at_t,  pre_fc_norm_hidden_,    main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 3. Concatenate norm_e and norm_h along the last dim → [1, 2*H].
    //    Manual D2D memcpy: row layout is [emb part | hidden part].
    auto concat = make({1, 2 * H});
    api->memcpy_sync(static_cast<std::byte*>(concat->data()),
                     static_cast<std::byte*>(norm_e->data()),
                     H * elt, ZEDINFER_MEMCPY_D2D);
    api->memcpy_sync(static_cast<std::byte*>(concat->data()) + H * elt,
                     static_cast<std::byte*>(norm_h->data()),
                     H * elt, ZEDINFER_MEMCPY_D2D);

    // 4. fc projection: 2*H -> H.
    auto h_in_raw = make({1, H});
    ops::linear(h_in_raw, concat, fc_weight_);

    // 5. Pre-attention norm (Qwen3_5MoeRMSNorm with (1+w)).
    auto h_in = make({1, H});
    ops::rms_norm(h_in, h_in_raw, in_layernorm_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 6. Attention block (simplified: single token, empty past KV).
    auto attn_out = mtp_attention_one_token(h_in, q_proj_, k_proj_, v_proj_,
                                            o_proj_, q_norm_, k_norm_,
                                            main_cfg_, exec);

    // 7. Residual after attention.
    auto h1 = make({1, H});
    ops::add(h1, h_in_raw, attn_out);

    // 8. Post-attention norm.
    auto h_post = make({1, H});
    ops::rms_norm(h_post, h1, post_layernorm_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 9. MoE block.
    auto mlp_out = mtp_moe_one_token(h_post, mlp_gate_router_, shared_expert_gate_,
                                     shared_expert_gate_proj_, shared_expert_up_proj_,
                                     shared_expert_down_proj_, *experts_,
                                     main_cfg_, exec);

    // 10. Residual after MoE.
    auto h_after_moe = make({1, H});
    ops::add(h_after_moe, h1, mlp_out);

    // 11. Final norm + lm_head.
    auto final_normed = make({1, H});
    ops::rms_norm(final_normed, h_after_moe, final_norm_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);
    auto logits = make({1, static_cast<size_t>(main_cfg_.vocab_size)});
    ops::linear(logits, final_normed, lm_head_w);
    return logits;
}

} // namespace zedinfer::model
