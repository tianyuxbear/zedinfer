#include "frontend/models/mtp_module.hpp"

#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/ops/mrope/mrope_3d.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/shared_expert_gate/shared_expert_gate.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/hybrid_forward_config.hpp"  // MRoPEConfig
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

// Persistent per-thread scratch for the MTP forward path. The original
// MTPModule::forward + mtp_attention_decode + mtp_moe_one_token make ~28
// Tensor::create calls per invocation. Under ALL_GPU the BestFitMemoryPool
// is heavily populated by ~14 GB of expert weights, so per-call alloc
// round-trips cost ~30+ ms cumulatively — the MTP "1-layer" forward
// measured at ~35 ms total per the Stage E perf write-up. This scratch
// makes the steady-state per-call cost just the actual kernel work.
//
// Sized at the model's MTP shape on first use; reallocated only if the shape
// changes (which it doesn't post-init).
struct MTPScratch {
    bool ready = false;
    // top of MTPModule::forward
    tensor_t id_tensor;     // [1]   I32
    tensor_t emb;           // [1, H]
    tensor_t norm_e;        // [1, H]
    tensor_t norm_h;        // [1, H]
    tensor_t concat;        // [1, 2H]
    tensor_t h_in_raw;      // [1, H]
    tensor_t h_in;          // [1, H]
    tensor_t h1;            // [1, H]
    tensor_t h_post;        // [1, H]
    tensor_t h_after_moe;   // [1, H]
    tensor_t final_normed;  // [1, H]
    tensor_t logits;        // [1, V]

    // mtp_attention_decode
    tensor_t att_q_raw;     // [1, q_dim]
    tensor_t att_gate;      // [1, q_dim]
    tensor_t att_k_raw;     // [1, kv_dim]
    tensor_t att_v;         // [1, kv_dim]
    tensor_t att_q_normed;  // [1, q_dim]
    tensor_t att_k_normed;  // [1, kv_dim]
    tensor_t att_pos_thw;   // [3, 1] I32
    tensor_t att_attn;      // [1, Hq, Dh]
    tensor_t att_out;       // [1, H]

    // mtp_moe_one_token
    tensor_t moe_router_logits; // [1, num_experts]
    tensor_t moe_out;           // [1, H]
    tensor_t moe_gate_buf;      // [1, moe_inter]
    tensor_t moe_up_buf;        // [1, moe_inter]
    tensor_t moe_act_buf;       // [1, moe_inter]
    tensor_t moe_down_buf;      // [1, H]
    tensor_t moe_sh_gate;       // [1, shared_inter]
    tensor_t moe_sh_up;         // [1, shared_inter]
    tensor_t moe_sh_act;        // [1, shared_inter]
    tensor_t moe_sh_down;       // [1, H]
    tensor_t moe_sh_gate_logit; // [1, 1]

    // mtp_dense_ffn_one_token (dense MTP layer: 27B). Sized [1, dense_inter].
    tensor_t ffn_gate;          // [1, dense_inter]
    tensor_t ffn_up;            // [1, dense_inter]
    tensor_t ffn_act;           // [1, dense_inter]
    tensor_t ffn_down;          // [1, H]

    // Cached shape signature.
    bool   is_moe = true;
    size_t H = 0, q_dim = 0, kv_dim = 0, V = 0;
    size_t Hq = 0, Dh = 0;
    size_t num_experts = 0, moe_inter = 0, shared_inter = 0, dense_inter = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int    device_id = -1;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_F32;
};

static thread_local MTPScratch s_mtp_scratch;

static MTPScratch& ensure_mtp_scratch(const Qwen3_5Config& cfg, const ExecutorConfig& exec,
                                      bool is_moe, size_t dense_inter) {
    const size_t H = cfg.hidden_size;
    const size_t Hq  = cfg.num_attention_heads;
    const size_t Hkv = cfg.num_key_value_heads;
    const size_t Dh  = cfg.head_dim > 0 ? cfg.head_dim : (H / Hq);
    const size_t q_dim   = Hq  * Dh;
    const size_t kv_dim  = Hkv * Dh;
    const size_t V       = cfg.vocab_size;
    // MoE-only fields live on the derived Qwen3_5MoEConfig; read them only on
    // the MoE path (the MoE model passes a real Qwen3_5MoEConfig here).
    size_t num_e = 0, moe_int = 0, sh_int = 0;
    if (is_moe) {
        const auto& mcfg = static_cast<const Qwen3_5MoEConfig&>(cfg);
        num_e   = static_cast<size_t>(mcfg.num_experts);
        moe_int = static_cast<size_t>(mcfg.moe_intermediate_size);
        sh_int  = static_cast<size_t>(mcfg.shared_expert_intermediate_size);
    }

    auto& s = s_mtp_scratch;
    if (s.ready && s.is_moe == is_moe && s.H == H && s.q_dim == q_dim && s.kv_dim == kv_dim && s.V == V
        && s.Hq == Hq && s.Dh == Dh && s.num_experts == num_e && s.moe_inter == moe_int && s.shared_inter == sh_int
        && s.dense_inter == dense_inter
        && s.device_type == exec.device_type && s.device_id == exec.device_id && s.dtype == exec.data_type) {
        return s;
    }
    auto mkf = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };
    auto mk  = [&](std::vector<size_t> shape, zedinferDataType_t t) {
        return Tensor::create(std::move(shape), t, exec.device_type, exec.device_id);
    };

    s.id_tensor    = mk({1}, ZEDINFER_DTYPE_I32);
    s.emb          = mkf({1, H});
    s.norm_e       = mkf({1, H});
    s.norm_h       = mkf({1, H});
    s.concat       = mkf({1, 2 * H});
    s.h_in_raw     = mkf({1, H});
    s.h_in         = mkf({1, H});
    s.h1           = mkf({1, H});
    s.h_post       = mkf({1, H});
    s.h_after_moe  = mkf({1, H});
    s.final_normed = mkf({1, H});
    s.logits       = mkf({1, V});

    s.att_q_raw    = mkf({1, q_dim});
    s.att_gate     = mkf({1, q_dim});
    s.att_k_raw    = mkf({1, kv_dim});
    s.att_v        = mkf({1, kv_dim});
    s.att_q_normed = mkf({1, q_dim});
    s.att_k_normed = mkf({1, kv_dim});
    s.att_pos_thw  = mk({3, 1}, ZEDINFER_DTYPE_I32);
    s.att_attn     = mkf({1, Hq, Dh});
    s.att_out      = mkf({1, H});

    if (is_moe) {
        s.moe_router_logits = mkf({1, num_e});
        s.moe_out           = mkf({1, H});
        s.moe_gate_buf      = mkf({1, moe_int});
        s.moe_up_buf        = mkf({1, moe_int});
        s.moe_act_buf       = mkf({1, moe_int});
        s.moe_down_buf      = mkf({1, H});
        if (sh_int > 0) {
            s.moe_sh_gate       = mkf({1, sh_int});
            s.moe_sh_up         = mkf({1, sh_int});
            s.moe_sh_act        = mkf({1, sh_int});
            s.moe_sh_down       = mkf({1, H});
            s.moe_sh_gate_logit = mkf({1, 1});
        }
    } else {
        s.ffn_gate = mkf({1, dense_inter});
        s.ffn_up   = mkf({1, dense_inter});
        s.ffn_act  = mkf({1, dense_inter});
        s.ffn_down = mkf({1, H});
    }
    s.is_moe = is_moe;
    s.H = H; s.q_dim = q_dim; s.kv_dim = kv_dim; s.V = V; s.Hq = Hq; s.Dh = Dh;
    s.num_experts = num_e; s.moe_inter = moe_int; s.shared_inter = sh_int; s.dense_inter = dense_inter;
    s.device_type = exec.device_type;
    s.device_id   = exec.device_id;
    s.dtype       = exec.data_type;
    s.ready       = true;
    LOGI.printf("[MTPScratch] allocated H=%zu V=%zu q_dim=%zu kv_dim=%zu is_moe=%d num_e=%zu moe_int=%zu sh_int=%zu dense_int=%zu",
                H, V, q_dim, kv_dim, (int)is_moe, num_e, moe_int, sh_int, dense_inter);
    return s;
}

} // anonymous namespace (MTPScratch only)

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

MTPModule::MTPModule(const Qwen3_5Config& main_cfg, ModelWeights& weights, const ExecutorConfig& exec)
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

    // ----- FFN: MoE (35B-A3B) or dense (27B), detected from the weights -----
    // MoE ships "mtp.layers.0.mlp.gate.weight" (the router); the dense 27B
    // ships "mtp.layers.0.mlp.gate_proj.weight" (a plain FFN) instead.
    is_moe_ = weights.has_tensor(p + "mlp.gate.weight");
    if (is_moe_) {
        // MoE router + shared expert. main_cfg_ is really a Qwen3_5MoEConfig
        // here (the MoE model passed it); read the MoE fields via static_cast.
        const auto& moe_cfg = static_cast<const Qwen3_5MoEConfig&>(main_cfg_);
        mlp_gate_router_           = fetch(weights, p + "mlp.gate.weight");
        shared_expert_gate_        = fetch(weights, p + "mlp.shared_expert_gate.weight");
        shared_expert_gate_proj_   = fetch(weights, p + "mlp.shared_expert.gate_proj.weight");
        shared_expert_up_proj_     = fetch(weights, p + "mlp.shared_expert.up_proj.weight");
        shared_expert_down_proj_   = fetch(weights, p + "mlp.shared_expert.down_proj.weight");

        const size_t num_experts = static_cast<size_t>(moe_cfg.num_experts);
        if (num_experts == 0) {
            throw std::runtime_error("[MTPModule] main_cfg.num_experts is 0; cannot build MTP MoE");
        }
        experts_ = std::make_unique<ExpertWeights>(/*num_layers=*/1, num_experts);
        wire_mtp_experts(weights, *experts_, num_experts);
    } else {
        // Dense FFN MTP layer (Qwen3.5/3.6-27B): a single gate/up/down_proj.
        mlp_gate_proj_ = fetch(weights, p + "mlp.gate_proj.weight");
        mlp_up_proj_   = fetch(weights, p + "mlp.up_proj.weight");
        mlp_down_proj_ = fetch(weights, p + "mlp.down_proj.weight");
        dense_inter_   = mlp_gate_proj_->shape()[0]; // gate_proj is [inter, H]
    }

    // ----- Final norm -----
    final_norm_ = fetch(weights, "mtp.norm.weight");

    ready_ = true;
    LOGI.printf("[MTPModule] loaded: 1 transformer layer + fc(2H->H) + final_norm; FFN=%s inter=%zu",
                is_moe_ ? "MoE" : "dense", dense_inter_);
}

MTPModule::~MTPModule() = default;

tensor_t MTPModule::prefill(InferenceRequest& req,
                            tensor_t hidden_main_seq,
                            const std::vector<int>& next_tokens,
                            tensor_t embed_tokens_w, tensor_t lm_head_w,
                            const ExecutorConfig& exec) const {
    if (!ready_) {
        throw std::runtime_error("[MTPModule] prefill called but module not ready");
    }
    if (!hidden_main_seq || hidden_main_seq->ndim() != 2) {
        throw std::runtime_error("[MTPModule] prefill expects 2-D hidden_main_seq [P, hidden]");
    }
    const size_t P = hidden_main_seq->shape()[0];
    if (next_tokens.size() != P) {
        throw std::runtime_error("[MTPModule] prefill: next_tokens.size()="
                                 + std::to_string(next_tokens.size())
                                 + " does not match P=" + std::to_string(P));
    }
    // Loop forward() per position. Each call advances req.mtp_past_seq_len
    // by 1. Intermediate logits are discarded; only the final (P-1)
    // prediction is returned. Sequential — Stage E will batch this into
    // a single attention call with seqlen_q = P.
    tensor_t last_logits;
    for (size_t i = 0; i < P; ++i) {
        auto row = hidden_main_seq->slice(0, i, i + 1);
        last_logits = forward(req, row, next_tokens[i], embed_tokens_w, lm_head_w, exec);
    }
    return last_logits;
}

namespace {

// Single-token decode attention for MTP with a growing K/V cache.
//
// K and V caches are laid out as [max_kv_len, Hkv*Dh] bf16 contiguous, but
// we reuse ops::attention with the paged-decode dispatch by treating the
// whole cache as a single "page" of size max_kv_len with page_table=[0].
// That way the existing `paged_attention_decode_kernel` reads positions
// 0..past at byte offset `pos * Hkv*Dh*elt` without any new GPU code.
//
// h_in:     [1, hidden]
// k_cache, v_cache: [max_kv_len, Hkv*Dh] bf16 (mutated: writes at row `past`)
// page_table_dev: device int [1] = {0}
// past:     number of positions ALREADY in the cache before this call
// Returns o_proj output [1, hidden_size].
tensor_t mtp_attention_decode(tensor_t h_in,
                              tensor_t q_proj, tensor_t k_proj, tensor_t v_proj,
                              tensor_t o_proj, tensor_t q_norm, tensor_t k_norm,
                              tensor_t k_cache, tensor_t v_cache,
                              tensor_t page_table_dev, int past, int max_kv_len,
                              const Qwen3_5MoEConfig& cfg, const ExecutorConfig& exec,
                              MTPScratch& s) {
    const size_t Hq  = cfg.num_attention_heads;
    const size_t Hkv = cfg.num_key_value_heads;
    const size_t Dh  = cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / Hq);
    const size_t q_dim  = Hq  * Dh;
    const size_t kv_dim = Hkv * Dh;

    auto* api = device::getRuntimeAPI(exec.device_type);
    const size_t elt = utils::dsize(exec.data_type);

    // q_proj is reordered like main full-attn (rows [0,q_dim)=q, [q_dim,2q_dim)=gate).
    auto w_q    = q_proj->slice(0, 0, q_dim);
    auto w_gate = q_proj->slice(0, q_dim, 2 * q_dim);

    auto& q_raw = s.att_q_raw;
    auto& gate  = s.att_gate;
    auto& k_raw = s.att_k_raw;
    auto& v     = s.att_v;
    ops::linear(q_raw, h_in, w_q);
    ops::linear(gate,  h_in, w_gate);
    ops::linear(k_raw, h_in, k_proj);
    ops::linear(v,     h_in, v_proj);

    auto& q_normed = s.att_q_normed;
    auto& k_normed = s.att_k_normed;
    ops::rms_norm(q_normed->view({Hq,  Dh}), q_raw->view({Hq,  Dh}),
                  q_norm, cfg.rms_norm_eps, /*add_one_to_weight=*/true);
    ops::rms_norm(k_normed->view({Hkv, Dh}), k_raw->view({Hkv, Dh}),
                  k_norm, cfg.rms_norm_eps, /*add_one_to_weight=*/true);

    // mrope_3d at position `past`. Build [3, 1] int32 = {past, past, past}.
    // Reuse the cached pos_thw tensor — its three int32 slots get refilled
    // every call. memcpy_sync is fine here: 12 bytes, host-side stack data.
    {
        int32_t host_pos[3] = {past, past, past};
        api->memcpy_sync(s.att_pos_thw->data(), host_pos, 3 * sizeof(int32_t), ZEDINFER_MEMCPY_H2D);
    }
    MRoPEConfig mrope_cfg;
    mrope_cfg.interleaved    = cfg.mrope_interleaved;
    mrope_cfg.section        = cfg.mrope_section;
    mrope_cfg.partial_factor = cfg.partial_rotary_factor;
    mrope_cfg.theta          = cfg.rope_theta;
    ops::mrope_3d(q_normed->view({1, Hq,  Dh}), s.att_pos_thw, mrope_cfg);
    ops::mrope_3d(k_normed->view({1, Hkv, Dh}), s.att_pos_thw, mrope_cfg);

    // Write rotated k_normed and raw v into K/V cache at row `past`.
    // Use the compute stream's async memcpy so we don't drain the stream between
    // the projections we just issued and the attention kernel that comes next.
    // The kernel and the copies share one stream, so FIFO ordering still holds.
    auto compute_stream = core::context().runtime().stream();
    api->memcpy_async(static_cast<std::byte*>(k_cache->data())
                          + static_cast<size_t>(past) * kv_dim * elt,
                      k_normed->data(), kv_dim * elt, ZEDINFER_MEMCPY_D2D, compute_stream);
    api->memcpy_async(static_cast<std::byte*>(v_cache->data())
                          + static_cast<size_t>(past) * kv_dim * elt,
                      v->data(), kv_dim * elt, ZEDINFER_MEMCPY_D2D, compute_stream);

    // Configure single-page paged attention.
    ops::AttentionConfig acfg{
        /*nhead=*/static_cast<int>(Hq),
        /*nkvhead=*/static_cast<int>(Hkv),
        /*head_dim=*/static_cast<int>(Dh),
        /*scale=*/1.0f / std::sqrt(static_cast<float>(Dh)),
        /*block_size=*/max_kv_len,
        /*dtype=*/exec.data_type,
        /*device_type=*/exec.device_type,
        /*device_id=*/exec.device_id,
    };
    auto& attn = s.att_attn;
    ops::AttentionParams params{acfg};
    params.use_flashinfer = false; // single-block + single-request → native paged decode is enough
    params.out = attn->view({Hq, Dh});
    params.q   = q_normed->view({Hq, Dh});
    params.k_pool_base = k_cache->data();
    params.v_pool_base = v_cache->data();
    params.page_table  = reinterpret_cast<const int*>(page_table_dev->data());
    params.seq_len     = past + 1;
    params.seqlen_q    = 1;
    ops::attention(params);

    // attn := sigmoid(gate) * attn   (in place)
    ops::attn_output_gate(attn, gate);

    // o_proj
    auto& out = s.att_out;
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
                           const Qwen3_5MoEConfig& cfg, const ExecutorConfig& exec,
                           MTPScratch& s) {
    (void)exec;
    const size_t num_e = s.num_experts;
    const size_t top_k = static_cast<size_t>(cfg.num_experts_per_tok);

    auto* api = device::getRuntimeAPI(s.device_type);

    // 1. Router: router_logits = h_post @ router_w.T
    auto& router_logits = s.moe_router_logits;
    ops::linear(router_logits, h_post, router_w);

    // 2. D2H + bf16->fp32 + global softmax + top-k + renorm.
    //    num_e is small (256 for Qwen3.5), so do it on host.
    //    Qwen3_5MoeTopKRouter ALWAYS renormalizes (modeling_qwen3_5_moe.py:788),
    //    so pass true here too. Thread_local pinned vector avoids per-call heap
    //    alloc — the D2H sync itself is unavoidable (CPU top-k consumer).
    static thread_local std::vector<uint16_t> rl_bf16_buf;
    static thread_local std::vector<float>    rl_f32_buf;
    if (rl_bf16_buf.size() < num_e) rl_bf16_buf.resize(num_e);
    if (rl_f32_buf.size()  < num_e) rl_f32_buf.resize(num_e);
    api->memcpy_sync(rl_bf16_buf.data(), router_logits->data(),
                     num_e * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    for (size_t i = 0; i < num_e; ++i) {
        uint32_t u = static_cast<uint32_t>(rl_bf16_buf[i]) << 16;
        std::memcpy(&rl_f32_buf[i], &u, sizeof(float));
    }
    auto topk = ops::moe::topk_softmax(rl_f32_buf.data(), /*N=*/1, num_e, top_k,
                                       /*norm_topk_prob=*/true);

    // 3. Output accumulator (bf16 on device).
    auto& out_bf16 = s.moe_out;
    ops::fill_zero(out_bf16);

    // 4. Loop top-k experts. Each expert contributes
    //    delta = down(swiglu(gate(h), up(h))) and we accumulate
    //    out += topk_weights[k] * delta.
    for (size_t k = 0; k < top_k; ++k) {
        const int    e_id = topk.expert_ids[k];
        const float  w_k  = topk.expert_weights[k];
        const auto& ffn = experts.at(/*layer=*/0, static_cast<size_t>(e_id));
        if (!ffn.gate_weight || !ffn.up_weight || !ffn.down_weight) {
            throw std::runtime_error("[MTPModule] expert " + std::to_string(e_id)
                                     + " missing bf16 weights (quantized MTP not supported yet)");
        }
        ops::linear(s.moe_gate_buf, h_post, ffn.gate_weight);
        ops::linear(s.moe_up_buf,   h_post, ffn.up_weight);
        ops::swiglu(s.moe_act_buf, s.moe_gate_buf, s.moe_up_buf);
        ops::linear(s.moe_down_buf, s.moe_act_buf, ffn.down_weight);
        ops::add_scaled(out_bf16, s.moe_down_buf, w_k);
    }

    // 5. Shared expert.
    auto& sh_gate = s.moe_sh_gate;
    auto& sh_up   = s.moe_sh_up;
    auto& sh_act  = s.moe_sh_act;
    auto& sh_down = s.moe_sh_down;
    ops::linear(sh_gate, h_post, shared_gate_proj);
    ops::linear(sh_up,   h_post, shared_up_proj);
    ops::swiglu(sh_act, sh_gate, sh_up);
    ops::linear(sh_down, sh_act, shared_down_proj);

    // 6. Shared expert gate (in-place on GPU via shared_expert_gate op).
    auto& sh_gate_logit = s.moe_sh_gate_logit;
    ops::linear(sh_gate_logit, h_post, shared_gate_w);
    ops::shared_expert_gate(sh_down, sh_gate_logit);

    // 7. out += sh_down (already gated by sigmoid above)
    ops::add_scaled(out_bf16, sh_down, 1.0f);

    return out_bf16;
}

// Dense MTP FFN (Qwen3.5/3.6-27B): out = down_proj(swiglu(gate_proj(h), up_proj(h))).
// The 27B's MTP layer has a plain MLP (no router / experts / shared expert), so
// this is just the single-expert path the MoE loop runs per expert.
tensor_t mtp_dense_ffn_one_token(tensor_t h_post,
                                 tensor_t gate_proj, tensor_t up_proj, tensor_t down_proj,
                                 MTPScratch& s) {
    ops::linear(s.ffn_gate, h_post, gate_proj);
    ops::linear(s.ffn_up,   h_post, up_proj);
    ops::swiglu(s.ffn_act,  s.ffn_gate, s.ffn_up);
    ops::linear(s.ffn_down, s.ffn_act, down_proj);
    return s.ffn_down;
}

} // namespace

// Public helper — lazily allocates request's MTP buffers, resets past.
void mtp_reset_request_state(InferenceRequest& req,
                             size_t max_kv_len,
                             size_t num_kv_heads, size_t head_dim,
                             const ExecutorConfig& exec) {
    const size_t kv_dim = num_kv_heads * head_dim;
    if (!req.mtp_k_cache) {
        req.mtp_k_cache = Tensor::create({max_kv_len, kv_dim}, exec.data_type,
                                         exec.device_type, exec.device_id);
        req.mtp_v_cache = Tensor::create({max_kv_len, kv_dim}, exec.data_type,
                                         exec.device_type, exec.device_id);
        req.mtp_page_table_dev = Tensor::create({1}, ZEDINFER_DTYPE_I32,
                                                exec.device_type, exec.device_id);
        const int32_t zero = 0;
        auto* api = device::getRuntimeAPI(exec.device_type);
        api->memcpy_sync(req.mtp_page_table_dev->data(), &zero, sizeof(int32_t),
                         ZEDINFER_MEMCPY_H2D);
    }
    req.mtp_past_seq_len    = 0;
    req.mtp_pending_draft   = -1;
}

tensor_t MTPModule::forward(InferenceRequest& req,
                            tensor_t hidden_at_t, int next_token_id,
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

    // Persistent per-thread scratch. First call allocates ~28 [1, ...] buffers;
    // subsequent calls just reuse them. Without this each MTP call paid ~30 ms
    // of BestFitPool round-trips alone (the Stage E perf write-up has the
    // detailed breakdown).
    auto& s = ensure_mtp_scratch(main_cfg_, exec, is_moe_, dense_inter_);

    auto* api = device::getRuntimeAPI(exec.device_type);
    const size_t elt = utils::dsize(exec.data_type);

    // 1. Embed lookup for the just-sampled main token. Reuse the scratch
    //    [1] I32 buffer instead of fresh-allocating an id tensor.
    const int32_t id32 = next_token_id;
    api->memcpy_sync(s.id_tensor->data(), &id32, sizeof(int32_t), ZEDINFER_MEMCPY_H2D);
    ops::embedding(s.emb, s.id_tensor, embed_tokens_w);

    // 2. Pre-fc norms: (1+w) RMSNorm on emb and on hidden_at_t.
    ops::rms_norm(s.norm_e, s.emb,        pre_fc_norm_embedding_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);
    ops::rms_norm(s.norm_h, hidden_at_t,  pre_fc_norm_hidden_,    main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 3. Concatenate norm_e and norm_h along the last dim → [1, 2*H].
    //    Manual D2D memcpy: row layout is [emb part | hidden part]. Run async
    //    on the compute stream — both halves are independent and the fc linear
    //    that consumes `concat` is enqueued after on the same stream.
    auto compute_stream = core::context().runtime().stream();
    api->memcpy_async(static_cast<std::byte*>(s.concat->data()),
                      static_cast<std::byte*>(s.norm_e->data()),
                      H * elt, ZEDINFER_MEMCPY_D2D, compute_stream);
    api->memcpy_async(static_cast<std::byte*>(s.concat->data()) + H * elt,
                      static_cast<std::byte*>(s.norm_h->data()),
                      H * elt, ZEDINFER_MEMCPY_D2D, compute_stream);

    // 4. fc projection: 2*H -> H.
    ops::linear(s.h_in_raw, s.concat, fc_weight_);

    // 5. Pre-attention norm (Qwen3_5MoeRMSNorm with (1+w)).
    ops::rms_norm(s.h_in, s.h_in_raw, in_layernorm_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 6. Attention with this request's K/V cache. Caller must have
    //    invoked mtp_reset_request_state(req, ...) once when admitting
    //    the request; that lazily allocates req.mtp_k_cache / v_cache
    //    and zeros req.mtp_past_seq_len.
    if (!req.mtp_k_cache) {
        const size_t Hkv = main_cfg_.num_key_value_heads;
        const size_t Dh  = main_cfg_.head_dim > 0 ? main_cfg_.head_dim
                                                  : (main_cfg_.hidden_size / main_cfg_.num_attention_heads);
        mtp_reset_request_state(req, max_kv_len_, Hkv, Dh, exec);
        LOGI.printf("[MTPModule] lazy-init request MTP K/V cache: %zu x %zu bf16",
                    max_kv_len_, Hkv * Dh);
    }
    if (static_cast<size_t>(req.mtp_past_seq_len) >= max_kv_len_) {
        throw std::runtime_error("[MTPModule] req.mtp_past_seq_len="
                                 + std::to_string(req.mtp_past_seq_len)
                                 + " >= max=" + std::to_string(max_kv_len_));
    }
    auto attn_out = mtp_attention_decode(s.h_in, q_proj_, k_proj_, v_proj_,
                                         o_proj_, q_norm_, k_norm_,
                                         req.mtp_k_cache, req.mtp_v_cache,
                                         req.mtp_page_table_dev,
                                         req.mtp_past_seq_len,
                                         static_cast<int>(max_kv_len_),
                                         main_cfg_, exec, s);
    req.mtp_past_seq_len += 1;

    // 7. Residual after attention.
    ops::add(s.h1, s.h_in_raw, attn_out);

    // 8. Post-attention norm.
    ops::rms_norm(s.h_post, s.h1, post_layernorm_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 9. FFN block: MoE (35B-A3B) or dense (27B).
    tensor_t mlp_out;
    if (is_moe_) {
        mlp_out = mtp_moe_one_token(s.h_post, mlp_gate_router_, shared_expert_gate_,
                                    shared_expert_gate_proj_, shared_expert_up_proj_,
                                    shared_expert_down_proj_, *experts_,
                                    static_cast<const Qwen3_5MoEConfig&>(main_cfg_), exec, s);
    } else {
        mlp_out = mtp_dense_ffn_one_token(s.h_post, mlp_gate_proj_, mlp_up_proj_, mlp_down_proj_, s);
    }

    // 10. Residual after MoE.
    ops::add(s.h_after_moe, s.h1, mlp_out);

    // 11. Final norm + lm_head.
    ops::rms_norm(s.final_normed, s.h_after_moe, final_norm_, main_cfg_.rms_norm_eps,
                  /*add_one_to_weight=*/true);
    ops::linear(s.logits, s.final_normed, lm_head_w);
    return s.logits;
}

} // namespace zedinfer::model
