#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/expert_weights.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include "zedinfer/activation.hpp" // ExecutorConfig (struct definition)
#include "zedinfer/request.hpp"    // InferenceRequest (mtp_k_cache, etc.)

#include <memory>
#include <vector>

namespace zedinfer::model {

// Qwen3.5 Multi-Token Prediction (MTP) head — predicts token t+2 given the
// main model's hidden state at position t and the token the main model just
// sampled at position t+1. Used for speculative decoding (NextN=1).
//
// Architecture (single MTP module; mtp_num_hidden_layers=1 for all known
// Qwen3.5/3.6 releases):
//
//   emb     = embed_tokens(next_token)              # reuse main embed weight
//   norm_e  = (1+pre_fc_norm_embedding) * RMSNorm(emb)
//   norm_h  = (1+pre_fc_norm_hidden)    * RMSNorm(hidden_at_t)
//   h       = fc( concat([norm_e, norm_h]) )        # [2*hidden -> hidden]
//   h       = mtp.layers.0(h)                       # 1 full-attn + 1 MoE block
//   h       = (1+mtp.norm) * RMSNorm(h)
//   logits  = main.lm_head(h)                       # reuse main lm_head
//
// MTP layer 0's attention is FULL-attention (not linear); it has its own
// KV cache (1 layer's worth). Stage A only builds + verifies + stubs the
// forward — speculative scheduling integration lands in Stage B.
class MTPModule {
public:
    // Verifies that all mtp.* tensors are present in `weights`, slices the
    // fused expert tensors into per-expert views (same trick as the main
    // MoE loader), constructs an ExpertWeights pool for layer 0's experts,
    // and stashes pointers to the standalone tensors (fc, norms).
    // Throws std::runtime_error with the missing key name on any miss.
    // main_cfg is the base Qwen3.5 config; the MTP layer's FFN is dense
    // (mtp.layers.0.mlp.{gate,up,down}_proj — Qwen3.5/3.6-27B) or MoE
    // (mtp.layers.0.mlp.gate + experts + shared_expert — 35B-A3B), detected
    // from the weights. The MoE branch reads expert-routing fields by
    // static_cast'ing main_cfg to Qwen3_5MoEConfig (valid: the MoE model
    // passes its Qwen3_5MoEConfig here).
    MTPModule(const Qwen3_5Config& main_cfg, ModelWeights& weights, const ExecutorConfig& exec);
    ~MTPModule();

    MTPModule(const MTPModule&) = delete;
    MTPModule& operator=(const MTPModule&) = delete;

    // Stage D.0: single-token decode forward — KV state lives on the
    // request, so concurrent requests run independently. The first call
    // for a request (mtp_past_seq_len == 0) sees no past context; later
    // calls accumulate as past_seq_len grows by 1 per call.
    //
    // Args:
    //   req:               request holding MTP K/V state (mutated)
    //   hidden_at_t:       [1, hidden_size] — main's PRE-final-norm residual
    //   next_token_id:     id sampled by main at position t+1
    //   embed_tokens_w:    main embed_tokens.weight [vocab, hidden] — reused
    //   lm_head_w:         main lm_head.weight [vocab, hidden] — reused
    //   exec:              compute device + dtype
    //
    // Returns: logits [1, vocab] — model's guess for token t+2.
    // Throws if !ready() OR if the request's cache is full (past >= max_kv_len_).
    tensor_t forward(InferenceRequest& req, tensor_t hidden_at_t, int next_token_id, tensor_t embed_tokens_w,
                     tensor_t lm_head_w, const ExecutorConfig& exec) const;

    // Stage C.2 / D.0: prefill MTP K/V across the prompt for this request.
    //
    // After main prefill of a P-token prompt, the caller gives us:
    //   hidden_main_seq: [P, hidden] — main's residuals at positions 0..P-1
    //   next_tokens:     size P, where next_tokens[i] is the token at
    //                    position i+1 (i.e. prompt[i+1] for i in 0..P-2,
    //                    and main's just-sampled t_P for i=P-1).
    //
    // Loops forward() per position; each call advances req.mtp_past_seq_len
    // by 1. Sequential — Stage E will batch this into a single attention
    // call with seqlen_q=P (~10x faster on the prompt).
    //
    // Returns the LAST-position logits [1, vocab], i.e. MTP's prediction
    // for token t_{P+1}.
    tensor_t prefill(InferenceRequest& req, tensor_t hidden_main_seq, const std::vector<int>& next_tokens,
                     tensor_t embed_tokens_w, tensor_t lm_head_w, const ExecutorConfig& exec) const;

    // Stage D max KV length per request (default matches Stage C.1's
    // single-tenant fixed buffer). Callers can override at engine init.
    size_t max_kv_len() const { return max_kv_len_; }

    // True iff a full MTP weight set was found at ctor time. False for
    // models that don't ship MTP (e.g. Qwen3 base, DeepSeek-R1 distill).
    // Callers (engine / scheduler) probe this before enabling spec decode.
    bool ready() const { return ready_; }

private:
    const Qwen3_5Config& main_cfg_;
    ExecutorConfig exec_;
    bool ready_ = false;
    // True if the MTP layer's FFN is MoE (experts + router + shared expert);
    // false if it is a plain dense FFN (gate/up/down_proj). Set in the ctor.
    bool is_moe_ = false;
    // Dense-FFN MTP path (is_moe_ == false): the single MLP's projections and
    // the inferred intermediate size. Null/0 on the MoE path.
    tensor_t mlp_gate_proj_; // [inter, H]
    tensor_t mlp_up_proj_;   // [inter, H]
    tensor_t mlp_down_proj_; // [H, inter]
    size_t dense_inter_ = 0;

    // Pre-fc projection norms (Qwen3_5MoeRMSNorm, (1+w) at kernel time).
    tensor_t pre_fc_norm_embedding_; // [hidden]
    tensor_t pre_fc_norm_hidden_;    // [hidden]

    // Fusion projection: 2*hidden -> hidden.
    tensor_t fc_weight_; // [hidden, 2*hidden]

    // Single transformer block (mtp.layers.0.*) — slice references into the
    // main ModelWeights map. Naming mirrors the main hybrid layer so the
    // forward (when implemented in Stage B) can reuse the per-layer helpers.
    tensor_t in_layernorm_;       // [hidden]   mtp.layers.0.input_layernorm.weight
    tensor_t post_layernorm_;     // [hidden]   mtp.layers.0.post_attention_layernorm.weight
    tensor_t q_proj_;             // [2*Hq*Dh, hidden]  (q + output-gate doubled, like main full-attn)
    tensor_t k_proj_;             // [Hkv*Dh, hidden]
    tensor_t v_proj_;             // [Hkv*Dh, hidden]
    tensor_t o_proj_;             // [hidden, Hq*Dh]
    tensor_t q_norm_;             // [Dh]      (1+w) per-head RMSNorm
    tensor_t k_norm_;             // [Dh]
    tensor_t mlp_gate_router_;    // [num_experts, hidden]  MoE router
    tensor_t shared_expert_gate_; // [1, hidden]            sigmoid gate for shared expert contribution
    tensor_t shared_expert_gate_proj_;
    tensor_t shared_expert_up_proj_;
    tensor_t shared_expert_down_proj_;

    // 256 experts for this single MTP layer, sliced per-expert from the fused
    // gate_up_proj / down_proj tensors. Owned (separate ExpertWeights from
    // the main model's). Stored in a one-layer ExpertWeights so we can reuse
    // moe_layer_forward in Stage B without touching its API.
    std::unique_ptr<ExpertWeights> experts_;

    // Final RMSNorm before lm_head, mtp.norm.weight.
    tensor_t final_norm_; // [hidden]

    // ----- Stage D.0: MTP K/V cache is now per-Request -----
    // Buffers live on InferenceRequest (req.mtp_k_cache / mtp_v_cache /
    // mtp_page_table_dev / mtp_past_seq_len). MTPModule itself only holds
    // weights + the cache size. This makes concurrent multi-request decode
    // safe — no shared mutable state to trample.
    static constexpr size_t kDefaultMaxKvLen_ = 4096;
    size_t max_kv_len_ = kDefaultMaxKvLen_;
};

// Helper: lazily allocate a request's MTP K/V buffers + reset past_seq_len.
// Idempotent — calling on an already-initialised request just zeros the
// past counter (buffer reuse). Public so serving_loop can call it before
// each turn / at request admission.
void mtp_reset_request_state(InferenceRequest& req, size_t max_kv_len, size_t num_kv_heads, size_t head_dim,
                             const ExecutorConfig& exec);

} // namespace zedinfer::model
