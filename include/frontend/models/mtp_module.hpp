#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/expert_weights.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include "zedinfer/activation.hpp"   // ExecutorConfig (struct definition)

#include <memory>

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
    MTPModule(const Qwen3_5MoEConfig& main_cfg, ModelWeights& weights, const ExecutorConfig& exec);
    ~MTPModule();

    MTPModule(const MTPModule&) = delete;
    MTPModule& operator=(const MTPModule&) = delete;

    // Stage B.1: single-token, no past KV — predicts t+2 given main's
    // residual stream at position t and the token main just sampled at t+1.
    //
    // The attention block is simplified: with an empty MTP KV cache it
    // attends only to the current position, so softmax is degenerate
    // (uniform-1 over the single token). Multi-step KV cache and the
    // proper prefill path (filling MTP's K/V across the whole prompt)
    // land in Stage C with the speculative-decode scheduler.
    //
    // Args:
    //   hidden_at_t:       [1, hidden_size] — main's PRE-final-norm residual
    //   next_token_id:     id sampled by main at position t+1
    //   embed_tokens_w:    main embed_tokens.weight [vocab, hidden] — reused
    //   lm_head_w:         main lm_head.weight [vocab, hidden] — reused
    //   exec:              compute device + dtype
    //
    // Returns: logits [1, vocab] — model's guess for token t+2.
    // Throws if !ready().
    tensor_t forward(tensor_t hidden_at_t, int next_token_id,
                     tensor_t embed_tokens_w, tensor_t lm_head_w,
                     const ExecutorConfig& exec) const;

    // True iff a full MTP weight set was found at ctor time. False for
    // models that don't ship MTP (e.g. Qwen3 base, DeepSeek-R1 distill).
    // Callers (engine / scheduler) probe this before enabling spec decode.
    bool ready() const { return ready_; }

private:
    const Qwen3_5MoEConfig& main_cfg_;
    ExecutorConfig          exec_;
    bool                 ready_ = false;

    // Pre-fc projection norms (Qwen3_5MoeRMSNorm, (1+w) at kernel time).
    tensor_t pre_fc_norm_embedding_; // [hidden]
    tensor_t pre_fc_norm_hidden_;    // [hidden]

    // Fusion projection: 2*hidden -> hidden.
    tensor_t fc_weight_;             // [hidden, 2*hidden]

    // Single transformer block (mtp.layers.0.*) — slice references into the
    // main ModelWeights map. Naming mirrors the main hybrid layer so the
    // forward (when implemented in Stage B) can reuse the per-layer helpers.
    tensor_t in_layernorm_;          // [hidden]   mtp.layers.0.input_layernorm.weight
    tensor_t post_layernorm_;        // [hidden]   mtp.layers.0.post_attention_layernorm.weight
    tensor_t q_proj_;                // [2*Hq*Dh, hidden]  (q + output-gate doubled, like main full-attn)
    tensor_t k_proj_;                // [Hkv*Dh, hidden]
    tensor_t v_proj_;                // [Hkv*Dh, hidden]
    tensor_t o_proj_;                // [hidden, Hq*Dh]
    tensor_t q_norm_;                // [Dh]      (1+w) per-head RMSNorm
    tensor_t k_norm_;                // [Dh]
    tensor_t mlp_gate_router_;       // [num_experts, hidden]  MoE router
    tensor_t shared_expert_gate_;    // [1, hidden]            sigmoid gate for shared expert contribution
    tensor_t shared_expert_gate_proj_;
    tensor_t shared_expert_up_proj_;
    tensor_t shared_expert_down_proj_;

    // 256 experts for this single MTP layer, sliced per-expert from the fused
    // gate_up_proj / down_proj tensors. Owned (separate ExpertWeights from
    // the main model's). Stored in a one-layer ExpertWeights so we can reuse
    // moe_layer_forward in Stage B without touching its API.
    std::unique_ptr<ExpertWeights> experts_;

    // Final RMSNorm before lm_head, mtp.norm.weight.
    tensor_t final_norm_;            // [hidden]
};

} // namespace zedinfer::model
