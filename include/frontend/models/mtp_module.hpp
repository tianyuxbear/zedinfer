#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/expert_weights.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include "zedinfer/activation.hpp"   // ExecutorConfig (struct definition)

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
    MTPModule(const Qwen3_5MoEConfig& main_cfg, ModelWeights& weights, const ExecutorConfig& exec);
    ~MTPModule();

    MTPModule(const MTPModule&) = delete;
    MTPModule& operator=(const MTPModule&) = delete;

    // Stage C.1: single-token decode forward WITH an internal K/V cache.
    //
    // Each call advances MTP's K/V cache by 1 position. The first call
    // after reset_kv_cache() sees past=0 (empty cache, attention
    // degenerate to self-attention), and each subsequent call sees
    // past++ positions of context. Quality climbs as the cache fills.
    //
    // Stage C.2 will add a multi-token batched variant that lets us
    // ALSO fill the cache across the prompt during main prefill,
    // closing the "first decode sees empty cache" gap.
    //
    // Args:
    //   hidden_at_t:       [1, hidden_size] — main's PRE-final-norm residual
    //   next_token_id:     id sampled by main at position t+1
    //   embed_tokens_w:    main embed_tokens.weight [vocab, hidden] — reused
    //   lm_head_w:         main lm_head.weight [vocab, hidden] — reused
    //   exec:              compute device + dtype
    //
    // Returns: logits [1, vocab] — model's guess for token t+2.
    // Throws if !ready() OR if the cache is full (past >= max_kv_len_).
    tensor_t forward(tensor_t hidden_at_t, int next_token_id,
                     tensor_t embed_tokens_w, tensor_t lm_head_w,
                     const ExecutorConfig& exec) const;

    // Stage C.2: prefill MTP K/V across the prompt.
    //
    // After main prefill of a P-token prompt, the caller gives us:
    //   hidden_main_seq: [P, hidden] — main's residuals at positions 0..P-1
    //   next_tokens:     size P, where next_tokens[i] is the token at
    //                    position i+1 (i.e. prompt[i+1] for i in 0..P-2,
    //                    and main's just-sampled t_P for i=P-1).
    //
    // We loop the per-position forward P times so each call advances the
    // internal K/V cache by 1. Sequential — Stage D will batch this into
    // a single attention call with seqlen_q=P (~10x faster on the prompt).
    //
    // Returns the LAST-position logits [1, vocab], i.e. MTP's prediction
    // for token t_{P+1}. Intermediate logits are computed and discarded.
    tensor_t prefill(tensor_t hidden_main_seq,
                     const std::vector<int>& next_tokens,
                     tensor_t embed_tokens_w, tensor_t lm_head_w,
                     const ExecutorConfig& exec) const;

    // Reset the internal K/V cache before processing a new request.
    // (Stage C.1 single-tenant design — Stage D will move cache into
    // per-Request state to support multi-request concurrent decode.)
    void reset_kv_cache() const { past_seq_len_ = 0; }

    // Current cached sequence length (debug / spec-decode accept-rate logging).
    int past_seq_len() const { return past_seq_len_; }

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

    // ----- Stage C.1: internal MTP K/V cache (single-tenant) -----
    // Logical layout: [max_kv_len_, num_kv_heads, head_dim] bf16, contiguous.
    // Treated as a single paged block (block_size = max_kv_len_,
    // page_table = [0]) so the existing ops::attention kernel works as-is.
    static constexpr size_t kDefaultMaxKvLen_ = 4096;
    size_t           max_kv_len_      = kDefaultMaxKvLen_;
    mutable tensor_t k_cache_;        // allocated lazily on first forward
    mutable tensor_t v_cache_;
    mutable tensor_t page_table_dev_; // [1] int32 = {0}, allocated once
    mutable int      past_seq_len_ = 0;
};

} // namespace zedinfer::model
