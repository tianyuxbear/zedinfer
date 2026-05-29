#pragma once

#include "backend/kvcache/block_pool.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer/generation_types.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace zedinfer {

/**
 * Result of a single generation call.
 * Returned by InferenceEngine::generate_tokens() and consumed by generate().
 */
struct GenerationResult {
    std::vector<int> output_ids;
    GenerationStats stats;
    // OpenAI-compatible finish_reason: "stop" (EOS / cancellation),
    // "length" (max_new_tokens reached), "tool_calls" (model emitted a tool call
    // sequence; set by the HTTP layer after parsing). Default "stop".
    std::string finish_reason = "stop";
};

/**
 * Phase of an inference request lifecycle.
 * Used by the scheduler to track request progress.
 */
enum class RequestPhase {
    QUEUED,  // waiting for scheduler admission
    PREFILL, // processing prompt tokens
    DECODE,  // generating tokens one-by-one
    COMPLETE // finished (EOS, max_tokens, or error)
};

/**
 * A single inference request.
 * Consumed by Scheduler to drive the generate loop.
 */
struct InferenceRequest {
    uint64_t request_id = 0;
    std::string session_id;
    std::vector<int> input_ids;
    GenerationConfig config;

    RequestPhase phase = RequestPhase::QUEUED;
    int generated_count = 0;
    int last_token = -1;

    std::vector<int> output_ids;

    // OpenAI-compatible finish_reason. Set by the scheduler when transitioning
    // a request to RequestPhase::COMPLETE; "stop" on EOS / cancellation,
    // "length" when generated_count reaches max_new_tokens. The HTTP layer may
    // overwrite this with "tool_calls" after parsing the model output for a
    // tool-call sequence.
    std::string finish_reason = "stop";

    // Latched flag set the first time the scheduler applies this request's
    // GenerationConfig.seed to the GeneralSampler. Stops the per-token
    // pick_sampler() path from re-seeding on every sample(), which would
    // collapse the RNG into a deterministic single-step sequence.
    bool sampler_seeded = false;

    // Reasoning-model thinking-block state (Qwen3.5 family). Tracks whether
    // the request is currently inside an open <think>...</think> block and
    // how many tokens have been generated since the last <think> was seen.
    // Scheduler updates these on every sampled token; engine resolves the
    // think_open_id / think_close_id token ids from the tokenizer at init
    // and passes them to Scheduler::process_results. Non-reasoning models
    // never enter a thinking state, so these fields stay at defaults.
    bool in_thinking = false;
    int  think_token_count = 0;

    // After scheduler force-emits </think>, the model's natural follow-up is
    // "\n\n" before the answer (see Qwen3.5's chat_template.jinja). When we
    // truncate thinking mid-token the model often continues the truncated
    // reasoning instead, dragging the drift into the answer. To anchor the
    // post-think state, the scheduler force-emits this many "\n\n" tokens
    // immediately after </think>, mirroring the trained pattern. Counter
    // decrements per step; reaches 0 → free-running again.
    int post_think_forced_newlines = 0;

    // Stream callback forwarded from GenerationConfig at request construction
    std::function<void(const std::string&)> stream_callback;

    // Timing
    std::chrono::steady_clock::time_point arrival_time;
    GenerationStats stats;

    // Continuous batching fields
    int prefill_progress = 0;                      // tokens already prefilled (chunked prefill)
    std::promise<GenerationResult> result_promise; // async result delivery
    std::shared_ptr<std::atomic<bool>> cancelled;  // set by HTTP handler on client disconnect

    // Block table: always accessed via block_table().
    // Session mode: points to session's table (borrowed, not freed by scheduler).
    // Batch mode: points to owned_block_table_ (freed by scheduler on completion).
    kvcache::SequenceBlockTable& block_table() { return *block_table_ptr_; }
    const kvcache::SequenceBlockTable& block_table() const { return *block_table_ptr_; }

    // Set borrowed block table (session mode — session owns the table)
    void borrow_block_table(kvcache::SequenceBlockTable& bt) {
        block_table_ptr_ = &bt;
        owns_block_table_ = false;
    }

    // Set owned block table (batch mode — scheduler allocated it)
    void own_block_table(kvcache::SequenceBlockTable bt) {
        owned_block_table_ = std::move(bt);
        block_table_ptr_ = &owned_block_table_;
        owns_block_table_ = true;
    }

    bool has_block_table() const { return block_table_ptr_ != nullptr; }
    bool owns_block_table() const { return owns_block_table_; }

    // Qwen3.5 hybrid-path state. Default-valued for non-hybrid models; the
    // Scheduler populates ssm_slot_idx_ at admit and clears it on finish.
    //
    // ssm_slot_idx_       : index into SSMStatePool; -1 = no slot held
    // input_embeds_       : pre-built layer-0 hidden state for multimodal
    //                       prefill. Holds the FULL input embedding sequence
    //                       [N_total, hidden] — text token embeddings with
    //                       vision-tower outputs already scattered into
    //                       <|image_pad|> positions. Despite the field name
    //                       containing only the image-scattered final tensor,
    //                       it is what the forward pass consumes as input.
    // pos_ids_thw_        : [3, N_total] int32 (t, h, w) positions per token
    //                       for 3D MRoPE; null for non-hybrid models
    // has_input_embeds_   : convenience flag mirroring input_embeds_ != nullptr
    int  ssm_slot_idx() const { return ssm_slot_idx_; }
    void set_ssm_slot_idx(int idx) { ssm_slot_idx_ = idx; }

    bool     has_input_embeds() const { return has_input_embeds_; }
    tensor_t input_embeds() const { return input_embeds_; }
    void     set_input_embeds(tensor_t e) {
        input_embeds_     = std::move(e);
        has_input_embeds_ = static_cast<bool>(input_embeds_);
    }

    tensor_t pos_ids_thw() const { return pos_ids_thw_; }
    void set_pos_ids_thw(tensor_t t) { pos_ids_thw_ = std::move(t); }

private:
    kvcache::SequenceBlockTable* block_table_ptr_ = nullptr;
    kvcache::SequenceBlockTable owned_block_table_;
    bool owns_block_table_ = false;

    int      ssm_slot_idx_ = -1;
    tensor_t input_embeds_;
    tensor_t pos_ids_thw_;
    bool     has_input_embeds_ = false;

public:
    // ----- Qwen3.5 MTP (Stage D) per-request K/V state -----
    // The MTP head has its own attention layer with its own K/V cache.
    // Storing the cache here (rather than as static state inside MTPModule)
    // lets concurrent requests run MTP in parallel without trampling each
    // other. Layout: [max_kv_len, num_kv_heads * head_dim] bf16 contiguous,
    // accessed via the "single huge page" trick (block_size=max_kv_len,
    // page_table=[0]) so the existing paged-attention kernel works as-is.
    // Buffers stay allocated for the lifetime of the request; only the
    // logical past_seq_len resets between conversation turns.
    tensor_t mtp_k_cache;
    tensor_t mtp_v_cache;
    tensor_t mtp_page_table_dev;       // [1] int32 = {0}
    int      mtp_past_seq_len = 0;     // committed K/V positions in cache

    // Stage D.1 speculative-decode pending draft. After each main forward
    // the engine asks MTP for the t+2 prediction; we stash it here so the
    // NEXT scheduled step can feed [last_token, mtp_pending_draft] to main
    // as a 2-token verify batch. -1 = no draft pending (regular 1-token
    // decode this step).
    int      mtp_pending_draft = -1;

    // Stage D.1 spec-decode counters (per request, lifetime of the request).
    // Useful for logging the accept rate observed during actual generation,
    // distinct from the off-line measurements done during Stage C tuning.
    int      mtp_accept_count = 0;
    int      mtp_reject_count = 0;

    // Number of tokens committed by the most recent scheduler step. Set by
    // Scheduler::process_results so serving_loop knows how many MTP forwards
    // to run to advance MTP's K/V cache. 1 for normal decode + reject, 2 on
    // spec accept. Reset to 0 after consumption.
    int      mtp_last_n_committed = 0;

    // Stage D.1 spec-decode recurrent-state handling (hybrid Qwen3.5 only).
    // The 2-token [last_token, draft] verify forward must not let the
    // speculative draft poison the linear-attention recurrent state (GDN
    // matrix + causal-conv window), which — unlike the paged KV cache — has no
    // positional addressing and cannot self-heal on reject. When this flag is
    // set, forward_linear_attn_layer commits token 0 (last_token) into the
    // request's real SSM slot and computes token 1 (draft) into the pool's
    // spec temp slot. On accept the scheduler promotes temp->real
    // (SSMStatePool::copy_slot_state); on reject the real slot already holds
    // the correct post-last_token state, so it just emits the corrected token —
    // no rollback, no redo. serving_loop sets the flag before a verify forward
    // and clears it after.
    bool     mtp_spec_verify_active = false;
};

} // namespace zedinfer
