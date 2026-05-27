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
    // ssm_slot_idx_   : index into SSMStatePool; -1 = no slot held
    // image_embeds_   : pre-computed vision-tower output, scattered into the
    //                   input embedding sequence at <|image_pad|> positions
    // pos_ids_thw_    : [3, N_total] int32 (t, h, w) positions per token for
    //                   3D MRoPE; null for non-hybrid models
    // has_images_    : convenience flag mirroring image_embeds_ != nullptr
    int  ssm_slot_idx() const { return ssm_slot_idx_; }
    void set_ssm_slot_idx(int idx) { ssm_slot_idx_ = idx; }

    bool has_images() const { return has_images_; }
    tensor_t image_embeds() const { return image_embeds_; }
    void set_image_embeds(tensor_t e) {
        image_embeds_ = std::move(e);
        has_images_ = static_cast<bool>(image_embeds_);
    }

    tensor_t pos_ids_thw() const { return pos_ids_thw_; }
    void set_pos_ids_thw(tensor_t t) { pos_ids_thw_ = std::move(t); }

private:
    kvcache::SequenceBlockTable* block_table_ptr_ = nullptr;
    kvcache::SequenceBlockTable owned_block_table_;
    bool owns_block_table_ = false;

    int      ssm_slot_idx_ = -1;
    tensor_t image_embeds_;
    tensor_t pos_ids_thw_;
    bool     has_images_ = false;
};

} // namespace zedinfer
