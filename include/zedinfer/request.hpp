#pragma once

#include "zedinfer/generation_types.hpp"
#include "backend/kvcache/block_pool.hpp"

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
    QUEUED,   // waiting for scheduler admission
    PREFILL,  // processing prompt tokens
    DECODE,   // generating tokens one-by-one
    COMPLETE  // finished (EOS, max_tokens, or error)
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

    // Stream callback forwarded from GenerationConfig at request construction
    std::function<void(const std::string &)> stream_callback;

    // Timing
    std::chrono::steady_clock::time_point arrival_time;
    GenerationStats stats;

    // Continuous batching fields
    int prefill_progress = 0;                        // tokens already prefilled (chunked prefill)
    std::promise<GenerationResult> result_promise;    // async result delivery
    std::shared_ptr<std::atomic<bool>> cancelled;     // set by HTTP handler on client disconnect

    // Block table: always accessed via block_table().
    // Session mode: points to session's table (borrowed, not freed by scheduler).
    // Batch mode: points to owned_block_table_ (freed by scheduler on completion).
    kvcache::SequenceBlockTable &block_table() { return *block_table_ptr_; }
    const kvcache::SequenceBlockTable &block_table() const { return *block_table_ptr_; }

    // Set borrowed block table (session mode — session owns the table)
    void borrow_block_table(kvcache::SequenceBlockTable &bt) {
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

private:
    kvcache::SequenceBlockTable *block_table_ptr_ = nullptr;
    kvcache::SequenceBlockTable owned_block_table_;
    bool owns_block_table_ = false;
};

} // namespace zedinfer
