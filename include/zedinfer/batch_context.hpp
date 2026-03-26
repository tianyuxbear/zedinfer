#pragma once

#include "backend/kvcache/block_pool.hpp"
#include "zedinfer/request.hpp"

#include <cstdint>
#include <vector>

namespace zedinfer {

/**
 * Context for a batched model forward pass.
 * Contains flattened tokens from all requests (no padding)
 * and per-slot metadata for attention routing.
 */
struct BatchContext {
    // Flattened token IDs from all requests
    std::vector<int> token_ids;

    // Global position IDs for each token (for RoPE)
    std::vector<int64_t> position_ids;

    // Per-slot metadata
    struct Slot {
        InferenceRequest* request;
        int token_offset; // start position in token_ids
        int num_tokens;   // tokens in this slot (1 for decode, N for prefill chunk)
        int past_len;     // tokens already in KV cache before this forward
        bool is_prefill;
    };
    std::vector<Slot> slots;

    // Paged attention metadata (indexed by slot index within decode/prefill groups)
    // For decode slots: parallel arrays
    std::vector<const int*> decode_k_block_tables;
    std::vector<const int*> decode_v_block_tables;
    std::vector<int> decode_seq_lens; // KV cache length per decode request (past_len + 1)
    int decode_token_offset = 0;      // start of decode tokens in token_ids

    // For prefill slots: processed sequentially
    std::vector<const int*> prefill_k_block_tables;
    std::vector<const int*> prefill_v_block_tables;
    std::vector<int> prefill_past_lens;   // past_len per prefill request
    std::vector<int> prefill_chunk_sizes; // tokens per prefill request in this iteration

    int total_tokens() const { return static_cast<int>(token_ids.size()); }
    int num_decode_slots() const { return static_cast<int>(decode_seq_lens.size()); }
    int num_prefill_slots() const { return static_cast<int>(prefill_chunk_sizes.size()); }
};

/**
 * Output of scheduler: which requests to process this iteration.
 */
struct ScheduledBatch {
    std::vector<InferenceRequest*> decode_requests;
    std::vector<InferenceRequest*> prefill_requests;
    std::vector<int> prefill_chunk_starts; // start offset in input_ids per prefill request
    std::vector<int> prefill_chunk_sizes;  // tokens to process per prefill request

    int total_tokens() const;
    bool empty() const;

    // Assemble into BatchContext for model forward
    BatchContext build_context() const;
};

} // namespace zedinfer
