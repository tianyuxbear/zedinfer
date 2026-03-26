#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

namespace zedinfer::ops {

/**
 * Session-constant attention configuration.
 * Created once per ForwardContext, reused across all layers.
 */
struct AttentionConfig {
    int nhead;
    int nkvhead;
    int head_dim;
    float scale;
    int block_size;
    zedinferDataType_t dtype;
    zedinferDeviceType_t device_type;
    int device_id;
};

/**
 * Per-call attention parameters.
 * Mode is determined by which fields are set:
 *   - batched_k_block_tables != nullptr  → batched paged decode
 *   - k_block_table != nullptr && seqlen_q == 1  → single paged decode
 *   - k_block_table != nullptr && seqlen_q > 1   → single paged prefill
 */
struct AttentionParams {
    const AttentionConfig& config;

    // Output and query
    tensor_t out = nullptr;
    tensor_t q = nullptr;

    // === Paged mode ===
    const void* pool_base = nullptr;

    // Single request
    const int* k_block_table = nullptr;
    const int* v_block_table = nullptr;
    int seq_len = 0;  // KV cache length (for decode)
    int seqlen_q = 0; // query length (1=decode, >1=prefill)
    int past_len = 0; // for prefill causal mask

    // Batched decode
    const int* batched_k_block_tables = nullptr; // [num_reqs * max_blocks_per_seq]
    const int* batched_v_block_tables = nullptr;
    const int* batched_seq_lens = nullptr;       // [num_reqs]
    int num_requests = 0;
    int max_blocks_per_seq = 0;

    // Helpers for dispatch
    bool is_batched() const { return batched_k_block_tables != nullptr; }
    bool is_decode() const { return seq_len > 0 && !is_batched(); }
};

} // namespace zedinfer::ops
