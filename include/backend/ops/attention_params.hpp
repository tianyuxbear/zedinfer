#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

namespace zedinfer::ops {

struct FlashInferDecodePlan {
    tensor_t float_workspace = nullptr; // byte-addressed scratch backing tmp_v/tmp_s
    tensor_t int_workspace = nullptr;   // byte-addressed scratch backing planner metadata
    int padded_batch_size = 0;
    int64_t v_offset = 0;
    int64_t s_offset = 0;
    int64_t request_indices_offset = 0;
    int64_t kv_tile_indices_offset = 0;
    int64_t o_indptr_offset = 0;
    int64_t block_valid_mask_offset = 0;
    int64_t kv_chunk_size_ptr_offset = 0;
    bool split_kv = false;
    bool valid = false;

    void reset() {
        float_workspace.reset();
        int_workspace.reset();
        padded_batch_size = 0;
        v_offset = 0;
        s_offset = 0;
        request_indices_offset = 0;
        kv_tile_indices_offset = 0;
        o_indptr_offset = 0;
        block_valid_mask_offset = 0;
        kv_chunk_size_ptr_offset = 0;
        split_kv = false;
        valid = false;
    }
};

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
 *   - batched_page_tables != nullptr  → batched paged decode
 *   - page_table != nullptr && seqlen_q == 1  → single paged decode
 *   - page_table != nullptr && seqlen_q > 1   → single paged prefill
 */
struct AttentionParams {
    const AttentionConfig& config;

    // Output and query
    tensor_t out = nullptr;
    tensor_t q = nullptr;

    // Optional NVIDIA FlashInfer path (GPU only).
    bool use_flashinfer = false;

    // === Paged mode ===
    const void* k_pool_base = nullptr;
    const void* v_pool_base = nullptr;

    // Single request
    const int* page_table = nullptr;
    int seq_len = 0;  // KV cache length (for decode)
    int seqlen_q = 0; // query length (1=decode, >1=prefill)
    int past_len = 0; // for prefill causal mask

    // Batched decode
    const int* batched_page_tables = nullptr; // [num_reqs * max_blocks_per_seq]
    const int* batched_seq_lens = nullptr;    // [num_reqs]
    int num_requests = 0;
    int max_blocks_per_seq = 0;

    // FlashInfer paged KV CSR
    const int* kv_indptr = nullptr;                       // device [kv_batch_size + 1]
    const int* kv_page_indices = nullptr;                 // device [nnz_pages]
    const int* kv_last_page_len = nullptr;                // device [kv_batch_size]
    const int* qo_indptr = nullptr;                       // device [kv_batch_size + 1], prefill only
    const int* fi_request_indices = nullptr;              // device [padded_batch_size], decode fast path only
    const int* fi_kv_tile_indices = nullptr;              // device [padded_batch_size], decode fast path only
    const int* fi_o_indptr = nullptr;                     // device [padded_batch_size + 1], decode fast path only
    const int* fi_kv_chunk_size_ptr = nullptr;            // device [1], decode fast path only
    const FlashInferDecodePlan* fi_decode_plan = nullptr; // cached planner metadata/workspaces, decode only

    // Host-side planners consume host CSR metadata directly.
    const int* kv_indptr_host = nullptr; // host [kv_batch_size + 1]
    const int* qo_indptr_host = nullptr; // host [kv_batch_size + 1], prefill only
    int kv_batch_size = 0;

    // Helpers for dispatch
    bool is_batched() const { return batched_page_tables != nullptr; }
    bool is_decode() const { return seq_len > 0 && !is_batched(); }
};

} // namespace zedinfer::ops
