#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

void add(tensor_t c, tensor_t a, tensor_t b);
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
void embedding(tensor_t out, tensor_t index, tensor_t weight);
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr);
void rearrange(tensor_t out, tensor_t in);
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void swiglu(tensor_t out, tensor_t gate, tensor_t up);

// Paged attention decode: Q is single token [nhead, head_dim].
// K/V are read directly from the block pool via block_table.
// Eliminates the contiguous gather copy for decode steps.
void paged_attention_decode(
    tensor_t attn_val,          // [nhead, head_dim]
    tensor_t q,                 // [nhead, head_dim]
    const void *pool_base,      // raw pointer to block pool memory (shared by K and V)
    const int *k_block_table,   // physical block IDs for K
    const int *v_block_table,   // physical block IDs for V
    int seq_len,                // actual token count in KV cache
    float scale,
    zedinferDataType_t dtype,
    zedinferDeviceType_t device_type,
    int device_id,
    int nhead, int nkvhead, int head_dim, int block_size);

// Paged attention prefill: multiple query tokens, K/V read via block table.
// Before calling, scatter current layer's write buffer to blocks.
void paged_attention_prefill(
    tensor_t attn_val,          // [seqlen_q, nhead, head_dim]
    tensor_t q,                 // [seqlen_q, nhead, head_dim]
    const void *pool_base,
    const int *k_block_table,
    const int *v_block_table,
    int seqlen_q,
    int past_len,
    float scale,
    zedinferDataType_t dtype,
    zedinferDeviceType_t device_type,
    int device_id,
    int nhead, int nkvhead, int head_dim, int block_size);

// Batched paged attention decode: N requests, each with 1 query token.
// Block tables are flattened: [num_reqs * max_blocks_per_seq].
void paged_attention_decode_batched(
    tensor_t attn_val,          // [num_reqs, nhead, head_dim]
    tensor_t q,                 // [num_reqs, nhead, head_dim]
    const void *pool_base,
    const int *k_block_tables,  // [num_reqs * max_blocks_per_seq]
    const int *v_block_tables,
    const int *seq_lens,        // [num_reqs]
    int num_reqs,
    int max_blocks_per_seq,
    float scale,
    zedinferDataType_t dtype,
    zedinferDeviceType_t device_type,
    int device_id,
    int nhead, int nkvhead, int head_dim, int block_size);

} // namespace zedinfer::ops