#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {

// Paged attention decode for CPU: single query token, K/V via block table.
void paged_attention_decode(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seq_len,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size);

// Paged attention prefill for CPU: multiple query tokens with causal mask.
void paged_attention_prefill(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seqlen_q, int past_len,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size);

// Paged attention decode batched for CPU: loop over requests.
void paged_attention_decode_batched(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const void *k_block_tables, const void *v_block_tables,
    const void *seq_lens,
    int num_requests, int max_blocks_per_seq,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size);

} // namespace zedinfer::ops::cpu
