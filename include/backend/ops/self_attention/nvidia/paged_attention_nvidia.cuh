#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

// Paged attention decode: single query token, K/V read via block table from pool.
// Grid: (nhead,), one block per attention head.
//
// Parameters:
//   attn_val:    output [nhead, head_dim]
//   q:           query  [nhead, head_dim]  (single token, seqlen=1)
//   k_pool:      K cache pool [num_blocks * block_size, nkvhead, head_dim]
//   v_pool:      V cache pool [num_blocks * block_size, nkvhead, head_dim]
//   block_table: [max_blocks_per_seq] int32 — physical block IDs for this sequence
//   seq_len:     actual token count in KV cache
//   scale:       1/sqrt(head_dim)
//   block_size:  tokens per block
void paged_attention_decode(std::byte* attn_val, const std::byte* q, const std::byte* pool_base,
                            const int* k_block_table, const int* v_block_table, int seq_len, float scale,
                            zedinferDataType_t type, int nhead, int nkvhead, int head_dim, int block_size);

// Paged attention prefill: multiple query tokens, K/V read via block table.
// Grid: (seqlen_q, nhead), one block per (query_position, head).
void paged_attention_prefill(std::byte* attn_val, const std::byte* q, const std::byte* pool_base,
                             const int* k_block_table, const int* v_block_table, int seqlen_q, int past_len,
                             float scale, zedinferDataType_t type, int nhead, int nkvhead, int head_dim,
                             int block_size);

// Batched paged attention decode: multiple requests, each with 1 query token.
// Grid: (num_requests, nhead).
void paged_attention_decode_batched(std::byte* attn_val, const std::byte* q, const std::byte* pool_base,
                                    const int* k_block_tables, // [num_reqs * max_blocks_per_seq] flattened
                                    const int* v_block_tables, // [num_reqs * max_blocks_per_seq] flattened
                                    const int* seq_lens,       // [num_reqs]
                                    int num_reqs, int max_blocks_per_seq, float scale, zedinferDataType_t type,
                                    int nhead, int nkvhead, int head_dim, int block_size);

} // namespace zedinfer::ops::nvidia
