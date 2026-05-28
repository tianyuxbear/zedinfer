#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

// Paged attention decode: single query token, K/V read via block table from pool.
// Grid: (nhead,), one block per attention head.
//
void paged_attention_decode(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                            const std::byte* v_pool_base, const int* page_table, int seq_len, float scale,
                            zedinferDataType_t type, int nhead, int nkvhead, int head_dim, int block_size);

// Paged attention prefill: multiple query tokens, K/V read via block table.
// Grid: (seqlen_q, nhead), one block per (query_position, head).
void paged_attention_prefill(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                             const std::byte* v_pool_base, const int* page_table, int seqlen_q, int past_len,
                             float scale, zedinferDataType_t type, int nhead, int nkvhead, int head_dim,
                             int block_size);

// Paged attention for small n_q (Qwen3.5 MTP spec-decode verify; n_q ∈ {2, 3, 4}).
// Same grid shape as paged_attention_decode (nhead,) — one block per head — but
// each block iterates over the KV cache *once* and scores all n_q queries against
// each KV chunk. This eliminates the 1/n_q reduction in K/V reuse that makes
// paged_attention_prefill slow at small n_q (~4.4x per-layer overhead at n_q=2).
// Returns false if n_q is outside the supported range; caller should fall back
// to paged_attention_prefill in that case.
bool paged_attention_small_nq(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                              const std::byte* v_pool_base, const int* page_table, int seqlen_q, int past_len,
                              float scale, zedinferDataType_t type, int nhead, int nkvhead, int head_dim,
                              int block_size);

// Batched paged attention decode: multiple requests, each with 1 query token.
// Grid: (num_requests, nhead).
void paged_attention_decode_batched(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                                    const std::byte* v_pool_base,
                                    const int* page_tables, // [num_reqs * max_blocks_per_seq] flattened
                                    const int* seq_lens,    // [num_reqs]
                                    int num_reqs, int max_blocks_per_seq, float scale, zedinferDataType_t type,
                                    int nhead, int nkvhead, int head_dim, int block_size);

} // namespace zedinfer::ops::nvidia
