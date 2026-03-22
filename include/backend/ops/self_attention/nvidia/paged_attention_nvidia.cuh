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
void paged_attention_decode(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seq_len,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size);

} // namespace zedinfer::ops::nvidia
