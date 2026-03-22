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

} // namespace zedinfer::ops::cpu
