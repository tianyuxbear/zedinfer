#pragma once

#include <cstddef>

namespace zedinfer::ops::nvidia {

void scatter_paged_kv(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base, const int* page_table,
                      int block_size, int past_len, int num_tokens, size_t token_bytes);

} // namespace zedinfer::ops::nvidia
