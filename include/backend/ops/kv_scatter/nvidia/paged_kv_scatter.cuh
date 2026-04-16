#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

void scatter_paged_kv(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base, const int* page_table,
                      int block_size, int past_len, int num_tokens, size_t token_bytes);

void append_paged_kv(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base,
                     const int* kv_page_indices, const int* kv_indptr, const int* kv_last_page_len,
                     const int* batch_indices, const int* positions, int batch_size, int nnz_tokens, int num_kv_heads,
                     int head_dim, int block_size, zedinferDataType_t dtype);

void append_paged_kv_decode(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base,
                            const int* kv_page_indices, const int* kv_indptr, const int* kv_last_page_len,
                            int batch_size, int num_kv_heads, int head_dim, int block_size, zedinferDataType_t dtype);

} // namespace zedinfer::ops::nvidia
