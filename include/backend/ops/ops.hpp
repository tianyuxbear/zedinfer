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

} // namespace zedinfer::ops