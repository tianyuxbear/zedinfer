#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// Bidirectional self-attention for the ViT path (Qwen3.5-VL vision tower).
//
//   q, k, v: [N, H, D]  (BF16)
//   out    : [N, H, D]  (BF16)
//   scale  : usually 1/sqrt(D); applied to the QK^T product.
//
// No causal mask — vision attention is fully bidirectional. No KV cache —
// each forward processes a single sequence of patch tokens from one or more
// images concatenated together. Head_dim is arbitrary (Qwen3.5-VL uses 72,
// which is outside FlashInfer's templated head_dim set), so we provide a
// naive but correct CUDA implementation rather than wrapping FlashInfer.
//
// All tensors must live on the same device and have the same dtype.
struct VisionAttentionParams {
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t out;
    float scale;
};

void vision_attention(const VisionAttentionParams& p);

} // namespace zedinfer::ops
