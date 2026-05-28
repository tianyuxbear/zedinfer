#pragma once

#include "backend/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zedinfer::ops::moe {

struct TopKResult {
    std::vector<int32_t> expert_ids;   // [N * top_k] selected expert indices
    std::vector<float> expert_weights; // [N * top_k] normalized routing weights
};

// CPU top-k softmax for MoE router gating.
// logits: [N, num_experts] float values (host pointer).
// Returns top_k expert IDs and softmax weights per token.
// If norm_topk_prob is true, weights are renormalized to sum to 1.
TopKResult topk_softmax(const float* logits, size_t N, size_t num_experts, size_t top_k, bool norm_topk_prob);

// GPU fused softmax+top-K kernel. Replaces the CPU softmax + D2H + partial_sort
// pipeline with a single block-per-row reduction, eliminating one
// stream-draining memcpy_sync per MoE layer.
//
//   logits          : [N, num_experts]  bf16/fp16/fp32 device tensor
//   expert_ids      : [N, top_k]        int32 device tensor (output)
//   expert_weights  : [N, top_k]        fp32 device tensor (output)
//
// `norm_topk_prob=true` makes the per-row top-k weights sum to 1 (Qwen3.5
// router behavior). `false` returns the un-renormalized global softmax
// probabilities; their sum is < 1.
//
// The kernel assumes num_experts ≤ 1024 and top_k ≤ 32 (sized in registers /
// shared memory). For Qwen3.5 (num_experts=256, top_k=8) this is far inside.
void topk_softmax_gpu(tensor_t expert_ids, tensor_t expert_weights, tensor_t logits, size_t top_k,
                      bool norm_topk_prob);

} // namespace zedinfer::ops::moe
