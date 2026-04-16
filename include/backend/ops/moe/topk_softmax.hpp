#pragma once

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

} // namespace zedinfer::ops::moe
