#include "backend/ops/moe/topk_softmax.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace zedinfer::ops::moe {

// Matches HuggingFace Qwen3MoE routing:
//   probs = softmax(logits, dim=-1) over ALL experts
//   topk_probs, topk_ids = topk(probs, k)
//   if norm_topk_prob: topk_probs /= topk_probs.sum()
//
// Note: top-k selection by probability is equivalent to top-k by raw logit (softmax is
// monotonic), so the selected expert set matches either way. But when norm_topk_prob=false,
// the returned weights must be the GLOBAL-softmax probabilities (sum < 1), not a local
// softmax over just the top-k (which would wrongly sum to 1).
TopKResult topk_softmax(const float* logits, size_t N, size_t num_experts, size_t top_k, bool norm_topk_prob) {
    TopKResult result;
    result.expert_ids.resize(N * top_k);
    result.expert_weights.resize(N * top_k);

    std::vector<int32_t> indices(num_experts);
    std::vector<float> probs(num_experts);

    for (size_t n = 0; n < N; ++n) {
        const float* row = logits + n * num_experts;

        // Global softmax over all experts.
        float max_logit = *std::max_element(row, row + num_experts);
        float sum_exp = 0.0f;
        for (size_t e = 0; e < num_experts; ++e) {
            probs[e] = std::exp(row[e] - max_logit);
            sum_exp += probs[e];
        }
        const float inv_sum = 1.0f / sum_exp;
        for (size_t e = 0; e < num_experts; ++e) {
            probs[e] *= inv_sum;
        }

        // Pick top-k by probability.
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(top_k), indices.end(),
                          [&probs](int32_t a, int32_t b) { return probs[a] > probs[b]; });

        // Collect top-k probabilities and compute their sum for optional renormalization.
        float topk_sum = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            float p = probs[indices[k]];
            result.expert_weights[n * top_k + k] = p;
            topk_sum += p;
        }

        if (norm_topk_prob && topk_sum > 0.0f) {
            const float inv_topk = 1.0f / topk_sum;
            for (size_t k = 0; k < top_k; ++k) {
                result.expert_weights[n * top_k + k] *= inv_topk;
            }
        }

        for (size_t k = 0; k < top_k; ++k) {
            result.expert_ids[n * top_k + k] = indices[k];
        }
    }

    return result;
}

} // namespace zedinfer::ops::moe
