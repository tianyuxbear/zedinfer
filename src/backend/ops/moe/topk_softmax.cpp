#include "backend/ops/moe/topk_softmax.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace zedinfer::ops::moe {

TopKResult topk_softmax(const float* logits, size_t N, size_t num_experts, size_t top_k, bool norm_topk_prob) {
    TopKResult result;
    result.expert_ids.resize(N * top_k);
    result.expert_weights.resize(N * top_k);

    std::vector<int32_t> indices(num_experts);

    for (size_t n = 0; n < N; ++n) {
        const float* row = logits + n * num_experts;

        // Find top-k expert indices by partial sort
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(top_k), indices.end(),
                          [row](int32_t a, int32_t b) { return row[a] > row[b]; });

        // Compute softmax over top-k logits
        float max_logit = row[indices[0]];
        float sum_exp = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            float exp_val = std::exp(row[indices[k]] - max_logit);
            result.expert_weights[n * top_k + k] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        if (norm_topk_prob && sum_exp > 0.0f) {
            for (size_t k = 0; k < top_k; ++k) {
                result.expert_weights[n * top_k + k] /= sum_exp;
            }
        }

        // Store expert IDs
        for (size_t k = 0; k < top_k; ++k) {
            result.expert_ids[n * top_k + k] = indices[k];
        }
    }

    return result;
}

} // namespace zedinfer::ops::moe
