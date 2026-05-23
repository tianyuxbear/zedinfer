#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops::mamba {

// In-place row-wise L2 normalization with optional per-row scale.
//
// Input layout: tensor shape [N, num_heads * head_dim] bf16. Each (n, head)
// row of length `head_dim` is independently normalized:
//   row_normed = scale * row / sqrt(sum(row^2) + eps)
//
// Use case: Qwen3.5 GatedDeltaNet feeds query/key into the recurrence with
// HF's `use_qk_l2norm_in_kernel=True`, which applies l2norm(query, eps=1e-6)
// and l2norm(key, eps=1e-6); the recurrence then scales the query by
// 1/sqrt(head_k_dim). Call this op once per layer for q (scale=1/sqrt(Dk))
// and once for k (scale=1) before invoking ops::mamba::gdn.
void qk_l2norm_inplace(tensor_t qk, int num_heads, int head_dim,
                       float scale = 1.0f, float eps = 1e-6f);

} // namespace zedinfer::ops::mamba
