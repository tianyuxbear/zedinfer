#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

namespace zedinfer::ops::mamba {

// Causal depthwise 1D convolution with persistent state, used by Qwen3.5's
// Mamba2 linear-attention block before the SSU kernel.
//
// Per-token contract (for each n in [0, N), per channel d):
//   window[0..K-1] = (state | x)[n-K+1 .. n]
//   y[n, d]        = sum_{i=0..K-1} window[i, d] * w[d, 0, i]
//   y[n, d]        = silu(y[n, d])              # y *= sigmoid(y)
// Side effect:
//   state := last (K-1) tokens of (state | x), i.e. the trailing window
//   becomes the new state for subsequent decode/prefill calls.
//
// Tensor layouts:
//   x:           [N, D]                bf16, contiguous (D == qkv_dim).
//   weight:      [D, 1, K]             bf16, depthwise.
//   out:         [N, D]                bf16, caller pre-allocated.
//   state:       [K-1, D]              bf16, owned by SSMStatePool at
//                                       conv_base + slot*conv_stride_slot
//                                       + layer*conv_stride_layer.
// Only K = 4 is currently implemented (Qwen3.5).
void causal_conv1d(tensor_t out, tensor_t x, tensor_t weight,
                   model::SSMStateView state_view, int slot_idx, int layer_idx);

} // namespace zedinfer::ops::mamba
