#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

namespace zedinfer::ops::mamba {

// Parameters for the Qwen3.5 Gated Delta Rule kernel.
//
// Per-token, per-V-head computation:
//   beta  = sigmoid(b)
//   decay = exp( -exp(A_log) * softplus(a + dt_bias) )
//   delta = v - S @ k
//   S     = decay * S + beta * outer(delta, k)
//   y     = S @ q
//
// Shapes (N = num_tokens this call):
//   q, k:    [N, num_k_heads * key_head_dim]  bf16 (post-conv, post-norm)
//   v:       [N, num_v_heads * value_head_dim] bf16 (post-conv)
//   b, a:    [N, num_v_heads]                 bf16 (raw, sigmoid/softplus applied in-kernel)
//   A_log:   [num_v_heads]                    f32  persistent weight
//   dt_bias: [num_v_heads]                    bf16 persistent weight
//   out:     [N, num_v_heads * value_head_dim] bf16 (pre-norm; caller applies rms_norm + silu(z) outside)
//
// State (in SSMStatePool):
//   slot[layer_idx]: [num_v_heads, value_head_dim, key_head_dim] f32
//   Reused from the existing pool — Qwen3.5 d_state == key_head_dim, so
//   the storage size is unchanged from M1. The interpretation flips from
//   "diagonal SSM state" to "delta-rule matrix state".
struct GDNParams {
    // State pool view + addressing
    model::SSMStateView state_view;
    int slot_idx = -1;
    int layer_idx = -1;

    // Inputs
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t b; // beta input (sigmoid applied in-kernel)
    tensor_t a; // gate input (softplus + decay applied in-kernel)

    // Persistent weights
    tensor_t A_log;
    tensor_t dt_bias;

    // Pre-allocated output [N, num_v_heads * value_head_dim] bf16.
    tensor_t out;

    // Number of tokens this call:
    //   1   → decode kernel (single-step, in-place state update).
    //   >1  → prefill kernel (sequential token loop; caller passes cu_seqlens=[0,N] semantically).
    int num_tokens = 0;
};

// Dispatch one GDN step against the pool slot.
//
// Side effects:
//   - state_view's ssm buffer at (slot_idx, layer_idx) is updated in place.
//   - out is written with the per-V-head readout for each input token.
//
// Caller contract:
//   - reset_slot(slot_idx) must have been called once at request start so
//     the state is zeroed before the first call for this request.
//   - Compute stream is the runtime's compute stream.
void gdn(const GDNParams& params);

} // namespace zedinfer::ops::mamba
