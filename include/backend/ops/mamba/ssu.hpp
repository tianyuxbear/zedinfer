#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

namespace zedinfer::ops::mamba {

// Parameters for the FlashInfer Mamba2 selective_state_update (SSU) kernel
// in MTP path. The same struct serves decode (num_tokens=1) and prefill
// (num_tokens>1, varlen via cu_seqlens=[0,N]) — the wrapper dispatches
// internally on num_tokens.
//
// All input/output tensors are bf16 on the compute device; A_log is f32.
// The state buffer is owned by SSMStatePool; we only carry a view + slot
// addressing here, never a separate device alloc.
struct SSUParams {
    // State pool view + addressing.
    model::SSMStateView state_view;
    int slot_idx = -1;
    int layer_idx = -1;

    // Inputs. Shape leading-dim is N (num_tokens this step):
    //   q, k:  [N, num_k_heads * key_head_dim]
    //   v:     [N, num_v_heads * value_head_dim]
    //   a, b:  [N, num_v_heads]
    //   z:     [N, num_v_heads * value_head_dim] (caller passes pre-silu'd)
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t a;
    tensor_t b;

    // Per-V-head persistent weights (loaded from in_proj_*.weight tensors):
    //   A_log:   [num_v_heads], float32
    //   dt_bias: [num_v_heads], bf16
    tensor_t A_log;
    tensor_t dt_bias;

    // Gate path (pre-silu'd by caller).
    tensor_t z;

    // Pre-allocated output tensor (shape [N, num_v_heads * value_head_dim]).
    tensor_t out;

    // Number of tokens processed this call.
    //   1   → decode STP path (single-token state update in place).
    //   >1  → prefill varlen path (FlashInfer cu_seqlens=[0,N]).
    int num_tokens = 0;
};

// Dispatch one SSU step against the pool slot.
//
// Side effects:
//   - state_view's ssm buffer at (slot_idx, layer_idx) is updated in place.
//   - out is written with the gated SSM readout for each input token.
//
// Caller contract:
//   - reset_slot(slot_idx) must have been called once at request start so
//     the SSM state is zeroed before the first call for this request.
//   - The compute stream is the runtime's compute stream; the wrapper runs
//     on it (no separate sync needed by callers in the layer loop).
void ssu(const SSUParams& params);

// Extract a contiguous [N, slice_width] tile from a strided [N, src_width]
// source buffer (e.g., split qkv into q / k / v after the depthwise causal
// conv1d). Implemented on the runtime's compute stream via cudaMemcpy2DAsync.
// `elem_bytes` is the per-element size of `dst`/`src` (e.g., 2 for bf16).
void copy_strided_rows(tensor_t dst, tensor_t src,
                        size_t src_offset_elems, size_t slice_width,
                        size_t src_width, size_t rows, size_t elem_bytes);

} // namespace zedinfer::ops::mamba
