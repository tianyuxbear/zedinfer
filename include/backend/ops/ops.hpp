#pragma once

#include "backend/ops/attention_params.hpp"
#include "backend/ops/silu_mul/silu_mul.hpp"
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

void add(tensor_t c, tensor_t a, tensor_t b);
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
void embedding(tensor_t out, tensor_t index, tensor_t weight);
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr);
void linear_quantized(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, tensor_t scale, tensor_t g_idx,
                      int num_bits, int group_size);
void rearrange(tensor_t out, tensor_t in);
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
void rope_qk(tensor_t q_out, tensor_t k_out, tensor_t q_in, tensor_t k_in, tensor_t pos_ids, float theta);
void swiglu(tensor_t out, tensor_t gate, tensor_t up);

// Unified attention dispatch.
// Mode determined by AttentionParams fields — see attention_params.hpp.
void attention(const AttentionParams& params);

// out[i] += alpha * a[i]  (in-place weighted accumulation / AXPY)
void add_scaled(tensor_t out, tensor_t a, float alpha);

// Fill tensor with zeros
void fill_zero(tensor_t t);

// dst[r, :] = src[indices[r], :] for r in [0, dst.shape[0]).
// dst/src 2D, same dtype + same device; indices 1D int32, length = dst.shape[0].
// Replaces a per-row memcpy loop with a single launch on GPU.
void gather_rows(tensor_t dst, tensor_t src, tensor_t indices);

// out[indices[r], :] += weights[r] * src[r, :] for r in [0, src.shape[0]).
// Within one call, indices[r] are required to be unique (no intra-call write race).
// Across calls, stream ordering serializes accumulations into the same row.
void scatter_add_rows(tensor_t out, tensor_t src, tensor_t indices, tensor_t weights);

} // namespace zedinfer::ops
