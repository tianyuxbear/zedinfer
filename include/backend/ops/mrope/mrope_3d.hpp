#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/hybrid_forward_config.hpp"

namespace zedinfer::ops {

// 3D multimodal rotary position embedding with partial rotary factor and
// interleaved pair layout. Used by Qwen3.5's softmax (full-attention) layer.
//
// In-place rotates the first (head_dim * cfg.partial_factor) dims of x. The
// rotary dims are split into dim-pairs (2*pi, 2*pi+1); cfg.section partitions
// the half-dim pairs across the (t, h, w) axes:
//   pi in [0, section[0])                       -> rotate against pos_t
//   pi in [section[0], section[0]+section[1])   -> rotate against pos_h
//   pi in [section[0]+section[1], half)         -> rotate against pos_w
//
// Per dim-pair pi the rotation is:
//   freq  = cfg.theta ^ (-(2*pi) / (2*half))      ; same convention as 1D RoPE
//   angle = pos[axis][n] * freq
//   (a, b) = (x[n, hd, 2*pi], x[n, hd, 2*pi+1])
//   x[n, hd, 2*pi]   = a*cos(angle) - b*sin(angle)
//   x[n, hd, 2*pi+1] = a*sin(angle) + b*cos(angle)
//
// Dims [Dh_rot, head_dim) are pass-through (unchanged).
//
// Tensor layouts:
//   x:           [N, num_heads, head_dim]  bf16, contiguous; in-place.
//   pos_ids_thw: [3, N]                    int32; row 0 = t, row 1 = h, row 2 = w.
void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const model::MRoPEConfig& cfg);

} // namespace zedinfer::ops
