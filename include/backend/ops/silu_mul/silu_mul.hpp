#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// Fused silu-and-multiply gate op used by Qwen3.5 GDN output gating.
//
//   out[i] = silu(z[i]) * x[i]
//   silu(z) = z / (1 + exp(-z))
//
// All three tensors must have identical shapes and dtype (bf16 on NVIDIA,
// f32 on CPU). This is a standalone allocation: out may alias neither z nor x.
void silu_mul(tensor_t out, tensor_t z, tensor_t x);

} // namespace zedinfer::ops
