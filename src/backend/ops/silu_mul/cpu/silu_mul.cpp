#include "backend/ops/silu_mul/silu_mul.hpp"

#include <cmath>

namespace zedinfer::ops {

#ifndef ENABLE_NVIDIA_API
// CPU fallback for builds without an NVIDIA target. Operates on fp32.
void silu_mul(tensor_t out, tensor_t z, tensor_t x) {
    const size_t n = out->numel();
    float*       ov = reinterpret_cast<float*>(out->data());
    const float* zv = reinterpret_cast<const float*>(z->data());
    const float* xv = reinterpret_cast<const float*>(x->data());
    for (size_t i = 0; i < n; ++i) {
        const float silu_z = zv[i] / (1.0f + std::exp(-zv[i]));
        ov[i] = silu_z * xv[i];
    }
}
#endif

} // namespace zedinfer::ops
