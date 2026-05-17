#include "backend/ops/attn_output_gate/attn_output_gate.hpp"

#include <cmath>

namespace zedinfer::ops {

#ifndef ENABLE_NVIDIA_API
// CPU fallback for builds without an NVIDIA target. Operates on fp32 in place.
void attn_output_gate(tensor_t attn, tensor_t g) {
    const size_t n = attn->numel();
    float* a = reinterpret_cast<float*>(attn->data());
    const float* gv = reinterpret_cast<const float*>(g->data());
    for (size_t i = 0; i < n; ++i) {
        const float s = 1.0f / (1.0f + std::exp(-gv[i]));
        a[i] *= s;
    }
}
#endif

} // namespace zedinfer::ops
