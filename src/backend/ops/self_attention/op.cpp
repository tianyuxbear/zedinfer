#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/self_attention/cpu/self_attention_cpu.hpp"
#include "backend/ops/self_attention/nvidia/self_attention_nvidia.cuh"
#include "utils/check.hpp"

namespace zedinfer::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous.");

    // always support cpu calculation
    if (attn_val->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->dim(0), attn_val->dim(1), attn_val->dim(2), k->dim(0), k->dim(1), k->dim(2));
    }

    zedinfer::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->dim(0), attn_val->dim(1), attn_val->dim(2), k->dim(0), k->dim(1), k->dim(2));
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->dim(0), attn_val->dim(1), attn_val->dim(2), k->dim(0), k->dim(1), k->dim(2));
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
