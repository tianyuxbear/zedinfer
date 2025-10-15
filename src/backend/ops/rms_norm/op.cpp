#include "backend/ops/rms_norm/op.hpp"
#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/rms_norm/cpu/rms_norm_cpu.hpp"
#include "utils/check.hpp"

namespace neollm::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMS_Norm: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == NEOLLM_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1));
    }

    neollm::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case NEOLLM_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1));
#ifdef ENABLE_NVIDIA_API
    case NEOLLM_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace neollm::ops
