#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/rope/cpu/rope_cpu.hpp"
#include "utils/check.hpp"

#include <cmath>

namespace neollm::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "Rope: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == NEOLLM_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->dim(0), out->dim(1), out->dim(2));
    }

    neollm::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case NEOLLM_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->dim(0), out->dim(1), out->dim(2));
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
