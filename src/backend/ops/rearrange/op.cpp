#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/rearrange/cpu/rearrange_cpu.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {
void rearrange(tensor_t out, const tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // always support cpu calculation
    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->dtype(), out->numel(), out->shape(), out->strides(), in->shape(), in->strides());
    }

    zedinfer::core::context().setDevice(out->deviceType(), in->deviceId());

    switch (out->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), out->numel(), out->shape(), out->strides(), in->shape(), in->strides());
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
