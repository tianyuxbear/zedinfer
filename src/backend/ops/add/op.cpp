#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/add/cpu/add_cpu.hpp"
#include "backend/ops/add/nvidia/add_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    // always support cpu calculation
    if (c->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

    zedinfer::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
