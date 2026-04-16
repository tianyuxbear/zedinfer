#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/moe/moe_ops_cpu.hpp"
#include "backend/ops/moe/nvidia/moe_ops_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

void add_scaled(tensor_t out, tensor_t a, float alpha) {
    CHECK_SAME_DEVICE(out, a);
    CHECK_SAME_SHAPE(out->shape(), a->shape());
    CHECK_SAME_DTYPE(out->dtype(), a->dtype());
    ASSERT(out->isContiguous() && a->isContiguous(), "add_scaled: tensors must be contiguous.");

    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::add_scaled(out->data(), a->data(), alpha, out->dtype(), out->numel());
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::add_scaled(out->data(), a->data(), alpha, out->dtype(), out->numel());
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::add_scaled(out->data(), a->data(), alpha, out->dtype(), out->numel());
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void fill_zero(tensor_t t) {
    ASSERT(t && t->isContiguous(), "fill_zero: tensor must be non-null and contiguous.");

    size_t size_bytes = t->numel() * t->elementSize();

    if (t->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::fill_zero(t->data(), size_bytes);
    }

    zedinfer::core::context().setDevice(t->deviceType(), t->deviceId());

    switch (t->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::fill_zero(t->data(), size_bytes);
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::fill_zero(t->data(), size_bytes);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
