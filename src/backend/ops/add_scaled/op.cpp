#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/add_scaled/cpu/add_scaled_cpu.hpp"
#include "backend/ops/add_scaled/nvidia/add_scaled_nvidia.cuh"
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

} // namespace zedinfer::ops
