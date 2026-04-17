#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/fill_zero/cpu/fill_zero_cpu.hpp"
#include "backend/ops/fill_zero/nvidia/fill_zero_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

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
