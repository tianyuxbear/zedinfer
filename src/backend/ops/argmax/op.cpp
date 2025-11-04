#include "backend/core/context/context.hpp"
#include "backend/ops/argmax/cpu/argmax_cpu.hpp"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(vals, max_idx, max_val);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(vals->dtype(), max_val->dtype());
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "Argmax: all tensors must be contiguous.");

    // always support cpu calculation
    if (vals->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    zedinfer::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
