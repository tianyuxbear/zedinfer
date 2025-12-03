#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/rms_norm/cpu/rms_norm_cpu.hpp"
#include "backend/ops/rms_norm/nvidia/rms_norm_nvidia.cuh"
#include "utils/check.hpp"

namespace zedinfer::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMS_Norm: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1));
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1));
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1));
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
