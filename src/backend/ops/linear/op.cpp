#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/linear/cpu/linear_cpu.hpp"
#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

#include <cstddef>

namespace zedinfer::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        // Only support contiguous inputs with same shape for now.
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
               "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        // Only support contiguous inputs with same shape for now.
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
               "Linear: all tensors must be contiguous.");
    }

    std::byte* bias_data = nullptr;
    if (bias) {
        bias_data = bias->data();
    }

    // always support cpu calculation
    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), out->dim(0), out->dim(1),
                           in->dim(1));
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), out->dim(0),
                               out->dim(1), in->dim(1));
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), out->dim(0),
                                  out->dim(1), in->dim(1));
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
