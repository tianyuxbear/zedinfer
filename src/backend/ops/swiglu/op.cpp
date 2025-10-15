#include "backend/ops/swiglu/op.hpp"
#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/swiglu/cpu/swiglu_cpu.hpp"
#include "utils/check.hpp"

namespace neollm::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == NEOLLM_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }

    neollm::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case NEOLLM_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
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
