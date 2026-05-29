#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/gelu_tanh/cpu/gelu_tanh_cpu.hpp"
#include "backend/ops/gelu_tanh/nvidia/gelu_tanh_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

void gelu_tanh(tensor_t output, tensor_t input) {
    CHECK_SAME_DEVICE(output, input);
    CHECK_SAME_SHAPE(output->shape(), input->shape());
    CHECK_SAME_DTYPE(output->dtype(), input->dtype());
    ASSERT(output->isContiguous() && input->isContiguous(), "gelu_tanh: tensors must be contiguous.");

    const size_t numel = output->numel();

    if (output->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::gelu_tanh(output->data(), input->data(), output->dtype(), numel);
    }

    core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::gelu_tanh(output->data(), input->data(), output->dtype(), numel);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
