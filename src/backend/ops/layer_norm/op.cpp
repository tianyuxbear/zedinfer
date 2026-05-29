#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/layer_norm/cpu/layer_norm_bias_cpu.hpp"
#include "backend/ops/layer_norm/nvidia/layer_norm_bias_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

void layer_norm_bias(tensor_t output, tensor_t input, tensor_t weight, tensor_t bias, float eps) {
    CHECK_SAME_DEVICE(output, input, weight, bias);
    CHECK_SAME_SHAPE(output->shape(), input->shape());
    CHECK_SAME_DTYPE(output->dtype(), input->dtype(), weight->dtype(), bias->dtype());
    ASSERT(output->isContiguous() && input->isContiguous() && weight->isContiguous() && bias->isContiguous(),
           "layer_norm_bias: all tensors must be contiguous.");

    const size_t seq_len     = output->dim(0);
    const size_t hidden_size = output->dim(1);

    if (output->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::layer_norm_bias(output->data(), input->data(), weight->data(), bias->data(), eps, output->dtype(),
                                    seq_len, hidden_size);
    }

    core::context().setDevice(output->deviceType(), output->deviceId());

    switch (output->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::layer_norm_bias(output->data(), input->data(), weight->data(), bias->data(), eps,
                                           output->dtype(), seq_len, hidden_size);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
