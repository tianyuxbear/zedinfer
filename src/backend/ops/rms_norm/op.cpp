#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/rms_norm/cpu/rms_norm_cpu.hpp"
#include "backend/ops/rms_norm/nvidia/rms_norm_nvidia.cuh"
#include "utils/check.hpp"

namespace zedinfer::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps, bool add_one_to_weight) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMS_Norm: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1),
                             add_one_to_weight);
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0), out->dim(1),
                                 add_one_to_weight);
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), out->dim(0),
                                    out->dim(1), add_one_to_weight);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void fused_add_rms_norm(tensor_t input, tensor_t residual, tensor_t weight, float eps, bool add_one_to_weight) {
    CHECK_SAME_DEVICE(input, residual, weight);
    CHECK_SAME_SHAPE(input->shape(), residual->shape());
    CHECK_SAME_DTYPE(input->dtype(), residual->dtype(), weight->dtype());
    ASSERT(input->isContiguous() && residual->isContiguous() && weight->isContiguous(),
           "fused_add_rms_norm: all tensors must be contiguous.");

    // CPU path: not perf-critical, just sequence add + rms_norm in place.
    if (input->deviceType() == ZEDINFER_DEVICE_CPU) {
        // residual := residual + input
        // input    := rmsnorm(residual) * (weight + bias)
        // Doing the two sequentially preserves the same final state as the
        // GPU-side fused kernel, with the small caveat that intermediate
        // memory traffic is 2x. CPU is not the perf bottleneck.
        const size_t numel = input->dim(0) * input->dim(1);
        // First: residual += input  (use the add op's CPU fallback semantics)
        // Implemented inline to avoid pulling in the add op header dependency.
        switch (input->dtype()) {
            case ZEDINFER_DTYPE_F32: {
                auto* r = reinterpret_cast<float*>(residual->data());
                auto* i = reinterpret_cast<float*>(input->data());
                for (size_t k = 0; k < numel; ++k) { r[k] += i[k]; }
                break;
            }
            default:
                throw std::runtime_error("fused_add_rms_norm CPU fallback supports F32 only");
        }
        return cpu::rms_norm(input->data(), residual->data(), weight->data(), eps, input->dtype(), input->dim(0),
                             input->dim(1), add_one_to_weight);
    }

    zedinfer::core::context().setDevice(input->deviceType(), input->deviceId());

    switch (input->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::fused_add_rms_norm(input->data(), residual->data(), weight->data(), eps, input->dtype(),
                                              input->dim(0), input->dim(1), add_one_to_weight);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
