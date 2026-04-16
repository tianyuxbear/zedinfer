#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/rope/cpu/rope_cpu.hpp"
#include "backend/ops/rope/nvidia/rope_nvidia.cuh"
#include "utils/check.hpp"

namespace zedinfer::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->dim(0), out->dim(1),
                         out->dim(2));
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->dim(0), out->dim(1),
                             out->dim(2));
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), out->dim(0), out->dim(1),
                                out->dim(2));
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void rope_qk(tensor_t q_out, tensor_t k_out, tensor_t q_in, tensor_t k_in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(q_out, k_out, q_in, k_in, pos_ids);
    CHECK_SAME_SHAPE(q_out->shape(), q_in->shape());
    CHECK_SAME_SHAPE(k_out->shape(), k_in->shape());
    CHECK_SAME_DTYPE(q_out->dtype(), q_in->dtype());
    CHECK_SAME_DTYPE(k_out->dtype(), k_in->dtype());
    CHECK_SAME_DTYPE(q_out->dtype(), k_out->dtype());
    ASSERT(q_out->isContiguous() && k_out->isContiguous() && q_in->isContiguous() && k_in->isContiguous()
               && pos_ids->isContiguous(),
           "RopeQK: all tensors must be contiguous.");

    if (q_out->deviceType() == ZEDINFER_DEVICE_CPU) {
        rope(q_out, q_in, pos_ids, theta);
        rope(k_out, k_in, pos_ids, theta);
        return;
    }

    zedinfer::core::context().setDevice(q_out->deviceType(), q_out->deviceId());

    switch (q_out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            rope(q_out, q_in, pos_ids, theta);
            rope(k_out, k_in, pos_ids, theta);
            return;
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::rope_qk(q_out->data(), k_out->data(), q_in->data(), k_in->data(), pos_ids->data(), theta,
                                   q_out->dtype(), q_out->dim(0), q_out->dim(1), k_out->dim(1), q_out->dim(2));
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
