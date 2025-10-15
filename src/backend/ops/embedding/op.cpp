#include "backend/ops/embedding/op.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/embedding/cpu/embedding_cpu.hpp"

#include "utils/check.hpp"

namespace neollm::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == NEOLLM_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->numel(), out->dim(1));
    }

    neollm::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case NEOLLM_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->numel(), out->dim(1));
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
