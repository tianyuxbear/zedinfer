#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/gather_rows/cpu/gather_rows_cpu.hpp"
#include "backend/ops/gather_rows/nvidia/gather_rows_nvidia.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

void gather_rows(tensor_t dst, tensor_t src, tensor_t indices) {
    ASSERT(dst && src && indices, "gather_rows: tensors must be non-null.");
    ASSERT(dst->isContiguous() && src->isContiguous() && indices->isContiguous(),
           "gather_rows: tensors must be contiguous.");
    CHECK_SAME_DEVICE(dst, src);
    CHECK_SAME_DEVICE(dst, indices);
    CHECK_SAME_DTYPE(dst->dtype(), src->dtype());
    ASSERT(indices->dtype() == ZEDINFER_DTYPE_I32, "gather_rows: indices must be int32.");
    ASSERT(dst->shape().size() == 2 && src->shape().size() == 2, "gather_rows: dst/src must be 2D.");
    ASSERT(indices->shape().size() == 1, "gather_rows: indices must be 1D.");
    ASSERT(dst->shape()[0] == indices->shape()[0], "gather_rows: dst rows must equal indices length.");
    ASSERT(dst->shape()[1] == src->shape()[1], "gather_rows: dst cols must equal src cols.");

    const std::size_t num_rows = dst->shape()[0];
    const std::size_t row_bytes = dst->shape()[1] * dst->elementSize();
    const auto* idx = reinterpret_cast<const std::int32_t*>(indices->data());

    if (dst->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::gather_rows(dst->data(), src->data(), idx, num_rows, row_bytes);
    }

    zedinfer::core::context().setDevice(dst->deviceType(), dst->deviceId());

    switch (dst->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::gather_rows(dst->data(), src->data(), idx, num_rows, row_bytes);
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::gather_rows(dst->data(), src->data(), idx, num_rows, row_bytes);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
