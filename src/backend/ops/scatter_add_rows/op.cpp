#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/scatter_add_rows/cpu/scatter_add_rows_cpu.hpp"
#include "backend/ops/scatter_add_rows/nvidia/scatter_add_rows_nvidia.cuh"
#include "utils/check.hpp"

namespace zedinfer::ops {

void scatter_add_rows(tensor_t out, tensor_t src, tensor_t indices, tensor_t weights) {
    ASSERT(out && src && indices && weights, "scatter_add_rows: tensors must be non-null.");
    ASSERT(out->isContiguous() && src->isContiguous() && indices->isContiguous() && weights->isContiguous(),
           "scatter_add_rows: tensors must be contiguous.");
    CHECK_SAME_DEVICE(out, src);
    CHECK_SAME_DEVICE(out, indices);
    CHECK_SAME_DEVICE(out, weights);
    CHECK_SAME_DTYPE(out->dtype(), src->dtype());
    ASSERT(indices->dtype() == ZEDINFER_DTYPE_I32, "scatter_add_rows: indices must be int32.");
    ASSERT(weights->dtype() == ZEDINFER_DTYPE_F32, "scatter_add_rows: weights must be float32.");
    ASSERT(out->shape().size() == 2 && src->shape().size() == 2, "scatter_add_rows: out/src must be 2D.");
    ASSERT(indices->shape().size() == 1 && weights->shape().size() == 1,
           "scatter_add_rows: indices/weights must be 1D.");
    ASSERT(src->shape()[0] == indices->shape()[0] && src->shape()[0] == weights->shape()[0],
           "scatter_add_rows: src rows must equal indices/weights length.");
    ASSERT(out->shape()[1] == src->shape()[1], "scatter_add_rows: out cols must equal src cols.");

    const std::size_t num_rows = src->shape()[0];
    const std::size_t row_elements = src->shape()[1];
    const auto* idx = reinterpret_cast<const std::int32_t*>(indices->data());
    const auto* w = reinterpret_cast<const float*>(weights->data());

    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::scatter_add_rows(out->data(), src->data(), idx, w, out->dtype(), num_rows, row_elements);
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::scatter_add_rows(out->data(), src->data(), idx, w, out->dtype(), num_rows, row_elements);
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::scatter_add_rows(out->data(), src->data(), idx, w, out->dtype(), num_rows, row_elements);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
