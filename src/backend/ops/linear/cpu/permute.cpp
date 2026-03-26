#include "backend/ops/linear/cpu/permute.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cstddef>
#include <cstdint>
#include <omp.h>

namespace zedinfer::ops::cpu {

template <typename T>
static void permute_cols_impl(T* output, const T* input, const int32_t* indices, size_t M, size_t K) {
#pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; ++m) {
        const T* src_row = input + m * K;
        T* dst_row = output + m * K;
        for (size_t k = 0; k < K; ++k) { dst_row[k] = src_row[static_cast<size_t>(indices[k])]; }
    }
}

tensor_t permute_cols(tensor_t input, tensor_t indices) {
    ASSERT(input && indices, "permute_cols requires input and indices.");
    ASSERT(input->shape().size() == 2, "permute_cols expects a 2D input tensor.");
    ASSERT(indices->shape().size() == 1, "permute_cols expects 1D indices.");
    ASSERT(indices->dtype() == ZEDINFER_DTYPE_I32, "permute_cols indices must be INT32.");
    ASSERT(input->isContiguous() && indices->isContiguous(), "permute_cols requires contiguous tensors.");
    ASSERT(input->dim(1) == indices->dim(0), "permute_cols indices length must match input columns.");

    const size_t M = input->dim(0);
    const size_t K = input->dim(1);
    const int32_t* perm = reinterpret_cast<const int32_t*>(indices->data());

    auto output = Tensor::create(input->shape(), input->dtype(), input->deviceType(), input->deviceId());

    switch (input->dtype()) {
        case ZEDINFER_DTYPE_F32:
            permute_cols_impl(reinterpret_cast<float*>(output->data()), reinterpret_cast<const float*>(input->data()),
                              perm, M, K);
            return output;
        case ZEDINFER_DTYPE_BF16:
            permute_cols_impl(reinterpret_cast<zedinfer::bf16_t*>(output->data()),
                              reinterpret_cast<const zedinfer::bf16_t*>(input->data()), perm, M, K);
            return output;
        case ZEDINFER_DTYPE_F16:
            permute_cols_impl(reinterpret_cast<zedinfer::fp16_t*>(output->data()),
                              reinterpret_cast<const zedinfer::fp16_t*>(input->data()), perm, M, K);
            return output;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(input->dtype());
    }
}

} // namespace zedinfer::ops::cpu
