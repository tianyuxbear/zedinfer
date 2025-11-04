#include "backend/ops/rearrange/cpu/rearrange_cpu.hpp"
#include "utils/check.hpp"
#include "utils/ops.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstddef>

template <typename T>
void rearrange_(T *out, const T *in, size_t numel, const std::vector<size_t> &out_shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<size_t> &in_shape, const std::vector<ptrdiff_t> &in_strides) {
    size_t ndim = out_shape.size();
    for (size_t i = 0; i < numel; ++i) {
        size_t out_idx = zedinfer::ops::indexToOffset(i, ndim, out_shape, out_strides);
        size_t in_idx = zedinfer::ops::indexToOffset(i, ndim, in_shape, in_strides);
        out[out_idx] = in[in_idx];
    }
}

namespace zedinfer::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, zedinferDataType_t type, size_t numel, const std::vector<size_t> &out_shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<size_t> &in_shape, const std::vector<ptrdiff_t> &in_strides) {
    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), numel, out_shape, out_strides, in_shape, in_strides);
    case ZEDINFER_DTYPE_BF16:
        return rearrange_(reinterpret_cast<zedinfer::bf16_t *>(out), reinterpret_cast<const zedinfer::bf16_t *>(in),
                          numel, out_shape, out_strides, in_shape, in_strides);
    case ZEDINFER_DTYPE_F16:
        return rearrange_(reinterpret_cast<zedinfer::fp16_t *>(out), reinterpret_cast<const zedinfer::fp16_t *>(in),
                          numel, out_shape, out_strides, in_shape, in_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::cpu
