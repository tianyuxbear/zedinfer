#include "backend/ops/scatter_add_rows/cpu/scatter_add_rows_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

namespace zedinfer::ops::cpu {

template <typename T>
static void scatter_add_rows_impl(T* out, const T* src, const std::int32_t* indices, const float* weights,
                                  std::size_t num_rows, std::size_t row_elements) {
    for (std::size_t r = 0; r < num_rows; ++r) {
        const std::size_t out_row = static_cast<std::size_t>(indices[r]);
        const float w = weights[r];
        T* out_ptr = out + out_row * row_elements;
        const T* src_ptr = src + r * row_elements;
        for (std::size_t c = 0; c < row_elements; ++c) {
            float acc = zedinfer::utils::cast<float>(out_ptr[c]) + w * zedinfer::utils::cast<float>(src_ptr[c]);
            out_ptr[c] = zedinfer::utils::cast<T>(acc);
        }
    }
}

void scatter_add_rows(std::byte* out, const std::byte* src, const std::int32_t* indices, const float* weights,
                      zedinferDataType_t type, std::size_t num_rows, std::size_t row_elements) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return scatter_add_rows_impl(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(src), indices,
                                         weights, num_rows, row_elements);
        case ZEDINFER_DTYPE_F16:
            return scatter_add_rows_impl(reinterpret_cast<zedinfer::fp16_t*>(out),
                                         reinterpret_cast<const zedinfer::fp16_t*>(src), indices, weights, num_rows,
                                         row_elements);
        case ZEDINFER_DTYPE_BF16:
            return scatter_add_rows_impl(reinterpret_cast<zedinfer::bf16_t*>(out),
                                         reinterpret_cast<const zedinfer::bf16_t*>(src), indices, weights, num_rows,
                                         row_elements);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
