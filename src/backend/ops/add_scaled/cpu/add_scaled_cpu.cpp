#include "backend/ops/add_scaled/cpu/add_scaled_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

namespace zedinfer::ops::cpu {

template <typename T> static void add_scaled_impl(T* out, const T* a, float alpha, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float f_out = zedinfer::utils::cast<float>(out[i]);
        float f_a = zedinfer::utils::cast<float>(a[i]);
        out[i] = zedinfer::utils::cast<T>(f_out + alpha * f_a);
    }
}

void add_scaled(std::byte* out, const std::byte* a, float alpha, zedinferDataType_t type, size_t numel) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return add_scaled_impl(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a), alpha, numel);
        case ZEDINFER_DTYPE_F16:
            return add_scaled_impl(reinterpret_cast<zedinfer::fp16_t*>(out),
                                   reinterpret_cast<const zedinfer::fp16_t*>(a), alpha, numel);
        case ZEDINFER_DTYPE_BF16:
            return add_scaled_impl(reinterpret_cast<zedinfer::bf16_t*>(out),
                                   reinterpret_cast<const zedinfer::bf16_t*>(a), alpha, numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
