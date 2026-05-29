#include "backend/ops/gelu_tanh/cpu/gelu_tanh_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>

namespace zedinfer::ops::cpu {

namespace {

inline float gelu_tanh_scalar(float x) {
    constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float c = 0.044715f;
    const float     inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

template <typename T> void gelu_tanh_(T* y, const T* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float xv = zedinfer::utils::cast<float>(x[i]);
        y[i]           = zedinfer::utils::cast<T>(gelu_tanh_scalar(xv));
    }
}

template <> void gelu_tanh_<float>(float* y, const float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = gelu_tanh_scalar(x[i]);
    }
}

} // namespace

void gelu_tanh(std::byte* output, const std::byte* input, zedinferDataType_t type, size_t numel) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return gelu_tanh_(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input), numel);
        case ZEDINFER_DTYPE_F16:
            return gelu_tanh_(reinterpret_cast<zedinfer::fp16_t*>(output),
                              reinterpret_cast<const zedinfer::fp16_t*>(input), numel);
        case ZEDINFER_DTYPE_BF16:
            return gelu_tanh_(reinterpret_cast<zedinfer::bf16_t*>(output),
                              reinterpret_cast<const zedinfer::bf16_t*>(input), numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
