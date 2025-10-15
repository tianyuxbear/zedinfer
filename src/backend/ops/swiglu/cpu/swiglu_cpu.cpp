#include "backend/ops/swiglu/cpu/swiglu_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstddef>
#include <omp.h>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
#pragma omp parallel for simd
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, neollm::bf16_t> || std::is_same_v<T, neollm::fp16_t>) {
            float gate_i = neollm::utils::cast<float>(gate[i]);
            float up_i = neollm::utils::cast<float>(up[i]);
            float denom = 1.0f + std::exp(-gate_i);
            float ans = up_i * gate_i / denom;
            out[i] = neollm::utils::cast<T>(ans);
        } else {
            T denom = T{1} + std::exp(-gate[i]);
            out[i] = up[i] * gate[i] / denom;
        }
    }
}

namespace neollm::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, NeollmDataType_t type, size_t numel) {
    switch (type) {
    case NEOLLM_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
    case NEOLLM_DTYPE_BF16:
        return swiglu_(reinterpret_cast<neollm::bf16_t *>(out), reinterpret_cast<const neollm::bf16_t *>(gate),
                       reinterpret_cast<const neollm::bf16_t *>(up), numel);
    case NEOLLM_DTYPE_F16:
        return swiglu_(reinterpret_cast<neollm::fp16_t *>(out), reinterpret_cast<const neollm::fp16_t *>(gate),
                       reinterpret_cast<const neollm::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace neollm::ops::cpu
