#include "backend/ops/rope/cpu/rope_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <vector>

namespace zedinfer::ops::cpu {

template <typename T>
void rope_(T *output, const T *input, const int64_t *pos_ids, size_t seq_len, size_t num_heads, size_t head_dim, const std::vector<float> &inv_theta) {

    size_t half_dim = head_dim / 2;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        std::vector<float> cos_cache(half_dim);
        std::vector<float> sin_cache(half_dim);
        int64_t p_i = pos_ids[i];

#pragma omp simd
        for (size_t k = 0; k < half_dim; ++k) {
            float angle = static_cast<float>(p_i * inv_theta[k]);
#ifdef __linux__
            sincosf(angle, &sin_cache[k], &cos_cache[k]);
#else
            cos_cache[k] = std::cos(angle);
            sin_cache[k] = std::sin(angle);
#endif
        }

        for (size_t j = 0; j < num_heads; ++j) {
            T *output_t = output + (i * num_heads + j) * head_dim;
            const T *input_t = input + (i * num_heads + j) * head_dim;

#pragma omp simd
            for (size_t k = 0; k < half_dim; ++k) {
                float cos_val = cos_cache[k];
                float sin_val = sin_cache[k];
                float a = zedinfer::utils::cast<float>(input_t[k]);
                float b = zedinfer::utils::cast<float>(input_t[k + half_dim]);
                output_t[k] = zedinfer::utils::cast<T>(a * cos_val - b * sin_val);
                output_t[k + half_dim] = zedinfer::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

void rope(std::byte *output, const std::byte *input, const std::byte *pos_ids, float theta, zedinferDataType_t type, size_t seq_len, size_t num_heads, size_t head_dim) {

    size_t half_dim = head_dim / 2;
    std::vector<float> inv_theta(half_dim);
    for (size_t k = 0; k < half_dim; ++k) {
        float expo = static_cast<float>(2 * k) / static_cast<float>(head_dim);
        inv_theta[k] = 1.0 / std::pow(theta, expo);
    }

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(output),
                     reinterpret_cast<const float *>(input),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_len, num_heads, head_dim, inv_theta);
    case ZEDINFER_DTYPE_F16:
        return rope_(reinterpret_cast<zedinfer::fp16_t *>(output),
                     reinterpret_cast<const zedinfer::fp16_t *>(input),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_len, num_heads, head_dim, inv_theta);
    case ZEDINFER_DTYPE_BF16:
        return rope_(reinterpret_cast<zedinfer::bf16_t *>(output),
                     reinterpret_cast<const zedinfer::bf16_t *>(input),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_len, num_heads, head_dim, inv_theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::cpu
