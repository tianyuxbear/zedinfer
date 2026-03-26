#include "backend/ops/rope/cpu/rope_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <omp.h>
#include <vector>

namespace zedinfer::ops::cpu {

template <typename T>
void rope_(T* output, const T* input, const int64_t* pos_ids, size_t seq_len, size_t num_heads, size_t head_dim,
           const std::vector<float>& inv_theta) {
    const size_t half_dim = head_dim / 2;

    // Pre-allocate per-thread sin/cos buffers
    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<float>> thread_cos(max_threads);
    std::vector<std::vector<float>> thread_sin(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_cos[t].resize(half_dim);
        thread_sin[t].resize(half_dim);
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seq_len; ++i) {
        const int tid = omp_get_thread_num();
        float* cos_buf = thread_cos[tid].data();
        float* sin_buf = thread_sin[tid].data();
        const int64_t p_i = pos_ids[i];

        for (size_t k = 0; k < half_dim; ++k) {
            float angle = static_cast<float>(p_i) * inv_theta[k];
#ifdef __linux__
            sincosf(angle, &sin_buf[k], &cos_buf[k]);
#else
            cos_buf[k] = std::cos(angle);
            sin_buf[k] = std::sin(angle);
#endif
        }

        for (size_t j = 0; j < num_heads; ++j) {
            T* out_head = output + (i * num_heads + j) * head_dim;
            const T* in_head = input + (i * num_heads + j) * head_dim;

#pragma omp simd
            for (size_t k = 0; k < half_dim; ++k) {
                float a = zedinfer::utils::cast<float>(in_head[k]);
                float b = zedinfer::utils::cast<float>(in_head[k + half_dim]);
                out_head[k] = zedinfer::utils::cast<T>(a * cos_buf[k] - b * sin_buf[k]);
                out_head[k + half_dim] = zedinfer::utils::cast<T>(b * cos_buf[k] + a * sin_buf[k]);
            }
        }
    }
}

void rope(std::byte* output, const std::byte* input, const std::byte* pos_ids, float theta, zedinferDataType_t type,
          size_t seq_len, size_t num_heads, size_t head_dim) {
    // Pre-compute inverse theta frequencies (once per call, tiny cost)
    const size_t half_dim = head_dim / 2;
    std::vector<float> inv_theta(half_dim);
    for (size_t k = 0; k < half_dim; ++k) {
        float expo = static_cast<float>(2 * k) / static_cast<float>(head_dim);
        inv_theta[k] = 1.0f / std::pow(theta, expo);
    }

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return rope_(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input),
                         reinterpret_cast<const int64_t*>(pos_ids), seq_len, num_heads, head_dim, inv_theta);
        case ZEDINFER_DTYPE_F16:
            return rope_(reinterpret_cast<zedinfer::fp16_t*>(output), reinterpret_cast<const zedinfer::fp16_t*>(input),
                         reinterpret_cast<const int64_t*>(pos_ids), seq_len, num_heads, head_dim, inv_theta);
        case ZEDINFER_DTYPE_BF16:
            return rope_(reinterpret_cast<zedinfer::bf16_t*>(output), reinterpret_cast<const zedinfer::bf16_t*>(input),
                         reinterpret_cast<const int64_t*>(pos_ids), seq_len, num_heads, head_dim, inv_theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
