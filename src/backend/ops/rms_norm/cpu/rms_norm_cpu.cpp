#include "backend/ops/rms_norm/cpu/rms_norm_cpu.hpp"
#include "backend/ops/rms_norm/cpu/sdot.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstddef>
#include <omp.h>
#include <type_traits>
#include <vector>

namespace zedinfer::ops::cpu {

template <typename T>
void rms_norm_(T* output, const T* input, const T* weight, float eps, size_t seq_len, size_t hidden_size) {
    if constexpr (std::is_same_v<T, float>) {
        // FP32 path: no conversion needed
#pragma omp parallel for
        for (size_t i = 0; i < seq_len; ++i) {
            const float* in_row = input + i * hidden_size;
            float* out_row = output + i * hidden_size;

            float sq_sum = sdot(in_row, in_row, hidden_size);
            float rstd = 1.0f / std::sqrt(sq_sum / static_cast<float>(hidden_size) + eps);

            for (size_t j = 0; j < hidden_size; ++j) { out_row[j] = weight[j] * in_row[j] * rstd; }
        }
    } else {
        // BF16/FP16 path: pre-allocate per-thread conversion buffers
        const int max_threads = omp_get_max_threads();
        std::vector<std::vector<float>> thread_in(max_threads);
        std::vector<std::vector<float>> thread_out(max_threads);
        for (int t = 0; t < max_threads; ++t) {
            thread_in[t].resize(hidden_size);
            thread_out[t].resize(hidden_size);
        }

        // Convert weight once (shared, read-only)
        std::vector<float> weight_f32(hidden_size);
        if constexpr (std::is_same_v<T, zedinfer::bf16_t>) {
            zedinfer::utils::bf16_to_fp32_batch(weight_f32.data(), weight, hidden_size);
        } else {
            zedinfer::utils::fp16_to_fp32_batch_f16c(weight_f32.data(), weight, hidden_size);
        }

#pragma omp parallel for
        for (size_t i = 0; i < seq_len; ++i) {
            const int tid = omp_get_thread_num();
            float* in_f32 = thread_in[tid].data();
            float* out_f32 = thread_out[tid].data();

            const T* in_row = input + i * hidden_size;
            T* out_row = output + i * hidden_size;

            if constexpr (std::is_same_v<T, zedinfer::bf16_t>) {
                zedinfer::utils::bf16_to_fp32_batch(in_f32, in_row, hidden_size);
            } else {
                zedinfer::utils::fp16_to_fp32_batch_f16c(in_f32, in_row, hidden_size);
            }

            float sq_sum = sdot(in_f32, in_f32, hidden_size);
            float rstd = 1.0f / std::sqrt(sq_sum / static_cast<float>(hidden_size) + eps);

            for (size_t j = 0; j < hidden_size; ++j) { out_f32[j] = weight_f32[j] * in_f32[j] * rstd; }

            if constexpr (std::is_same_v<T, zedinfer::bf16_t>) {
                zedinfer::utils::fp32_to_bf16_batch(out_row, out_f32, hidden_size);
            } else {
                zedinfer::utils::fp32_to_fp16_batch_f16c(out_row, out_f32, hidden_size);
            }
        }
    }
}

void rms_norm(std::byte* output, const std::byte* input, const std::byte* weight, float eps, zedinferDataType_t type,
              size_t seq_len, size_t hidden_size) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return rms_norm_(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input),
                             reinterpret_cast<const float*>(weight), eps, seq_len, hidden_size);
        case ZEDINFER_DTYPE_F16:
            return rms_norm_(reinterpret_cast<zedinfer::fp16_t*>(output),
                             reinterpret_cast<const zedinfer::fp16_t*>(input),
                             reinterpret_cast<const zedinfer::fp16_t*>(weight), eps, seq_len, hidden_size);
        case ZEDINFER_DTYPE_BF16:
            return rms_norm_(reinterpret_cast<zedinfer::bf16_t*>(output),
                             reinterpret_cast<const zedinfer::bf16_t*>(input),
                             reinterpret_cast<const zedinfer::bf16_t*>(weight), eps, seq_len, hidden_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
