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
void rms_norm_(T *output, const T *input, const T *weight, float eps, size_t seq_len, size_t hidden_size) {
#pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        T *output_t = output + i * hidden_size;
        const T *input_t = input + i * hidden_size;

        if constexpr (std::is_same_v<T, zedinfer::bf16_t>) {
            float rms{}, square_sum{}, avg_square_sum{};
            std::vector<float> in_f32(hidden_size);
            std::vector<float> weight_f32(hidden_size);
            std::vector<float> output_f32(hidden_size);

            zedinfer::utils::bf16_to_fp32_batch(in_f32.data(), input_t, hidden_size);
            zedinfer::utils::bf16_to_fp32_batch(weight_f32.data(), weight, hidden_size);

            square_sum = sdot(in_f32.data(), in_f32.data(), hidden_size);
            avg_square_sum = square_sum / static_cast<float>(hidden_size);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < hidden_size; ++j) {
                output_f32[j] = weight_f32[j] * in_f32[j] / rms;
            }
            zedinfer::utils::fp32_to_bf16_batch(output_t, output_f32.data(), hidden_size);

        } else if constexpr (std::is_same_v<T, zedinfer::fp16_t>) {
            float rms{}, square_sum{}, avg_square_sum{};
            std::vector<float> in_f32(hidden_size);
            std::vector<float> weight_f32(hidden_size);
            std::vector<float> output_f32(hidden_size);

            zedinfer::utils::fp16_to_fp32_batch_f16c(in_f32.data(), input_t, hidden_size);
            zedinfer::utils::fp16_to_fp32_batch_f16c(weight_f32.data(), weight, hidden_size);

            square_sum = sdot(in_f32.data(), in_f32.data(), hidden_size);
            avg_square_sum = square_sum / static_cast<float>(hidden_size);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < hidden_size; ++j) {
                output_f32[j] = weight_f32[j] * in_f32[j] / rms;
            }
            zedinfer::utils::fp32_to_fp16_batch_f16c(output_t, output_f32.data(), hidden_size);

        } else {
            T rms{}, square_sum{}, avg_square_sum{};
            square_sum = sdot(input_t, input_t, hidden_size);
            avg_square_sum = square_sum / static_cast<T>(hidden_size);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < hidden_size; ++j) {
                output_t[j] = weight[j] * input_t[j] / rms;
            }
        }
    }
}

void rms_norm(std::byte *output, const std::byte *input, const std::byte *weight, float eps, zedinferDataType_t type, size_t seq_len, size_t hidden_size) {
    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(output),
                         reinterpret_cast<const float *>(input),
                         reinterpret_cast<const float *>(weight),
                         eps, seq_len, hidden_size);
    case ZEDINFER_DTYPE_F16:
        return rms_norm_(reinterpret_cast<zedinfer::fp16_t *>(output),
                         reinterpret_cast<const zedinfer::fp16_t *>(input),
                         reinterpret_cast<const zedinfer::fp16_t *>(weight),
                         eps, seq_len, hidden_size);
    case ZEDINFER_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<zedinfer::bf16_t *>(output),
                         reinterpret_cast<const zedinfer::bf16_t *>(input),
                         reinterpret_cast<const zedinfer::bf16_t *>(weight),
                         eps, seq_len, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::cpu
