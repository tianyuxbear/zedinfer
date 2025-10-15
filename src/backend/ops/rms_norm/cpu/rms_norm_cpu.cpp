#include "backend/ops/rms_norm/cpu/rms_norm_cpu.hpp"
#include "backend/ops/rms_norm/cpu/sdot.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstddef>
#include <omp.h>
#include <type_traits>
#include <vector>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t nrow, size_t ncol) {
#pragma omp parallel for
    for (size_t i = 0; i < nrow; ++i) {
        T *out_t = out + i * ncol;
        const T *in_t = in + i * ncol;

        if constexpr (std::is_same_v<T, neollm::bf16_t>) {
            float rms{}, square_sum{}, avg_square_sum{};
            std::vector<float> in_f32(ncol);
            std::vector<float> weight_f32(ncol);
            std::vector<float> out_f32(ncol);

            // for (size_t j = 0; j < ncol; ++j) {
            //     in_f32[j] = neollm::utils::cast<float>(in_t[j]);
            //     weight_f32[j] = neollm::utils::cast<float>(weight[j]);
            // }
            neollm::utils::bf16_to_fp32_batch(in_f32.data(), in_t, ncol);
            neollm::utils::bf16_to_fp32_batch(weight_f32.data(), weight, ncol);

            square_sum = neollm::ops::cpu::sdot(in_f32.data(), in_f32.data(), ncol);
            avg_square_sum = square_sum / static_cast<float>(ncol);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < ncol; ++j) {
                out_f32[j] = weight_f32[j] * in_f32[j] / rms;
            }
            neollm::utils::fp32_to_bf16_batch(out_t, out_f32.data(), ncol);

        } else if constexpr (std::is_same_v<T, neollm::fp16_t>) {
            float rms{}, square_sum{}, avg_square_sum{};
            std::vector<float> in_f32(ncol);
            std::vector<float> weight_f32(ncol);
            std::vector<float> out_f32(ncol);

            // for (size_t j = 0; j < ncol; ++j) {
            //     in_f32[j] = neollm::utils::cast<float>(in_t[j]);
            //     weight_f32[j] = neollm::utils::cast<float>(weight[j]);
            // }
            neollm::utils::fp16_to_fp32_batch_f16c(in_f32.data(), in_t, ncol);
            neollm::utils::fp16_to_fp32_batch_f16c(weight_f32.data(), weight, ncol);

            square_sum = neollm::ops::cpu::sdot(in_f32.data(), in_f32.data(), ncol);
            avg_square_sum = square_sum / static_cast<float>(ncol);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < ncol; ++j) {
                out_f32[j] = weight_f32[j] * in_f32[j] / rms;
            }
            neollm::utils::fp32_to_fp16_batch_f16c(out_t, out_f32.data(), ncol);

        } else {
            T rms{}, square_sum{}, avg_square_sum{};
            square_sum = neollm::ops::cpu::sdot(in_t, in_t, ncol);
            avg_square_sum = square_sum / static_cast<T>(ncol);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < ncol; ++j) {
                out_t[j] = weight[j] * in_t[j] / rms;
            }
        }
    }
}

namespace neollm::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, NeollmDataType_t type, size_t nrow, size_t ncol) {
    switch (type) {
    case NEOLLM_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, nrow, ncol);
    case NEOLLM_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<neollm::bf16_t *>(out), reinterpret_cast<const neollm::bf16_t *>(in),
                         reinterpret_cast<const neollm::bf16_t *>(weight), eps, nrow, ncol);
    case NEOLLM_DTYPE_F16:
        return rms_norm_(reinterpret_cast<neollm::fp16_t *>(out), reinterpret_cast<const neollm::fp16_t *>(in),
                         reinterpret_cast<const neollm::fp16_t *>(weight), eps, nrow, ncol);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace neollm::ops::cpu
