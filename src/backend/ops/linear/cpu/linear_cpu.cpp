#include "backend/ops/linear/cpu/linear_cpu.hpp"
#include "backend/ops/linear/cpu/matmul_quantized.hpp"
#include "backend/ops/linear/cpu/vecmul_quantized.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#ifdef USE_ONEDNN
#include "backend/ops/linear/cpu/linear_onednn.hpp"
#else
#include "backend/ops/linear/cpu/matmul.hpp"
#include "backend/ops/linear/cpu/vecmul.hpp"
#include <cstring>
#include <vector>
#endif

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifndef USE_ONEDNN

// Handwritten fallback (used when oneDNN is not available)
template <typename T>
static void linear_fallback(T* output, const T* input, const T* weight, const T* bias, size_t M, size_t N, size_t K) {
    if constexpr (std::is_same_v<T, float>) {
        if (bias) {
            for (size_t i = 0; i < M; ++i) { std::memcpy(output + i * N, bias, N * sizeof(T)); }
        } else {
            std::memset(output, 0, M * N * sizeof(T));
        }
        if (M == 1) {
            vecmul(input, weight, output, N, K);
        } else {
            matmul(input, weight, output, M, N, K);
        }
    } else if constexpr (std::is_same_v<T, zedinfer::fp16_t>) {
        std::vector<float> input_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> output_fp32(M * N, 0.0f);

        zedinfer::utils::fp16_to_fp32_batch_f16c(input_fp32.data(), input, M * K);
        zedinfer::utils::fp16_to_fp32_batch_f16c(weight_fp32.data(), weight, N * K);

        if (bias) {
            std::vector<float> bias_fp32(N);
            zedinfer::utils::fp16_to_fp32_batch_f16c(bias_fp32.data(), bias, N);
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(output_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }
        }

        if (M == 1) {
            vecmul(input_fp32.data(), weight_fp32.data(), output_fp32.data(), N, K);
        } else {
            matmul(input_fp32.data(), weight_fp32.data(), output_fp32.data(), M, N, K);
        }

        zedinfer::utils::fp32_to_fp16_batch_f16c(output, output_fp32.data(), M * N);
    } else if constexpr (std::is_same_v<T, zedinfer::bf16_t>) {
        std::vector<float> input_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> output_fp32(M * N, 0.0f);

        zedinfer::utils::bf16_to_fp32_batch(input_fp32.data(), input, M * K);
        zedinfer::utils::bf16_to_fp32_batch(weight_fp32.data(), weight, N * K);

        if (bias) {
            std::vector<float> bias_fp32(N);
            zedinfer::utils::bf16_to_fp32_batch(bias_fp32.data(), bias, N);
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(output_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }
        }

        if (M == 1) {
            vecmul(input_fp32.data(), weight_fp32.data(), output_fp32.data(), N, K);
        } else {
            matmul(input_fp32.data(), weight_fp32.data(), output_fp32.data(), M, N, K);
        }

        zedinfer::utils::fp32_to_bf16_batch(output, output_fp32.data(), M * N);
    }
}

#endif // !USE_ONEDNN

template <typename T>
static void linear_quantized_impl(
    T *output,
    const T *input,
    const void *weight,
    const T *bias,
    const T *scales,
    int num_bits,
    int group_size,
    size_t M,
    size_t N,
    size_t K) {
    group_size = (group_size > 0) ? group_size : static_cast<int>(K);
    const int num_groups = static_cast<int>(K) / group_size;

    if constexpr (std::is_same_v<T, float>) {
        if (M == 1) {
            if (num_bits == 8) {
                zedinfer::ops::cpu::detail::vecmul_q8_soa(
                    output, input,
                    reinterpret_cast<const int8_t *>(weight),
                    bias, scales, static_cast<int>(N),
                    static_cast<int>(K), group_size);
            } else {
                zedinfer::ops::cpu::detail::vecmul_q4_soa(
                    output, input,
                    reinterpret_cast<const int32_t *>(weight),
                    bias, scales, static_cast<int>(N),
                    static_cast<int>(K), group_size);
            }
            return;
        }

        std::memset(output, 0, M * N * sizeof(float));
        zedinfer::ops::cpu::detail::matmul_packed_unified(
            input, weight, output,
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            scales, num_bits, group_size);

        if (!bias) {
            return;
        }

        #pragma omp parallel for collapse(2)
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                output[m * N + n] += bias[n];
            }
        }
    } else if constexpr (std::is_same_v<T, zedinfer::bf16_t>) {
        std::vector<float> input_fp32(M * K);
        std::vector<float> scales_fp32(N * static_cast<size_t>(num_groups));
        std::vector<float> output_fp32(M * N);
        std::vector<float> bias_fp32;
        const float *bias_ptr = nullptr;

        zedinfer::utils::bf16_to_fp32_batch(input_fp32.data(), input, M * K);
        zedinfer::utils::bf16_to_fp32_batch(
            scales_fp32.data(), scales, N * static_cast<size_t>(num_groups));

        if (bias) {
            bias_fp32.resize(N);
            zedinfer::utils::bf16_to_fp32_batch(bias_fp32.data(), bias, N);
            bias_ptr = bias_fp32.data();
        }

        linear_quantized_impl(
            output_fp32.data(), input_fp32.data(), weight,
            bias_ptr, scales_fp32.data(), num_bits, group_size, M, N, K);
        zedinfer::utils::fp32_to_bf16_batch(output, output_fp32.data(), M * N);
    } else if constexpr (std::is_same_v<T, zedinfer::fp16_t>) {
        std::vector<float> input_fp32(M * K);
        std::vector<float> scales_fp32(N * static_cast<size_t>(num_groups));
        std::vector<float> output_fp32(M * N);
        std::vector<float> bias_fp32;
        const float *bias_ptr = nullptr;

        zedinfer::utils::fp16_to_fp32_batch_f16c(input_fp32.data(), input, M * K);
        zedinfer::utils::fp16_to_fp32_batch_f16c(
            scales_fp32.data(), scales, N * static_cast<size_t>(num_groups));

        if (bias) {
            bias_fp32.resize(N);
            zedinfer::utils::fp16_to_fp32_batch_f16c(bias_fp32.data(), bias, N);
            bias_ptr = bias_fp32.data();
        }

        linear_quantized_impl(
            output_fp32.data(), input_fp32.data(), weight,
            bias_ptr, scales_fp32.data(), num_bits, group_size, M, N, K);
        zedinfer::utils::fp32_to_fp16_batch_f16c(output, output_fp32.data(), M * N);
    }
}

namespace zedinfer::ops::cpu {

void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K) {
#ifdef USE_ONEDNN
    // oneDNN handles all dtypes (FP32, BF16, FP16) with automatic ISA dispatch.
    // No manual type conversion needed.
    onednn::linear(output, input, weight, bias, type, M, N, K);
#else
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return linear_fallback(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input),
                                   reinterpret_cast<const float*>(weight), reinterpret_cast<const float*>(bias), M, N,
                                   K);
        case ZEDINFER_DTYPE_BF16:
            return linear_fallback(reinterpret_cast<zedinfer::bf16_t*>(output),
                                   reinterpret_cast<const zedinfer::bf16_t*>(input),
                                   reinterpret_cast<const zedinfer::bf16_t*>(weight),
                                   reinterpret_cast<const zedinfer::bf16_t*>(bias), M, N, K);
        case ZEDINFER_DTYPE_F16:
            return linear_fallback(reinterpret_cast<zedinfer::fp16_t*>(output),
                                   reinterpret_cast<const zedinfer::fp16_t*>(input),
                                   reinterpret_cast<const zedinfer::fp16_t*>(weight),
                                   reinterpret_cast<const zedinfer::fp16_t*>(bias), M, N, K);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
#endif
}

void linear_quantized(std::byte *output, const std::byte *input, const std::byte *weight_packed, const std::byte *bias, const std::byte *scale, const std::byte *g_idx, zedinferDataType_t type, int num_bits, int group_size, size_t M, size_t N, size_t K) {
    (void)g_idx;

    if (num_bits != 4 && num_bits != 8) {
        throw std::runtime_error(
            "linear_quantized CPU: only 4-bit and 8-bit weights are supported");
    }

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return linear_quantized_impl(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(input),
            weight_packed,
            reinterpret_cast<const float *>(bias),
            reinterpret_cast<const float *>(scale),
            num_bits, group_size, M, N, K);
    case ZEDINFER_DTYPE_BF16:
        return linear_quantized_impl(
            reinterpret_cast<zedinfer::bf16_t *>(output),
            reinterpret_cast<const zedinfer::bf16_t *>(input),
            weight_packed,
            reinterpret_cast<const zedinfer::bf16_t *>(bias),
            reinterpret_cast<const zedinfer::bf16_t *>(scale),
            num_bits, group_size, M, N, K);
    case ZEDINFER_DTYPE_F16:
        return linear_quantized_impl(
            reinterpret_cast<zedinfer::fp16_t *>(output),
            reinterpret_cast<const zedinfer::fp16_t *>(input),
            weight_packed,
            reinterpret_cast<const zedinfer::fp16_t *>(bias),
            reinterpret_cast<const zedinfer::fp16_t *>(scale),
            num_bits, group_size, M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
