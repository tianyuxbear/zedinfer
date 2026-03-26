#pragma once

#include "backend/ops/linear/cpu/quantized_common.hpp"

#include <cstddef>
#include <immintrin.h>

namespace zedinfer::ops::cpu::detail {

inline void vecmul_q8_soa(float *__restrict__ output, const float *__restrict__ input,
                          const int8_t *__restrict__ weight_q,
                          const float *__restrict__ bias,
                          const float *__restrict__ scales,
                          int n, int k, int group_size) {
    const int num_groups = k / group_size;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        const int8_t *weight_row = weight_q + static_cast<std::size_t>(row) * k;
        const float *scale_row = scales + static_cast<std::size_t>(row) * num_groups;

        float output_value = 0.0f;
        for (int group = 0; group < num_groups; ++group) {
            const int k_start = group * group_size;
            const int k_end = k_start + group_size;
            float group_dot = 0.0f;
            int col = k_start;

#if defined(__AVX512F__)
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            for (; col + 63 < k_end; col += 64) {
                const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight_row + col));
                const __m128i w1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight_row + col + 16));
                const __m128i w2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight_row + col + 32));
                const __m128i w3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight_row + col + 48));

                sum0 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w0)),
                    _mm512_loadu_ps(input + col), sum0);
                sum1 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w1)),
                    _mm512_loadu_ps(input + col + 16), sum1);
                sum2 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w2)),
                    _mm512_loadu_ps(input + col + 32), sum2);
                sum3 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w3)),
                    _mm512_loadu_ps(input + col + 48), sum3);
            }

            for (; col + 15 < k_end; col += 16) {
                const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight_row + col));
                sum0 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w0)),
                    _mm512_loadu_ps(input + col), sum0);
            }

            const __m512 total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
            group_dot = hsum_avx512(total);

#elif defined(__AVX2__)
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            for (; col + 31 < k_end; col += 32) {
                const __m128i w0 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(weight_row + col));
                const __m128i w1 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(weight_row + col + 8));
                const __m128i w2 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(weight_row + col + 16));
                const __m128i w3 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(weight_row + col + 24));

                sum0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)),
                    _mm256_loadu_ps(input + col), sum0);
                sum1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)),
                    _mm256_loadu_ps(input + col + 8), sum1);
                sum2 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w2)),
                    _mm256_loadu_ps(input + col + 16), sum2);
                sum3 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w3)),
                    _mm256_loadu_ps(input + col + 24), sum3);
            }

            const __m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            group_dot = hsum_avx256(total);
#endif

            for (; col < k_end; ++col) {
                group_dot += input[col] * static_cast<float>(weight_row[col]);
            }

            output_value += scale_row[group] * group_dot;
        }

        if (bias) {
            output_value += bias[row];
        }
        output[row] = output_value;
    }
}

inline void vecmul_q4_soa(float *__restrict__ output, const float *__restrict__ input,
                          const int32_t *__restrict__ weight_packed,
                          const float *__restrict__ bias,
                          const float *__restrict__ scales,
                          int n, int k, int group_size) {
    const int num_groups = k / group_size;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        const int32_t *weight_row =
            weight_packed + static_cast<std::size_t>(row) * (k / kInt4PackSize);
        const float *scale_row = scales + static_cast<std::size_t>(row) * num_groups;

        float output_value = 0.0f;
        for (int group = 0; group < num_groups; ++group) {
            const int k_start = group * group_size;
            const int group_len = group_size;
            const int packed_offset = k_start / kInt4PackSize;

            const int32_t *packed_group = weight_row + packed_offset;
            const float *input_group = input + k_start;

            float dot_product = 0.0f;
            float sum_input = 0.0f;
            int offset = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
            __m512 dot_acc = _mm512_setzero_ps();
            __m512 input_acc = _mm512_setzero_ps();

            for (; offset + 63 < group_len; offset += 64) {
                const __m256i packed = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(packed_group + offset / kInt4PackSize));
                const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed);
                const __m512i lo = _mm512_and_si512(packed_u16, _mm512_set1_epi16(0x000F));
                const __m512i hi = _mm512_and_si512(
                    _mm512_srli_epi16(packed_u16, 4), _mm512_set1_epi16(0x000F));
                const __m512i unpacked = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));

                const __m128i w0 = _mm512_castsi512_si128(unpacked);
                const __m128i w1 = _mm512_extracti32x4_epi32(unpacked, 1);
                const __m128i w2 = _mm512_extracti32x4_epi32(unpacked, 2);
                const __m128i w3 = _mm512_extracti32x4_epi32(unpacked, 3);

                const __m512 in0 = _mm512_loadu_ps(input_group + offset);
                const __m512 in1 = _mm512_loadu_ps(input_group + offset + 16);
                const __m512 in2 = _mm512_loadu_ps(input_group + offset + 32);
                const __m512 in3 = _mm512_loadu_ps(input_group + offset + 48);

                dot_acc = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(w0)), in0, dot_acc);
                dot_acc = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(w1)), in1, dot_acc);
                dot_acc = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(w2)), in2, dot_acc);
                dot_acc = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(w3)), in3, dot_acc);

                input_acc = _mm512_add_ps(input_acc, in0);
                input_acc = _mm512_add_ps(input_acc, in1);
                input_acc = _mm512_add_ps(input_acc, in2);
                input_acc = _mm512_add_ps(input_acc, in3);
            }

            dot_product = hsum_avx512(dot_acc);
            sum_input = hsum_avx512(input_acc);

#elif defined(__AVX2__)
            __m256 dot_acc = _mm256_setzero_ps();
            __m256 input_acc = _mm256_setzero_ps();

            for (; offset + 31 < group_len; offset += 32) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(packed_group + offset / kInt4PackSize));
                const __m128i mask = _mm_set1_epi8(0x0F);
                const __m128i lo = _mm_and_si128(packed, mask);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);

                const __m128i w0 = _mm_unpacklo_epi8(lo, hi);
                const __m128i w1 = _mm_unpackhi_epi8(lo, hi);

                __m256 weights = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(w0));
                __m256 inputs = _mm256_loadu_ps(input_group + offset);
                dot_acc = _mm256_fmadd_ps(weights, inputs, dot_acc);
                input_acc = _mm256_add_ps(input_acc, inputs);

                weights = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(w0, 8)));
                inputs = _mm256_loadu_ps(input_group + offset + 8);
                dot_acc = _mm256_fmadd_ps(weights, inputs, dot_acc);
                input_acc = _mm256_add_ps(input_acc, inputs);

                weights = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(w1));
                inputs = _mm256_loadu_ps(input_group + offset + 16);
                dot_acc = _mm256_fmadd_ps(weights, inputs, dot_acc);
                input_acc = _mm256_add_ps(input_acc, inputs);

                weights = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(w1, 8)));
                inputs = _mm256_loadu_ps(input_group + offset + 24);
                dot_acc = _mm256_fmadd_ps(weights, inputs, dot_acc);
                input_acc = _mm256_add_ps(input_acc, inputs);
            }

            dot_product = hsum_avx256(dot_acc);
            sum_input = hsum_avx256(input_acc);
#endif

            for (; offset < group_len; ++offset) {
                dot_product += input_group[offset] *
                               static_cast<float>(unpack_int4_unsigned(weight_row, k_start + offset));
                sum_input += input_group[offset];
            }

            output_value += scale_row[group] * (dot_product - kInt4ZeroPoint * sum_input);
        }

        if (bias) {
            output_value += bias[row];
        }
        output[row] = output_value;
    }
}

} // namespace zedinfer::ops::cpu::detail
