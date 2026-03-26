#include "backend/ops/rms_norm/cpu/sdot.hpp"

#include <cstddef>
#include <immintrin.h>

float sdot(const float* x, const float* y, const size_t n) {
#if defined(__AVX512F__)
    // AVX-512: 16 floats per iteration
    __m512 accum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        accum = _mm512_fmadd_ps(vx, vy, accum);
    }
    float sum = _mm512_reduce_add_ps(accum);
    for (; i < n; ++i) { sum += x[i] * y[i]; }
    return sum;

#elif defined(__AVX2__) && defined(__FMA__)
    // AVX2 + FMA: 8 floats per iteration, 4 accumulators
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), acc3);
    }
    for (; i + 7 < n; i += 8) { acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0); }
    // Reduce 4 accumulators -> 1
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    // Horizontal sum of 8 floats in __m256
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);
    for (; i < n; ++i) { sum += x[i] * y[i]; }
    return sum;

#else
    // Scalar fallback
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { sum += x[i] * y[i]; }
    return sum;
#endif
}
