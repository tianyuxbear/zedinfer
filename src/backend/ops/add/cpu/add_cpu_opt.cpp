#include "backend/ops/add/cpu/add_cpu_opt.hpp"

#include <immintrin.h>
#include <omp.h>

#ifndef SIMD_THRESHOLD
#define SIMD_THRESHOLD 64
#endif

#ifndef PARALLEL_THRESHOLD
#define PARALLEL_THRESHOLD 16384
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

// ============================================================================
// Tiny scalar path (< SIMD_THRESHOLD elements, all dtypes)
// ============================================================================

template <typename T> FORCE_INLINE void scalar_add_tiny(T* c, const T* a, const T* b, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float f_a = zedinfer::utils::cast<float>(a[i]);
        float f_b = zedinfer::utils::cast<float>(b[i]);
        c[i] = zedinfer::utils::cast<T>(f_a + f_b);
    }
}

// ============================================================================
// BF16 add
// ============================================================================

// AVX2 BF16 add: process 8 elements per iteration
// BF16 -> shift left 16 to get FP32 -> add -> shift right 16 to get BF16
#if defined(__AVX2__) && !defined(__AVX512F__)
static void add_bf16_avx2_loop(zedinfer::bf16_t* c, const zedinfer::bf16_t* a, const zedinfer::bf16_t* b, size_t start,
                               size_t end) {
    constexpr size_t vec_size = 8;
    size_t i = start;
    size_t aligned_end = start + ((end - start) / vec_size) * vec_size;
    for (; i < aligned_end; i += vec_size) {
        __m128i a_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i b_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m256i a_int = _mm256_cvtepu16_epi32(a_raw);
        __m256i b_int = _mm256_cvtepu16_epi32(b_raw);
        __m256 a_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(a_int, 16));
        __m256 b_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(b_int, 16));
        __m256 c_f32 = _mm256_add_ps(a_f32, b_f32);
        // Pack back: shift right 16, then truncate 32->16 via shuffle+pack
        __m256i c_int = _mm256_srli_epi32(_mm256_castps_si256(c_f32), 16);
        // Extract low and high 128-bit lanes, pack 32->16
        __m128i lo = _mm256_castsi256_si128(c_int);
        __m128i hi = _mm256_extracti128_si256(c_int, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i), packed);
    }
    for (; i < end; ++i) {
        float f_a = zedinfer::utils::cast<float>(a[i]);
        float f_b = zedinfer::utils::cast<float>(b[i]);
        c[i] = zedinfer::utils::cast<zedinfer::bf16_t>(f_a + f_b);
    }
}
#endif

#if defined(__AVX512F__)
static void add_bf16_avx512_loop(zedinfer::bf16_t* c, const zedinfer::bf16_t* a, const zedinfer::bf16_t* b,
                                 size_t start, size_t end) {
    constexpr size_t vec_size = 16;
    size_t i = start;
    size_t aligned_end = start + ((end - start) / vec_size) * vec_size;
    for (; i < aligned_end; i += vec_size) {
        __m256i a_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i b_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m512i a_int = _mm512_cvtepu16_epi32(a_raw);
        __m512i b_int = _mm512_cvtepu16_epi32(b_raw);
        __m512 a_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(a_int, 16));
        __m512 b_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(b_int, 16));
        __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);
        __m256i c_vec = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(c_f32), 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), c_vec);
    }
    for (; i < end; ++i) {
        float f_a = zedinfer::utils::cast<float>(a[i]);
        float f_b = zedinfer::utils::cast<float>(b[i]);
        c[i] = zedinfer::utils::cast<zedinfer::bf16_t>(f_a + f_b);
    }
}
#endif

void add_bf16(zedinfer::bf16_t* c, const zedinfer::bf16_t* a, const zedinfer::bf16_t* b, size_t numel) {
    if (numel <= SIMD_THRESHOLD) {
        scalar_add_tiny(c, a, b, numel);
        return;
    }

#if defined(__AVX512F__)
    auto loop = add_bf16_avx512_loop;
#elif defined(__AVX2__)
    auto loop = add_bf16_avx2_loop;
#else
    auto loop = [](zedinfer::bf16_t* c, const zedinfer::bf16_t* a, const zedinfer::bf16_t* b, size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) {
            c[i] = zedinfer::utils::cast<zedinfer::bf16_t>(zedinfer::utils::cast<float>(a[i])
                                                           + zedinfer::utils::cast<float>(b[i]));
        }
    };
#endif

    if (numel < PARALLEL_THRESHOLD) {
        loop(c, a, b, 0, numel);
    } else {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            size_t chunk = numel / nt;
            size_t start = tid * chunk;
            size_t end = (tid == nt - 1) ? numel : start + chunk;
            loop(c, a, b, start, end);
        }
    }
}

// ============================================================================
// FP16 add
// ============================================================================

#if defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)
static void add_f16_avx2_loop(zedinfer::fp16_t* c, const zedinfer::fp16_t* a, const zedinfer::fp16_t* b, size_t start,
                              size_t end) {
    constexpr size_t vec_size = 8;
    size_t i = start;
    size_t aligned_end = start + ((end - start) / vec_size) * vec_size;
    for (; i < aligned_end; i += vec_size) {
        __m128i a_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i b_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m256 a_f32 = _mm256_cvtph_ps(a_raw);
        __m256 b_f32 = _mm256_cvtph_ps(b_raw);
        __m256 c_f32 = _mm256_add_ps(a_f32, b_f32);
        __m128i c_raw = _mm256_cvtps_ph(c_f32, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i), c_raw);
    }
    for (; i < end; ++i) {
        c[i] = zedinfer::utils::cast<zedinfer::fp16_t>(zedinfer::utils::cast<float>(a[i])
                                                       + zedinfer::utils::cast<float>(b[i]));
    }
}
#endif

#if defined(__AVX512F__) && defined(__F16C__)
static void add_f16_avx512_loop(zedinfer::fp16_t* c, const zedinfer::fp16_t* a, const zedinfer::fp16_t* b, size_t start,
                                size_t end) {
    constexpr size_t vec_size = 16;
    size_t i = start;
    size_t aligned_end = start + ((end - start) / vec_size) * vec_size;
    for (; i < aligned_end; i += vec_size) {
        __m256i a_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i b_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m512 a_f32 = _mm512_cvtph_ps(a_raw);
        __m512 b_f32 = _mm512_cvtph_ps(b_raw);
        __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);
        __m256i c_raw = _mm512_cvtps_ph(c_f32, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), c_raw);
    }
    for (; i < end; ++i) {
        c[i] = zedinfer::utils::cast<zedinfer::fp16_t>(zedinfer::utils::cast<float>(a[i])
                                                       + zedinfer::utils::cast<float>(b[i]));
    }
}
#endif

void add_f16(zedinfer::fp16_t* c, const zedinfer::fp16_t* a, const zedinfer::fp16_t* b, size_t numel) {
    if (numel <= SIMD_THRESHOLD) {
        scalar_add_tiny(c, a, b, numel);
        return;
    }

#if defined(__AVX512F__) && defined(__F16C__)
    auto loop = add_f16_avx512_loop;
#elif defined(__AVX2__) && defined(__F16C__)
    auto loop = add_f16_avx2_loop;
#else
    auto loop = [](zedinfer::fp16_t* c, const zedinfer::fp16_t* a, const zedinfer::fp16_t* b, size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) {
            c[i] = zedinfer::utils::cast<zedinfer::fp16_t>(zedinfer::utils::cast<float>(a[i])
                                                           + zedinfer::utils::cast<float>(b[i]));
        }
    };
#endif

    if (numel < PARALLEL_THRESHOLD) {
        loop(c, a, b, 0, numel);
    } else {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            size_t chunk = numel / nt;
            size_t start = tid * chunk;
            size_t end = (tid == nt - 1) ? numel : start + chunk;
            loop(c, a, b, start, end);
        }
    }
}

// ============================================================================
// FP32 add
// ============================================================================

#if defined(__AVX2__) && !defined(__AVX512F__)
static void add_f32_avx2_loop(float* c, const float* a, const float* b, size_t start, size_t end) {
    constexpr size_t vec_size = 8;
    size_t i = start;
    size_t aligned_end = start + ((end - start) / vec_size) * vec_size;
    for (; i + 15 < aligned_end; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va0, vb0));
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        _mm256_storeu_ps(c + i + 8, _mm256_add_ps(va1, vb1));
    }
    for (; i < aligned_end; i += vec_size) {
        _mm256_storeu_ps(c + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < end; ++i) { c[i] = a[i] + b[i]; }
}
#endif

#if defined(__AVX512F__)
static void add_f32_avx512_loop(float* c, const float* a, const float* b, size_t start, size_t end) {
    constexpr size_t vec_size = 16;
    size_t i = start;
    size_t aligned_end = start + ((end - start) / vec_size) * vec_size;
    for (; i + 31 < aligned_end; i += 32) {
        _mm512_storeu_ps(c + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
        _mm512_storeu_ps(c + i + 16, _mm512_add_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16)));
    }
    for (; i < aligned_end; i += vec_size) {
        _mm512_storeu_ps(c + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    for (; i < end; ++i) { c[i] = a[i] + b[i]; }
}
#endif

void add_f32(float* c, const float* a, const float* b, size_t numel) {
    if (numel <= SIMD_THRESHOLD) {
        for (size_t i = 0; i < numel; ++i) { c[i] = a[i] + b[i]; }
        return;
    }

#if defined(__AVX512F__)
    auto loop = add_f32_avx512_loop;
#elif defined(__AVX2__)
    auto loop = add_f32_avx2_loop;
#else
    auto loop = [](float* c, const float* a, const float* b, size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) { c[i] = a[i] + b[i]; }
    };
#endif

    if (numel < PARALLEL_THRESHOLD) {
        loop(c, a, b, 0, numel);
    } else {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            size_t chunk = numel / nt;
            size_t start = tid * chunk;
            size_t end = (tid == nt - 1) ? numel : start + chunk;
            loop(c, a, b, start, end);
        }
    }
}
