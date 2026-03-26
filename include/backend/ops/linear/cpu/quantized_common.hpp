#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// Blocking parameters tuned for AVX-512 on typical server CPUs.
#ifndef GEMM_MR
#define GEMM_MR 14
#endif
#ifndef GEMM_NR
#define GEMM_NR 32
#endif
#ifndef GEMM_NTHREADS
#define GEMM_NTHREADS 24
#endif
#ifndef GEMM_MC
#define GEMM_MC (GEMM_MR * GEMM_NTHREADS * 5)
#endif
#ifndef GEMM_NC
#define GEMM_NC (GEMM_NR * GEMM_NTHREADS * 30)
#endif
#ifndef GEMM_KC
#define GEMM_KC 512
#endif

namespace zedinfer::ops::cpu::detail {

inline constexpr int kGemmMr = GEMM_MR;
inline constexpr int kGemmNr = GEMM_NR;
inline constexpr int kGemmNumThreads = GEMM_NTHREADS;
inline constexpr int kGemmMc = GEMM_MC;
inline constexpr int kGemmNc = GEMM_NC;
inline constexpr int kGemmKc = GEMM_KC;

inline constexpr int kInt4PackSize = 8;
inline constexpr float kInt4ZeroPoint = 8.0f;
inline constexpr std::size_t kCacheAlignment = 64;

alignas(kCacheAlignment) inline float g_block_a_packed[kGemmMc * kGemmKc];
alignas(kCacheAlignment) inline float g_block_b_packed[kGemmNc * kGemmKc];

inline int unpack_int4_unsigned(const int32_t *packed_row, int k) {
    const int packed_idx = k / kInt4PackSize;
    const int shift = (k % kInt4PackSize) * 4;
    return (packed_row[packed_idx] >> shift) & 0xF;
}

inline float dequantize_int4_weight(const int32_t *packed_row, int k) {
    return static_cast<float>(unpack_int4_unsigned(packed_row, k)) - kInt4ZeroPoint;
}

inline float hsum_avx256(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    low = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(low);
    __m128 sums = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

#if defined(__AVX512F__)
inline float hsum_avx512(__m512 value) {
    return _mm512_reduce_add_ps(value);
}
#endif

} // namespace zedinfer::ops::cpu::detail
