#pragma once

#include "backend/ops/linear/cpu/quantized_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <omp.h>

namespace zedinfer::ops::cpu::detail {

inline void pack_panel_a(const float* a_panel, float* dst, int mr, int kc, int k) {
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i) { *dst++ = a_panel[static_cast<std::size_t>(i) * k + p]; }
        for (int i = mr; i < kGemmMr; ++i) { *dst++ = 0.0f; }
    }
}

inline void pack_block_a(const float* a_block, float* dst, int mc, int kc, int k) {
#pragma omp parallel for num_threads(kGemmNumThreads)
    for (int i = 0; i < mc; i += kGemmMr) {
        const int mr = std::min(kGemmMr, mc - i);
        pack_panel_a(a_block + static_cast<std::size_t>(i) * k, dst + static_cast<std::size_t>(i) * kc, mr, kc, k);
    }
}

template <typename LoadWeightFn>
inline void pack_panel_b(float* dst, const float* scales, int nr, int kc, int group_size, int num_groups, int n_offset,
                         int k_offset, LoadWeightFn load_weight) {
    for (int p = 0; p < kc; ++p) {
        const int k_global = k_offset + p;
        const int group_idx = k_global / group_size;

        for (int j = 0; j < nr; ++j) {
            const int n_global = n_offset + j;
            const float scale = scales[static_cast<std::size_t>(n_global) * num_groups + group_idx];
            *dst++ = load_weight(n_global, k_global) * scale;
        }
        for (int j = nr; j < kGemmNr; ++j) { *dst++ = 0.0f; }
    }
}

inline void pack_block_b_q8(const int8_t* weight, float* dst, const float* scales, int nc, int kc, int k,
                            int group_size, int num_groups, int n_offset, int k_offset) {
#pragma omp parallel for num_threads(kGemmNumThreads)
    for (int j = 0; j < nc; j += kGemmNr) {
        const int nr = std::min(kGemmNr, nc - j);
        pack_panel_b(dst + static_cast<std::size_t>(j) * kc, scales, nr, kc, group_size, num_groups, n_offset + j,
                     k_offset, [&](int n_global, int k_global) {
                         return static_cast<float>(weight[static_cast<std::size_t>(n_global) * k + k_global]);
                     });
    }
}

inline void pack_block_b_q4(const int32_t* weight_packed, float* dst, const float* scales, int nc, int kc, int k,
                            int group_size, int num_groups, int n_offset, int k_offset) {
#pragma omp parallel for num_threads(kGemmNumThreads)
    for (int j = 0; j < nc; j += kGemmNr) {
        const int nr = std::min(kGemmNr, nc - j);
        pack_panel_b(dst + static_cast<std::size_t>(j) * kc, scales, nr, kc, group_size, num_groups, n_offset + j,
                     k_offset, [&](int n_global, int k_global) {
                         const int32_t* row = weight_packed + static_cast<std::size_t>(n_global) * (k / kInt4PackSize);
                         return dequantize_int4_weight(row, k_global);
                     });
    }
}

#if defined(__AVX512F__)

inline __mmask16 gemm_create_mask(int nr) {
    nr = std::clamp(nr, 0, 16);
    return _cvtu32_mask16((1u << nr) - 1);
}

inline void gemm_fma_loop(const float* a_packed, const float* b_packed, __m512 accum[kGemmMr][2], int kc) {
    for (int p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_loadu_ps(b_packed);
        const __m512 b1 = _mm512_loadu_ps(b_packed + 16);

#define GEMM_FMA_ROW(i)                                                                                                \
    {                                                                                                                  \
        const __m512 a = _mm512_set1_ps(a_packed[i]);                                                                  \
        accum[i][0] = _mm512_fmadd_ps(a, b0, accum[i][0]);                                                             \
        accum[i][1] = _mm512_fmadd_ps(a, b1, accum[i][1]);                                                             \
    }

        GEMM_FMA_ROW(0)
        GEMM_FMA_ROW(1)
        GEMM_FMA_ROW(2)
        GEMM_FMA_ROW(3)
        GEMM_FMA_ROW(4)
        GEMM_FMA_ROW(5)
        GEMM_FMA_ROW(6)
        GEMM_FMA_ROW(7)
        GEMM_FMA_ROW(8)
        GEMM_FMA_ROW(9)
        GEMM_FMA_ROW(10)
        GEMM_FMA_ROW(11)
        GEMM_FMA_ROW(12)
        GEMM_FMA_ROW(13)

#undef GEMM_FMA_ROW

        a_packed += kGemmMr;
        b_packed += kGemmNr;
    }
}

inline void gemm_micro_kernel(const float* a_packed, const float* b_packed, float* c, int mr, int nr, int kc, int n) {
    __m512 accum[kGemmMr][2];

    if (nr == kGemmNr) {
        for (int i = 0; i < mr; ++i) {
            accum[i][0] = _mm512_loadu_ps(&c[static_cast<std::size_t>(i) * n]);
            accum[i][1] = _mm512_loadu_ps(&c[static_cast<std::size_t>(i) * n + 16]);
        }

        gemm_fma_loop(a_packed, b_packed, accum, kc);

        for (int i = 0; i < mr; ++i) {
            _mm512_storeu_ps(&c[static_cast<std::size_t>(i) * n], accum[i][0]);
            _mm512_storeu_ps(&c[static_cast<std::size_t>(i) * n + 16], accum[i][1]);
        }
        return;
    }

    const __mmask16 mask0 = gemm_create_mask(nr);
    const __mmask16 mask1 = gemm_create_mask(nr - 16);
    for (int i = 0; i < mr; ++i) {
        accum[i][0] = _mm512_maskz_loadu_ps(mask0, &c[static_cast<std::size_t>(i) * n]);
        accum[i][1] = _mm512_maskz_loadu_ps(mask1, &c[static_cast<std::size_t>(i) * n + 16]);
    }

    gemm_fma_loop(a_packed, b_packed, accum, kc);

    for (int i = 0; i < mr; ++i) {
        _mm512_mask_storeu_ps(&c[static_cast<std::size_t>(i) * n], mask0, accum[i][0]);
        _mm512_mask_storeu_ps(&c[static_cast<std::size_t>(i) * n + 16], mask1, accum[i][1]);
    }
}

#else

inline void gemm_micro_kernel(const float* a_packed, const float* b_packed, float* c, int mr, int nr, int kc, int n) {
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i) {
            const float a_value = a_packed[static_cast<std::size_t>(p) * kGemmMr + i];
            for (int j = 0; j < nr; ++j) {
                c[static_cast<std::size_t>(i) * n + j] += a_value * b_packed[static_cast<std::size_t>(p) * kGemmNr + j];
            }
        }
    }
}

#endif

inline void matmul_packed_unified(const float* a, const void* weight, float* c, int m, int n, int k,
                                  const float* scales, int num_bits, int group_size) {
    const int num_groups = k / group_size;

    for (int j = 0; j < n; j += kGemmNc) {
        const int nc = std::min(kGemmNc, n - j);

        for (int p = 0; p < k; p += kGemmKc) {
            const int kc = std::min(kGemmKc, k - p);

            if (num_bits == 8) {
                pack_block_b_q8(reinterpret_cast<const int8_t*>(weight), g_block_b_packed, scales, nc, kc, k,
                                group_size, num_groups, j, p);
            } else {
                pack_block_b_q4(reinterpret_cast<const int32_t*>(weight), g_block_b_packed, scales, nc, kc, k,
                                group_size, num_groups, j, p);
            }

            for (int i = 0; i < m; i += kGemmMc) {
                const int mc = std::min(kGemmMc, m - i);
                pack_block_a(a + static_cast<std::size_t>(i) * k + p, g_block_a_packed, mc, kc, k);

#pragma omp parallel for num_threads(kGemmNumThreads)
                for (int jr = 0; jr < nc; jr += kGemmNr) {
                    const int nr = std::min(kGemmNr, nc - jr);
                    for (int ir = 0; ir < mc; ir += kGemmMr) {
                        const int mr = std::min(kGemmMr, mc - ir);
                        gemm_micro_kernel(g_block_a_packed + static_cast<std::size_t>(kc) * ir,
                                          g_block_b_packed + static_cast<std::size_t>(kc) * jr,
                                          c + static_cast<std::size_t>(i + ir) * n + (j + jr), mr, nr, kc, n);
                    }
                }
            }
        }
    }
}

} // namespace zedinfer::ops::cpu::detail
