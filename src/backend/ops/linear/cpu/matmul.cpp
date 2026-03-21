#include "backend/ops/linear/cpu/matmul.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <omp.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define MR 14
#define NR 32
#define KC 512

// Query actual thread count at runtime, capped at a reasonable max
static int get_nthreads() {
    static int n = std::min(omp_get_max_threads(), 128);
    return n;
}

// Dynamically sized packing buffers — allocated once, reused across calls.
// Thread-safe: only used within OMP parallel regions with proper partitioning.
struct PackBuffers {
    std::unique_ptr<float[], decltype(&std::free)> A{nullptr, std::free};
    std::unique_ptr<float[], decltype(&std::free)> B{nullptr, std::free};
    size_t A_size = 0;
    size_t B_size = 0;

    void ensure(int mc, int nc, int kc) {
        size_t need_A = static_cast<size_t>(mc) * kc;
        size_t need_B = static_cast<size_t>(nc) * kc;
        if (need_A > A_size) {
            A.reset(static_cast<float *>(std::aligned_alloc(64, need_A * sizeof(float))));
            A_size = need_A;
        }
        if (need_B > B_size) {
            B.reset(static_cast<float *>(std::aligned_alloc(64, need_B * sizeof(float))));
            B_size = need_B;
        }
    }
};

static PackBuffers &get_pack_buffers() {
    static PackBuffers bufs;
    return bufs;
}

static inline __mmask16 create_mask(int nr) {
    nr = (nr < 0) ? 0 : (nr > 16) ? 16 : nr;
    return _cvtu32_mask16((1u << nr) - 1);
}

static inline void pack_panelA(const float *A, float *packed, int mr, int kc, int K) {
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i)
            *packed++ = A[i * K + p];
        for (int i = mr; i < MR; ++i)
            *packed++ = 0;
    }
}

static inline void pack_blockA(const float *A, float *packed, int mc, int kc, int K) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        pack_panelA(&A[i * K], &packed[i * kc], mr, kc, K);
    }
}

static inline void pack_panelB(const float *B, float *packed, int nr, int kc, int K) {
    for (int p = 0; p < kc; ++p) {
        for (int j = 0; j < nr; ++j)
            *packed++ = B[j * K + p];
        for (int j = nr; j < NR; ++j)
            *packed++ = 0;
    }
}

static inline void pack_blockB(const float *B, float *packed, int nc, int kc, int K) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < nc; j += NR) {
        int nr = std::min(NR, nc - j);
        pack_panelB(&B[j * K], &packed[j * kc], nr, kc, K);
    }
}

static inline void fma_loop(float *blockA_packed, float *blockB_packed,
                            __m512 C_accum[MR][2], int kc) {
    for (int p = 0; p < kc; ++p) {
        __m512 b0 = _mm512_loadu_ps(blockB_packed);
        __m512 b1 = _mm512_loadu_ps(blockB_packed + 16);

#define UNROLL_FMA(i)                                        \
    {                                                        \
        __m512 a = _mm512_set1_ps(blockA_packed[i]);         \
        C_accum[i][0] = _mm512_fmadd_ps(a, b0, C_accum[i][0]); \
        C_accum[i][1] = _mm512_fmadd_ps(a, b1, C_accum[i][1]); \
    }

        UNROLL_FMA(0) UNROLL_FMA(1) UNROLL_FMA(2) UNROLL_FMA(3)
        UNROLL_FMA(4) UNROLL_FMA(5) UNROLL_FMA(6) UNROLL_FMA(7)
        UNROLL_FMA(8) UNROLL_FMA(9) UNROLL_FMA(10) UNROLL_FMA(11)
        UNROLL_FMA(12) UNROLL_FMA(13)

#undef UNROLL_FMA

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

static inline void micro_kernel(float *blockA_packed, float *blockB_packed,
                                float *C, int mr, int nr, int kc, int N) {
    __m512 C_accum[MR][2];

    if (likely(nr == NR)) {
        for (int i = 0; i < mr; ++i) {
            C_accum[i][0] = _mm512_loadu_ps(&C[i * N]);
            C_accum[i][1] = _mm512_loadu_ps(&C[i * N + 16]);
        }
        fma_loop(blockA_packed, blockB_packed, C_accum, kc);
        for (int i = 0; i < mr; ++i) {
            _mm512_storeu_ps(&C[i * N], C_accum[i][0]);
            _mm512_storeu_ps(&C[i * N + 16], C_accum[i][1]);
        }
    } else {
        __mmask16 mask0 = create_mask(nr);
        __mmask16 mask1 = create_mask(nr - 16);
        for (int i = 0; i < mr; ++i) {
            C_accum[i][0] = _mm512_maskz_loadu_ps(mask0, &C[i * N]);
            C_accum[i][1] = _mm512_maskz_loadu_ps(mask1, &C[i * N + 16]);
        }
        fma_loop(blockA_packed, blockB_packed, C_accum, kc);
        for (int i = 0; i < mr; ++i) {
            _mm512_mask_storeu_ps(&C[i * N], mask0, C_accum[i][0]);
            _mm512_mask_storeu_ps(&C[i * N + 16], mask1, C_accum[i][1]);
        }
    }
}

// C = A * B^T + C, all row-major
// C: [M, N], A: [M, K], B: [N, K]
void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    const int nthreads = get_nthreads();
    const int MC = MR * nthreads * 5;
    const int NC = NR * nthreads * 30;

    auto &bufs = get_pack_buffers();
    bufs.ensure(MC, NC, KC);

    for (int j = 0; j < N; j += NC) {
        int nc = std::min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            int kc = std::min(KC, K - p);
            pack_blockB(&B[j * K + p], bufs.B.get(), nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                int mc = std::min(MC, M - i);
                pack_blockA(&A[i * K + p], bufs.A.get(), mc, kc, K);
#pragma omp parallel for schedule(static)
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);
                        micro_kernel(&bufs.A.get()[kc * ir], &bufs.B.get()[kc * jr],
                                     &C[(i + ir) * N + (j + jr)], mr, nr, kc, N);
                    }
                }
            }
        }
    }
}
