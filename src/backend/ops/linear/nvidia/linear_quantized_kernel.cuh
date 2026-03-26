#pragma once

#include "utils/nvidia/common.cuh"
#include "utils/nvidia/types.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace zedinfer::ops::nvidia {

namespace wmma = nvcuda::wmma;

namespace detail {

inline constexpr unsigned kFullWarpMask = 0xffffffffu;
inline constexpr int kWarpSize = 32;

__device__ __forceinline__ void unpack_row_int4x64(int8_t* __restrict__ dst, const uint8_t* __restrict__ src) {
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t packed = src4[i];
        uint32_t lo = packed & 0x0F0F0F0Fu;
        uint32_t hi = (packed >> 4) & 0x0F0F0F0Fu;
        uint32_t lo_signed = __vsub4(lo, 0x08080808u);
        uint32_t hi_signed = __vsub4(hi, 0x08080808u);

        dst4[i * 2] = __byte_perm(lo_signed, hi_signed, 0x5140u);
        dst4[i * 2 + 1] = __byte_perm(lo_signed, hi_signed, 0x7362u);
    }
}

__device__ __forceinline__ void unpack_row_int4x32(int8_t* __restrict__ dst, const uint8_t* __restrict__ src) {
    const uint32_t* src4 = reinterpret_cast<const uint32_t*>(src);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t packed = src4[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t byte = (packed >> (j * 8)) & 0xFF;
            dst[i * 8 + j * 2] = static_cast<int8_t>(byte & 0x0F) - 8;
            dst[i * 8 + j * 2 + 1] = static_cast<int8_t>(byte >> 4) - 8;
        }
    }
}

__device__ __forceinline__ int dp4a_q4_chunk(uint32_t packed_weight, int act_low, int act_high, int acc) {
    uint32_t low = packed_weight & 0x0F0F0F0F;
    uint32_t high = (packed_weight >> 4) & 0x0F0F0F0F;
    uint32_t low_signed = low ^ 0x08080808;
    uint32_t high_signed = high ^ 0x08080808;
    uint32_t low_bytes = low_signed | ((low_signed & 0x08080808) * 30);
    uint32_t high_bytes = high_signed | ((high_signed & 0x08080808) * 30);

    acc = __dp4a(static_cast<int>(__byte_perm(low_bytes, high_bytes, 0x5140)), act_low, acc);
    acc = __dp4a(static_cast<int>(__byte_perm(low_bytes, high_bytes, 0x7362)), act_high, acc);
    return acc;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared_sum, int tid, int num_threads) {
    const int lane_id = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    const int num_warps = (num_threads + kWarpSize - 1) / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane_id == 0) {
        shared_sum[warp_id] = value;
    }
    __syncthreads();

    if (tid < kWarpSize) {
        value = (tid < num_warps) ? shared_sum[tid] : 0.0f;
        value = warp_reduce_sum(value);
    }
    return value;
}

template <typename T> __device__ __forceinline__ float load_or_zero(const T* values, int index, int limit) {
    return (index < limit && values) ? to_float(values[index]) : 0.0f;
}

template <typename T>
__device__ __forceinline__ void store_matvec_output(T* output, const T* bias, size_t row_idx, float block_sum,
                                                    float act_scale, int limit) {
    const float bias_value = load_or_zero(bias, static_cast<int>(row_idx), limit);
    output[row_idx] = from_float<T>(block_sum * act_scale + bias_value);
}

__device__ __forceinline__ void cp_async_16b(void* shared_ptr, const void* global_ptr, bool valid) {
    const uint32_t shared_addr = __cvta_generic_to_shared(shared_ptr);
    const int copy_bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_addr), "l"(global_ptr),
                 "r"(copy_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group0() {
    asm("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group1() {
    asm("cp.async.wait_group 1;\n" ::);
}

} // namespace detail

template <typename T>
__global__ void matvec_q4_0_q8_row_soa_kernel(T* __restrict__ out, const int32_t* __restrict__ weight_q,
                                              const T* __restrict__ weight_scales, const int8_t* __restrict__ act_q,
                                              const T* __restrict__ act_scales, const T* __restrict__ bias, size_t N,
                                              size_t K, int group_size) {
    const size_t row_idx = blockIdx.x;
    if (row_idx >= N) {
        return;
    }

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int32_t* w_row = weight_q + row_idx * (K / 8);
    const T* w_scales_row = weight_scales + row_idx * (K / group_size);

    float d_a = __ldg(&act_scales[0]);
    float thread_sum = 0.0f;
    const int num_blocks = K / 32;

    for (int b = tid; b < num_blocks; b += num_threads) {
        float d_w = to_float(__ldg(&w_scales_row[b * 32 / group_size]));

        int4 w_v = __ldg(reinterpret_cast<const int4*>(w_row + b * 4));
        int4 a0 = __ldg(reinterpret_cast<const int4*>(act_q + b * 32));
        int4 a1 = __ldg(reinterpret_cast<const int4*>(act_q + b * 32 + 16));

        int sumi = 0;
        sumi = detail::dp4a_q4_chunk(w_v.x, a0.x, a0.y, sumi);
        sumi = detail::dp4a_q4_chunk(w_v.y, a0.z, a0.w, sumi);
        sumi = detail::dp4a_q4_chunk(w_v.z, a1.x, a1.y, sumi);
        sumi = detail::dp4a_q4_chunk(w_v.w, a1.z, a1.w, sumi);

        thread_sum += static_cast<float>(sumi) * d_w;
    }

    __shared__ float shared_sum[detail::kWarpSize];
    thread_sum = detail::block_reduce_sum(thread_sum, shared_sum, tid, num_threads);
    if (tid < 32) {
        if (tid == 0) {
            detail::store_matvec_output(out, bias, row_idx, thread_sum, d_a, static_cast<int>(N));
        }
    }
}

template <typename T>
__global__ void matvec_q8_0_q8_row_soa_kernel(T* __restrict__ out, const int8_t* __restrict__ weight_q,
                                              const T* __restrict__ weight_scales, const int8_t* __restrict__ act_q,
                                              const T* __restrict__ act_scales, const T* __restrict__ bias, size_t N,
                                              size_t K, int group_size) {
    const size_t row_idx = blockIdx.x;
    if (row_idx >= N) {
        return;
    }

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int8_t* w_row = weight_q + row_idx * K;
    const T* w_scales_row = weight_scales + row_idx * (K / group_size);

    float d_a = __ldg(&act_scales[0]);
    float thread_sum = 0.0f;
    const int num_blocks = K / 32;

    for (int b = tid; b < num_blocks; b += num_threads) {
        float d_w = to_float(__ldg(&w_scales_row[b * 32 / group_size]));

        int4 w0 = __ldg(reinterpret_cast<const int4*>(w_row + b * 32));
        int4 w1 = __ldg(reinterpret_cast<const int4*>(w_row + b * 32 + 16));
        int4 a0 = __ldg(reinterpret_cast<const int4*>(act_q + b * 32));
        int4 a1 = __ldg(reinterpret_cast<const int4*>(act_q + b * 32 + 16));

        int sumi = 0;
        sumi = __dp4a(w0.x, a0.x, sumi);
        sumi = __dp4a(w0.y, a0.y, sumi);
        sumi = __dp4a(w0.z, a0.z, sumi);
        sumi = __dp4a(w0.w, a0.w, sumi);
        sumi = __dp4a(w1.x, a1.x, sumi);
        sumi = __dp4a(w1.y, a1.y, sumi);
        sumi = __dp4a(w1.z, a1.z, sumi);
        sumi = __dp4a(w1.w, a1.w, sumi);

        thread_sum += static_cast<float>(sumi) * d_w;
    }

    __shared__ float shared_sum[detail::kWarpSize];
    thread_sum = detail::block_reduce_sum(thread_sum, shared_sum, tid, num_threads);
    if (tid < 32) {
        if (tid == 0) {
            detail::store_matvec_output(out, bias, row_idx, thread_sum, d_a, static_cast<int>(N));
        }
    }
}

template <typename T>
__global__ void w8a8_gemm_soa_64x128_kernel(T* __restrict__ C, const int8_t* __restrict__ A_q,
                                            const T* __restrict__ A_scales, const int8_t* __restrict__ B_q,
                                            const T* __restrict__ B_scales, const T* __restrict__ bias, const size_t M,
                                            const size_t N, const size_t K, const int group_size) {
    constexpr int BM = 64, BN = 128, BK = 64;
    constexpr int APAD = 16, BPAD = 16;
    constexpr int A_STRIDE = BK + APAD;
    constexpr int B_STRIDE = BK + BPAD;

    int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int wid = tid >> 5, lane_id = tid & 31;

    extern __shared__ int8_t smem_raw[];
    constexpr size_t a_buf_size = BM * A_STRIDE;
    constexpr size_t b_buf_size = BN * B_STRIDE;

    int8_t* s_a = smem_raw;
    int8_t* s_b = s_a + 2 * a_buf_size;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[4][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c_int32[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c_fp32[2][4];

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_c_int32[i][j], 0);
            wmma::fill_fragment(frag_c_fp32[i][j], 0.0f);
        }

    int base_m = comp_c_frag_m * 32;
    int base_n = comp_c_frag_n * 64;

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 32;
    int gmem_m_a = by * BM + load_a_smem_m;
    int gmem_n_b = bx * BN + tid;

    auto cp_async_16b = [&](void* smem_ptr, const void* gmem_ptr, bool valid) {
        uint32_t sa = __cvta_generic_to_shared(smem_ptr);
        int bytes = valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem_ptr), "r"(bytes));
    };

    auto load_tile = [&](int k_start, int buf) {
        bool va0 = (gmem_m_a < (int)M && k_start + load_a_smem_k < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k],
                     va0 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k] : A_q, va0);

        bool va1 = (gmem_m_a < (int)M && k_start + load_a_smem_k + 16 < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k + 16],
                     va1 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k + 16] : A_q, va1);

        bool vb = (gmem_n_b < (int)N);
#pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            bool v = vb && (k_start + chunk * 16 < (int)K);
            cp_async_16b(&s_b[buf * b_buf_size + tid * B_STRIDE + chunk * 16],
                         v ? &B_q[gmem_n_b * K + k_start + chunk * 16] : B_q, v);
        }
    };

    // Prologue
    load_tile(0, 0);
    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int num_k_tiles = div_ceil((int)K, BK);
    int g_size = (group_size > 0) ? group_size : (int)K;
    int k_tiles_per_group = g_size / BK;
    int tile_count = 0, w_scale_idx = 0;
    int num_groups = (int)K / g_size;

    for (int bk = 1; bk <= num_k_tiles; bk++) {
        if (bk < num_k_tiles) {
            load_tile(bk * BK, bk & 1);
            asm("cp.async.commit_group;\n" ::);
        }

        // Compute current tile
        {
            int a_buf = (bk - 1) & 1;
            int b_buf = (bk - 1) & 1;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                int k_off = step * 16;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    wmma::load_matrix_sync(frag_a[step][i],
                                           &s_a[a_buf * a_buf_size + (base_m + i * 16) * A_STRIDE + k_off], A_STRIDE);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(frag_b[step][j],
                                           &s_b[b_buf * b_buf_size + (base_n + j * 16) * B_STRIDE + k_off], B_STRIDE);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        wmma::mma_sync(frag_c_int32[i][j], frag_a[step][i], frag_b[step][j], frag_c_int32[i][j]);
                    }
            }
        }

        // Per-group scale fold
        tile_count++;
        if (tile_count == k_tiles_per_group || bk == num_k_tiles) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int gm0 = by * BM + base_m + i * 16 + (lane_id / 4);
                float sa0_g = (gm0 < (int)M) ? to_float(A_scales[gm0 * num_groups + w_scale_idx]) : 0.0f;
                float sa1_g = (gm0 + 8 < (int)M) ? to_float(A_scales[(gm0 + 8) * num_groups + w_scale_idx]) : 0.0f;

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int gn0 = bx * BN + base_n + j * 16 + (lane_id % 4) * 2;
                    float sw0 = (gn0 < (int)N) ? to_float(B_scales[gn0 * num_groups + w_scale_idx]) : 0.0f;
                    float sw1 = (gn0 + 1 < (int)N) ? to_float(B_scales[(gn0 + 1) * num_groups + w_scale_idx]) : 0.0f;
                    float sw2 = (gn0 + 8 < (int)N) ? to_float(B_scales[(gn0 + 8) * num_groups + w_scale_idx]) : 0.0f;
                    float sw3 = (gn0 + 9 < (int)N) ? to_float(B_scales[(gn0 + 9) * num_groups + w_scale_idx]) : 0.0f;

                    frag_c_fp32[i][j].x[0] = fmaf((float)frag_c_int32[i][j].x[0], sa0_g * sw0, frag_c_fp32[i][j].x[0]);
                    frag_c_fp32[i][j].x[1] = fmaf((float)frag_c_int32[i][j].x[1], sa0_g * sw1, frag_c_fp32[i][j].x[1]);
                    frag_c_fp32[i][j].x[2] = fmaf((float)frag_c_int32[i][j].x[2], sa1_g * sw0, frag_c_fp32[i][j].x[2]);
                    frag_c_fp32[i][j].x[3] = fmaf((float)frag_c_int32[i][j].x[3], sa1_g * sw1, frag_c_fp32[i][j].x[3]);
                    frag_c_fp32[i][j].x[4] = fmaf((float)frag_c_int32[i][j].x[4], sa0_g * sw2, frag_c_fp32[i][j].x[4]);
                    frag_c_fp32[i][j].x[5] = fmaf((float)frag_c_int32[i][j].x[5], sa0_g * sw3, frag_c_fp32[i][j].x[5]);
                    frag_c_fp32[i][j].x[6] = fmaf((float)frag_c_int32[i][j].x[6], sa1_g * sw2, frag_c_fp32[i][j].x[6]);
                    frag_c_fp32[i][j].x[7] = fmaf((float)frag_c_int32[i][j].x[7], sa1_g * sw3, frag_c_fp32[i][j].x[7]);

                    wmma::fill_fragment(frag_c_int32[i][j], 0);
                }
            }
            tile_count = 0;
            w_scale_idx++;
        }

        if (bk < num_k_tiles) {
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }
    }

    __syncthreads();

    // Epilogue
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    float* s_c_float = reinterpret_cast<float*>(smem_raw);
    T* s_c_final = reinterpret_cast<T*>(s_c_float + 4 * 256);

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c_fp32[i][j], 16, wmma::mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_n = idx & 15;
                float b_val = (tile_n + local_n < (int)N && bias) ? to_float(bias[tile_n + local_n]) : 0.0f;
                s_c_final[wid * 256 + idx] = from_float<T>(s_c_float[wid * 256 + idx] + b_val);
            }
            __syncwarp();

            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;
            int gm = tile_m + row;
            int gn = tile_n + col;

            if (gm < (int)M && gn + 7 < (int)N) {
                if (reinterpret_cast<uintptr_t>(&C[gm * N + gn]) % 16 == 0) {
                    *reinterpret_cast<int4*>(&C[gm * N + gn])
                        = *reinterpret_cast<int4*>(&s_c_final[wid * 256 + row * 16 + col]);
                } else {
#pragma unroll
                    for (int c = 0; c < 8; c++) { C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c]; }
                }
            } else if (gm < (int)M) {
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (gn + c < (int)N) {
                        C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}


// ============================================================================
// W8A8 GEMM 128×256 tile — BK=64 + per-group activation scales
// ============================================================================
template <typename T>
__global__ void w8a8_gemm_soa_128x256_kernel(T* __restrict__ C, const int8_t* __restrict__ A_q,
                                             const T* __restrict__ A_scales, const int8_t* __restrict__ B_q,
                                             const T* __restrict__ B_scales, const T* __restrict__ bias, const size_t M,
                                             const size_t N, const size_t K, const int group_size) {
    constexpr int BM = 128, BN = 256, BK = 64;
    constexpr int APAD = 16, BPAD = 16;
    constexpr int A_STRIDE = BK + APAD;
    constexpr int B_STRIDE = BK + BPAD;

    int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int wid = tid >> 5, lane_id = tid & 31;

    extern __shared__ int8_t smem_raw[];
    constexpr size_t a_buf_size = BM * A_STRIDE;
    constexpr size_t b_buf_size = BN * B_STRIDE;

    int8_t* s_a = smem_raw;
    int8_t* s_b = s_a + 2 * a_buf_size;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[4][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c_int32[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c_fp32[4][4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_c_int32[i][j], 0);
            wmma::fill_fragment(frag_c_fp32[i][j], 0.0f);
        }

    int base_m = comp_c_frag_m * 64;
    int base_n = comp_c_frag_n * 64;

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 32;
    int gmem_m_a = by * BM + load_a_smem_m;
    int gmem_n_b = bx * BN + tid;

    auto cp_async_16b = [&](void* smem_ptr, const void* gmem_ptr, bool valid) {
        uint32_t sa = __cvta_generic_to_shared(smem_ptr);
        int bytes = valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem_ptr), "r"(bytes));
    };

    auto load_tile = [&](int k_start, int buf) {
        bool va0 = (gmem_m_a < (int)M && k_start + load_a_smem_k < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k],
                     va0 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k] : A_q, va0);

        bool va1 = (gmem_m_a < (int)M && k_start + load_a_smem_k + 16 < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k + 16],
                     va1 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k + 16] : A_q, va1);

        bool vb = (gmem_n_b < (int)N);
#pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            bool v = vb && (k_start + chunk * 16 < (int)K);
            cp_async_16b(&s_b[buf * b_buf_size + tid * B_STRIDE + chunk * 16],
                         v ? &B_q[gmem_n_b * K + k_start + chunk * 16] : B_q, v);
        }
    };

    load_tile(0, 0);
    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int num_k_tiles = div_ceil((int)K, BK);
    int g_size = (group_size > 0) ? group_size : (int)K;
    int k_tiles_per_group = g_size / BK;
    int tile_count = 0, w_scale_idx = 0;
    int num_groups = (int)K / g_size;

    for (int bk = 1; bk <= num_k_tiles; bk++) {
        if (bk < num_k_tiles) {
            load_tile(bk * BK, bk & 1);
            asm("cp.async.commit_group;\n" ::);
        }

        {
            int buf = (bk - 1) & 1;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                int k_off = step * 16;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wmma::load_matrix_sync(frag_a[step][i],
                                           &s_a[buf * a_buf_size + (base_m + i * 16) * A_STRIDE + k_off], A_STRIDE);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(frag_b[step][j],
                                           &s_b[buf * b_buf_size + (base_n + j * 16) * B_STRIDE + k_off], B_STRIDE);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        wmma::mma_sync(frag_c_int32[i][j], frag_a[step][i], frag_b[step][j], frag_c_int32[i][j]);
                    }
            }
        }

        // Scale fold with per-group activation scales
        tile_count++;
        if (tile_count == k_tiles_per_group || bk == num_k_tiles) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int gm0 = by * BM + base_m + i * 16 + (lane_id / 4);
                float sa0_g = (gm0 < (int)M) ? to_float(A_scales[gm0 * num_groups + w_scale_idx]) : 0.0f;
                float sa1_g = (gm0 + 8 < (int)M) ? to_float(A_scales[(gm0 + 8) * num_groups + w_scale_idx]) : 0.0f;

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int gn0 = bx * BN + base_n + j * 16 + (lane_id % 4) * 2;
                    float sw0 = (gn0 < (int)N) ? to_float(B_scales[gn0 * num_groups + w_scale_idx]) : 0.0f;
                    float sw1 = (gn0 + 1 < (int)N) ? to_float(B_scales[(gn0 + 1) * num_groups + w_scale_idx]) : 0.0f;
                    float sw2 = (gn0 + 8 < (int)N) ? to_float(B_scales[(gn0 + 8) * num_groups + w_scale_idx]) : 0.0f;
                    float sw3 = (gn0 + 9 < (int)N) ? to_float(B_scales[(gn0 + 9) * num_groups + w_scale_idx]) : 0.0f;

                    frag_c_fp32[i][j].x[0] = fmaf((float)frag_c_int32[i][j].x[0], sa0_g * sw0, frag_c_fp32[i][j].x[0]);
                    frag_c_fp32[i][j].x[1] = fmaf((float)frag_c_int32[i][j].x[1], sa0_g * sw1, frag_c_fp32[i][j].x[1]);
                    frag_c_fp32[i][j].x[2] = fmaf((float)frag_c_int32[i][j].x[2], sa1_g * sw0, frag_c_fp32[i][j].x[2]);
                    frag_c_fp32[i][j].x[3] = fmaf((float)frag_c_int32[i][j].x[3], sa1_g * sw1, frag_c_fp32[i][j].x[3]);
                    frag_c_fp32[i][j].x[4] = fmaf((float)frag_c_int32[i][j].x[4], sa0_g * sw2, frag_c_fp32[i][j].x[4]);
                    frag_c_fp32[i][j].x[5] = fmaf((float)frag_c_int32[i][j].x[5], sa0_g * sw3, frag_c_fp32[i][j].x[5]);
                    frag_c_fp32[i][j].x[6] = fmaf((float)frag_c_int32[i][j].x[6], sa1_g * sw2, frag_c_fp32[i][j].x[6]);
                    frag_c_fp32[i][j].x[7] = fmaf((float)frag_c_int32[i][j].x[7], sa1_g * sw3, frag_c_fp32[i][j].x[7]);

                    wmma::fill_fragment(frag_c_int32[i][j], 0);
                }
            }
            tile_count = 0;
            w_scale_idx++;
        }

        if (bk < num_k_tiles) {
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }
    }

    __syncthreads();

    // Epilogue
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    float* s_c_float = reinterpret_cast<float*>(smem_raw);
    T* s_c_final = reinterpret_cast<T*>(s_c_float + 8 * 256);

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c_fp32[i][j], 16, wmma::mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_n = idx & 15;
                float b_val = (tile_n + local_n < (int)N && bias) ? to_float(bias[tile_n + local_n]) : 0.0f;
                s_c_final[wid * 256 + idx] = from_float<T>(s_c_float[wid * 256 + idx] + b_val);
            }
            __syncwarp();

            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;
            int gm = tile_m + row;
            int gn = tile_n + col;

            if (gm < (int)M && gn + 7 < (int)N) {
                if (reinterpret_cast<uintptr_t>(&C[gm * N + gn]) % 16 == 0) {
                    *reinterpret_cast<int4*>(&C[gm * N + gn])
                        = *reinterpret_cast<int4*>(&s_c_final[wid * 256 + row * 16 + col]);
                } else {
#pragma unroll
                    for (int c = 0; c < 8; c++) { C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c]; }
                }
            } else if (gm < (int)M) {
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (gn + c < (int)N) {
                        C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}


// ============================================================================
// W4A8 GEMM 64×128 tile — Triple-buffered + BK=64 + per-group act scales
// ============================================================================
template <typename T>
__global__ void w4a8_gemm_soa_64x128_kernel(T* __restrict__ C, const int8_t* __restrict__ A_q,
                                            const T* __restrict__ A_scales, const uint8_t* __restrict__ B_q_packed,
                                            const T* __restrict__ B_scales, const T* __restrict__ bias, const size_t M,
                                            const size_t N, const size_t K, const int group_size) {
    constexpr int BM = 64, BN = 128, BK = 64;
    constexpr int APAD = 16, BPAD = 16;
    constexpr int A_STRIDE = BK + APAD;
    constexpr int B_STRIDE = BK + BPAD;
    constexpr int BK_PACKED = BK / 2;

    int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int wid = tid >> 5, lane_id = tid & 31;

    extern __shared__ int8_t smem_raw[];

    constexpr size_t a_buf_size = BM * A_STRIDE;
    constexpr size_t bp_buf_size = BN * BK_PACKED;
    constexpr size_t b_buf_size = BN * B_STRIDE;

    int8_t* s_a = smem_raw;
    uint8_t* s_b_packed = reinterpret_cast<uint8_t*>(s_a + 3 * a_buf_size);
    int8_t* s_b = reinterpret_cast<int8_t*>(s_b_packed + 3 * bp_buf_size);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[4][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c_int32[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c_fp32[2][4];

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_c_int32[i][j], 0);
            wmma::fill_fragment(frag_c_fp32[i][j], 0.0f);
        }

    int base_m = comp_c_frag_m * 32;
    int base_n = comp_c_frag_n * 64;

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 32;
    int gmem_m_a = by * BM + load_a_smem_m;
    int gmem_n_b = bx * BN + tid;

    auto cp_async_16b = [&](void* smem_ptr, const void* gmem_ptr, bool valid) {
        uint32_t sa = __cvta_generic_to_shared(smem_ptr);
        int bytes = valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem_ptr), "r"(bytes));
    };

    auto load_tile = [&](int k_start, int buf) {
        bool va0 = (gmem_m_a < (int)M && k_start + load_a_smem_k < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k],
                     va0 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k] : A_q, va0);

        bool va1 = (gmem_m_a < (int)M && k_start + load_a_smem_k + 16 < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k + 16],
                     va1 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k + 16] : A_q, va1);

        bool vb = (gmem_n_b < (int)N && k_start < (int)K);
        cp_async_16b(&s_b_packed[buf * bp_buf_size + tid * BK_PACKED],
                     vb ? &B_q_packed[gmem_n_b * (K / 2) + k_start / 2] : B_q_packed, vb);
        cp_async_16b(&s_b_packed[buf * bp_buf_size + tid * BK_PACKED + 16],
                     vb ? &B_q_packed[gmem_n_b * (K / 2) + k_start / 2 + 16] : B_q_packed, vb);
    };

    auto unpack_tile = [&](int src_buf, int dst_buf) {
        detail::unpack_row_int4x64(&s_b[dst_buf * b_buf_size + tid * B_STRIDE],
                                   &s_b_packed[src_buf * bp_buf_size + tid * BK_PACKED]);
    };

    auto compute_tile = [&](int a_buf, int b_buf) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            int k_off = step * 16;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(frag_a[step][i], &s_a[a_buf * a_buf_size + (base_m + i * 16) * A_STRIDE + k_off],
                                       A_STRIDE);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                wmma::load_matrix_sync(frag_b[step][j], &s_b[b_buf * b_buf_size + (base_n + j * 16) * B_STRIDE + k_off],
                                       B_STRIDE);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    wmma::mma_sync(frag_c_int32[i][j], frag_a[step][i], frag_b[step][j], frag_c_int32[i][j]);
        }
    };

    int g_size_val = (group_size > 0) ? group_size : (int)K;
    int num_groups = (int)K / g_size_val;

    auto fold_scales = [&](int w_scale_idx) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int gm0 = by * BM + base_m + i * 16 + (lane_id / 4);
            float sa0_g = (gm0 < (int)M) ? to_float(A_scales[gm0 * num_groups + w_scale_idx]) : 0.0f;
            float sa1_g = (gm0 + 8 < (int)M) ? to_float(A_scales[(gm0 + 8) * num_groups + w_scale_idx]) : 0.0f;

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gn0 = bx * BN + base_n + j * 16 + (lane_id % 4) * 2;
                float sw0 = (gn0 < (int)N) ? to_float(B_scales[gn0 * num_groups + w_scale_idx]) : 0.0f;
                float sw1 = (gn0 + 1 < (int)N) ? to_float(B_scales[(gn0 + 1) * num_groups + w_scale_idx]) : 0.0f;
                float sw2 = (gn0 + 8 < (int)N) ? to_float(B_scales[(gn0 + 8) * num_groups + w_scale_idx]) : 0.0f;
                float sw3 = (gn0 + 9 < (int)N) ? to_float(B_scales[(gn0 + 9) * num_groups + w_scale_idx]) : 0.0f;

                frag_c_fp32[i][j].x[0] = fmaf((float)frag_c_int32[i][j].x[0], sa0_g * sw0, frag_c_fp32[i][j].x[0]);
                frag_c_fp32[i][j].x[1] = fmaf((float)frag_c_int32[i][j].x[1], sa0_g * sw1, frag_c_fp32[i][j].x[1]);
                frag_c_fp32[i][j].x[2] = fmaf((float)frag_c_int32[i][j].x[2], sa1_g * sw0, frag_c_fp32[i][j].x[2]);
                frag_c_fp32[i][j].x[3] = fmaf((float)frag_c_int32[i][j].x[3], sa1_g * sw1, frag_c_fp32[i][j].x[3]);
                frag_c_fp32[i][j].x[4] = fmaf((float)frag_c_int32[i][j].x[4], sa0_g * sw2, frag_c_fp32[i][j].x[4]);
                frag_c_fp32[i][j].x[5] = fmaf((float)frag_c_int32[i][j].x[5], sa0_g * sw3, frag_c_fp32[i][j].x[5]);
                frag_c_fp32[i][j].x[6] = fmaf((float)frag_c_int32[i][j].x[6], sa1_g * sw2, frag_c_fp32[i][j].x[6]);
                frag_c_fp32[i][j].x[7] = fmaf((float)frag_c_int32[i][j].x[7], sa1_g * sw3, frag_c_fp32[i][j].x[7]);
                wmma::fill_fragment(frag_c_int32[i][j], 0);
            }
        }
    };

    // Pipeline
    int num_k_tiles = div_ceil((int)K, BK);
    int k_tiles_per_group = g_size_val / BK;
    int tile_count = 0, w_scale_idx = 0;

    load_tile(0, 0);
    asm("cp.async.commit_group;\n" ::);

    if (num_k_tiles > 1) {
        load_tile(BK, 1);
        asm("cp.async.commit_group;\n" ::);
    }

    if (num_k_tiles > 1) {
        asm("cp.async.wait_group 1;\n" ::);
    } else {
        asm("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();
    unpack_tile(0, 0);
    __syncthreads();

    for (int bk = 0; bk < num_k_tiles; bk++) {
        int a_buf = bk % 3;
        int b_buf = bk & 1;

        if (bk + 2 < num_k_tiles) {
            int load_buf = (bk + 2) % 3;
            load_tile((bk + 2) * BK, load_buf);
            asm("cp.async.commit_group;\n" ::);
        }

        if (bk + 1 < num_k_tiles) {
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }

        compute_tile(a_buf, b_buf);

        tile_count++;
        if (tile_count == k_tiles_per_group || bk == num_k_tiles - 1) {
            fold_scales(w_scale_idx);
            tile_count = 0;
            w_scale_idx++;
        }

        if (bk + 1 < num_k_tiles) {
            int next_packed_buf = (bk + 1) % 3;
            int next_b_buf = (bk + 1) & 1;
            unpack_tile(next_packed_buf, next_b_buf);
            __syncthreads();
        }
    }

    __syncthreads();

    // Epilogue
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    float* s_c_float = reinterpret_cast<float*>(smem_raw);
    T* s_c_final = reinterpret_cast<T*>(s_c_float + 4 * 256);

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c_fp32[i][j], 16, wmma::mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_n = idx & 15;
                float b_val = (tile_n + local_n < (int)N && bias) ? to_float(bias[tile_n + local_n]) : 0.0f;
                s_c_final[wid * 256 + idx] = from_float<T>(s_c_float[wid * 256 + idx] + b_val);
            }
            __syncwarp();

            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;
            int gm = tile_m + row;
            int gn = tile_n + col;

            if (gm < (int)M && gn + 7 < (int)N) {
                if (reinterpret_cast<uintptr_t>(&C[gm * N + gn]) % 16 == 0) {
                    *reinterpret_cast<int4*>(&C[gm * N + gn])
                        = *reinterpret_cast<int4*>(&s_c_final[wid * 256 + row * 16 + col]);
                } else {
#pragma unroll
                    for (int c = 0; c < 8; c++) { C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c]; }
                }
            } else if (gm < (int)M) {
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (gn + c < (int)N) {
                        C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}


// ============================================================================
// W4A8 GEMM 128×256 tile — BK=64 + per-group act scales
// ============================================================================
template <typename T>
__global__ void w4a8_gemm_soa_128x256_kernel(T* __restrict__ C, const int8_t* __restrict__ A_q,
                                             const T* __restrict__ A_scales, const uint8_t* __restrict__ B_q_packed,
                                             const T* __restrict__ B_scales, const T* __restrict__ bias, const size_t M,
                                             const size_t N, const size_t K, const int group_size) {
    constexpr int BM = 128, BN = 256, BK = 64;
    constexpr int APAD = 16, BPAD = 16;
    constexpr int A_STRIDE = BK + APAD;
    constexpr int B_STRIDE = BK + BPAD;
    constexpr int BK_PACKED = BK / 2;

    int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int wid = tid >> 5, lane_id = tid & 31;

    extern __shared__ int8_t smem_raw[];

    constexpr size_t a_buf_size = BM * A_STRIDE;
    constexpr size_t bp_buf_size = BN * BK_PACKED;

    int8_t* s_a = smem_raw;
    uint8_t* s_b_packed = reinterpret_cast<uint8_t*>(s_a + 2 * a_buf_size);
    int8_t* s_b = reinterpret_cast<int8_t*>(s_b_packed + 2 * bp_buf_size);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[4][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c_int32[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c_fp32[4][4];

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_c_int32[i][j], 0);
            wmma::fill_fragment(frag_c_fp32[i][j], 0.0f);
        }

    int base_m = comp_c_frag_m * 64;
    int base_n = comp_c_frag_n * 64;
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 32;
    int gmem_m_a = by * BM + load_a_smem_m;
    int gmem_n_b = bx * BN + tid;

    auto cp_async_16b = [&](void* smem_ptr, const void* gmem_ptr, bool valid) {
        uint32_t sa = __cvta_generic_to_shared(smem_ptr);
        int bytes = valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem_ptr), "r"(bytes));
    };

    auto load_tile = [&](int k_start, int buf) {
        bool va0 = (gmem_m_a < (int)M && k_start + load_a_smem_k < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k],
                     va0 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k] : A_q, va0);

        bool va1 = (gmem_m_a < (int)M && k_start + load_a_smem_k + 16 < (int)K);
        cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k + 16],
                     va1 ? &A_q[gmem_m_a * K + k_start + load_a_smem_k + 16] : A_q, va1);

        bool vb = (gmem_n_b < (int)N && k_start < (int)K);
        cp_async_16b(&s_b_packed[buf * bp_buf_size + tid * BK_PACKED],
                     vb ? &B_q_packed[gmem_n_b * (K / 2) + k_start / 2] : B_q_packed, vb);
        cp_async_16b(&s_b_packed[buf * bp_buf_size + tid * BK_PACKED + 16],
                     vb ? &B_q_packed[gmem_n_b * (K / 2) + k_start / 2 + 16] : B_q_packed, vb);
    };

    // Prologue
    load_tile(0, 0);
    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    detail::unpack_row_int4x64(&s_b[tid * B_STRIDE], &s_b_packed[tid * BK_PACKED]);
    __syncthreads();

    int num_k_tiles = div_ceil((int)K, BK);
    int g_size = (group_size > 0) ? group_size : (int)K;
    int k_tiles_per_group = g_size / BK;
    int tile_count = 0, w_scale_idx = 0;
    int num_groups = (int)K / g_size;

    for (int bk = 1; bk <= num_k_tiles; bk++) {
        if (bk < num_k_tiles) {
            int buf = bk & 1;
            load_tile(bk * BK, buf);
            asm("cp.async.commit_group;\n" ::);
        }

        // Compute
        {
            int a_buf = (bk - 1) & 1;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                int k_off = step * 16;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wmma::load_matrix_sync(frag_a[step][i],
                                           &s_a[a_buf * a_buf_size + (base_m + i * 16) * A_STRIDE + k_off], A_STRIDE);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(frag_b[step][j], &s_b[(base_n + j * 16) * B_STRIDE + k_off], B_STRIDE);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        wmma::mma_sync(frag_c_int32[i][j], frag_a[step][i], frag_b[step][j], frag_c_int32[i][j]);
                    }
            }
        }

        // Scale fold with per-group activation scales
        tile_count++;
        if (tile_count == k_tiles_per_group || bk == num_k_tiles) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int gm0 = by * BM + base_m + i * 16 + (lane_id / 4);
                float sa0_g = (gm0 < (int)M) ? to_float(A_scales[gm0 * num_groups + w_scale_idx]) : 0.0f;
                float sa1_g = (gm0 + 8 < (int)M) ? to_float(A_scales[(gm0 + 8) * num_groups + w_scale_idx]) : 0.0f;

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int gn0 = bx * BN + base_n + j * 16 + (lane_id % 4) * 2;
                    float sw0 = (gn0 < (int)N) ? to_float(B_scales[gn0 * num_groups + w_scale_idx]) : 0.0f;
                    float sw1 = (gn0 + 1 < (int)N) ? to_float(B_scales[(gn0 + 1) * num_groups + w_scale_idx]) : 0.0f;
                    float sw2 = (gn0 + 8 < (int)N) ? to_float(B_scales[(gn0 + 8) * num_groups + w_scale_idx]) : 0.0f;
                    float sw3 = (gn0 + 9 < (int)N) ? to_float(B_scales[(gn0 + 9) * num_groups + w_scale_idx]) : 0.0f;

                    frag_c_fp32[i][j].x[0] = fmaf((float)frag_c_int32[i][j].x[0], sa0_g * sw0, frag_c_fp32[i][j].x[0]);
                    frag_c_fp32[i][j].x[1] = fmaf((float)frag_c_int32[i][j].x[1], sa0_g * sw1, frag_c_fp32[i][j].x[1]);
                    frag_c_fp32[i][j].x[2] = fmaf((float)frag_c_int32[i][j].x[2], sa1_g * sw0, frag_c_fp32[i][j].x[2]);
                    frag_c_fp32[i][j].x[3] = fmaf((float)frag_c_int32[i][j].x[3], sa1_g * sw1, frag_c_fp32[i][j].x[3]);
                    frag_c_fp32[i][j].x[4] = fmaf((float)frag_c_int32[i][j].x[4], sa0_g * sw2, frag_c_fp32[i][j].x[4]);
                    frag_c_fp32[i][j].x[5] = fmaf((float)frag_c_int32[i][j].x[5], sa0_g * sw3, frag_c_fp32[i][j].x[5]);
                    frag_c_fp32[i][j].x[6] = fmaf((float)frag_c_int32[i][j].x[6], sa1_g * sw2, frag_c_fp32[i][j].x[6]);
                    frag_c_fp32[i][j].x[7] = fmaf((float)frag_c_int32[i][j].x[7], sa1_g * sw3, frag_c_fp32[i][j].x[7]);
                    wmma::fill_fragment(frag_c_int32[i][j], 0);
                }
            }
            tile_count = 0;
            w_scale_idx++;
        }

        if (bk < num_k_tiles) {
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();

            int buf = bk & 1;
            detail::unpack_row_int4x64(&s_b[tid * B_STRIDE], &s_b_packed[buf * bp_buf_size + tid * BK_PACKED]);
            __syncthreads();
        }
    }

    __syncthreads();

    // Epilogue — unchanged
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    float* s_c_float = reinterpret_cast<float*>(smem_raw);
    T* s_c_final = reinterpret_cast<T*>(s_c_float + 8 * 256);

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c_fp32[i][j], 16, wmma::mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_n = idx & 15;
                float b_val = (tile_n + local_n < (int)N && bias) ? to_float(bias[tile_n + local_n]) : 0.0f;
                s_c_final[wid * 256 + idx] = from_float<T>(s_c_float[wid * 256 + idx] + b_val);
            }
            __syncwarp();

            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;
            int gm = tile_m + row;
            int gn = tile_n + col;

            if (gm < (int)M && gn + 7 < (int)N) {
                if (reinterpret_cast<uintptr_t>(&C[gm * N + gn]) % 16 == 0) {
                    *reinterpret_cast<int4*>(&C[gm * N + gn])
                        = *reinterpret_cast<int4*>(&s_c_final[wid * 256 + row * 16 + col]);
                } else {
#pragma unroll
                    for (int c = 0; c < 8; c++) { C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c]; }
                }
            } else if (gm < (int)M) {
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (gn + c < (int)N) {
                        C[gm * N + gn + c] = s_c_final[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}


// ============================================================================
// Split-K Reduction
// ============================================================================
template <typename T>
__global__ void splitk_reduce_bias_kernel(T* __restrict__ output, const float* __restrict__ workspace,
                                          const T* __restrict__ bias, const size_t M, const size_t N,
                                          const int split_k) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= M * N) {
        return;
    }

    const size_t m = idx / N;
    const size_t n = idx % N;

    float acc = 0.0f;
#pragma unroll 8
    for (int s = 0; s < split_k; ++s) { acc += workspace[static_cast<size_t>(s) * M * N + m * N + n]; }

    float b_val = (bias && n < N) ? to_float(bias[n]) : 0.0f;
    output[m * N + n] = from_float<T>(acc + b_val);
}

template <typename T>
void launch_splitk_reduce(T* output, const float* workspace, const T* bias, size_t M, size_t N, int split_k,
                          cudaStream_t stream = nullptr) {
    size_t total = M * N;
    int block = 256;
    int grid = (int)div_ceil(total, (size_t)block);
    splitk_reduce_bias_kernel<T><<<grid, block, 0, stream>>>(output, workspace, bias, M, N, split_k);
}

template <typename T>
__global__ void w8a8_gemm_soa_64x128_splitk_kernel(float* __restrict__ workspace, // [split_k, M, N]
                                                   const int8_t* __restrict__ A_q, const T* __restrict__ A_scales,
                                                   const int8_t* __restrict__ B_q, const T* __restrict__ B_scales,
                                                   const size_t M, const size_t N, const size_t K, const int group_size,
                                                   const int split_k) {
    const int BM = 64, BN = 128, BK = 32;
    const int APAD = 16, BPAD = 16;

    int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int wid = tid >> 5, lane_id = tid & 31;

    // ---- Split-K slice parameters ----
    int k_slice = blockIdx.z;
    int g_size = (group_size > 0) ? group_size : (int)K;
    // Align to max(BK, group_size) so groups don't straddle slices
    int align = max(BK, g_size);
    int K_per_slice = ((int)K / split_k / align) * align;
    int k_base = k_slice * K_per_slice;
    int K_local = (k_slice == split_k - 1) ? ((int)K - k_base) : K_per_slice;
    int num_k_tiles = div_ceil(K_local, BK);
    if (num_k_tiles == 0) {
        return;
    }

    // ---- Shared memory layout ----
    extern __shared__ int8_t smem_raw[];
    int8_t* s_a = smem_raw;
    int8_t* s_b = s_a + 2 * BM * (BK + APAD);
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + BPAD);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[2][4];
    // [FIX #3] Added fp32 accumulators for correct per-group scale folding
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c_int32[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c_fp32[2][4];

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_c_int32[i][j], 0);
            wmma::fill_fragment(frag_c_fp32[i][j], 0.0f);
        }

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 16;
    int load_b_smem_n = tid / 2;
    int load_b_smem_k = (tid % 2) * 16;

    auto issue_cp_async = [&](int8_t* smem_ptr, const int8_t* gmem_ptr, bool valid) {
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
        int copy_bytes = valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_ptr),
                     "r"(copy_bytes));
    };

    int gmem_m_a = by * BM + load_a_smem_m;
    int gmem_n_b0 = bx * BN + load_b_smem_n;
    int gmem_n_b1 = gmem_n_b0 + 64;

    int base_m = comp_c_frag_m * 32;
    int base_n = comp_c_frag_n * 64;

    int num_groups = (int)K / g_size;
    int k_tiles_per_group = g_size / BK;
    int tile_count = 0;
    int w_scale_idx = k_base / g_size; // Start from the correct group for this slice

    // ---- Prologue: load tile 0 ----
    bool v_a = (gmem_m_a < (int)M && k_base + load_a_smem_k < (int)K);
    issue_cp_async(&s_a[OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD)],
                   v_a ? &A_q[gmem_m_a * K + k_base + load_a_smem_k] : A_q, v_a);

    bool v_b0 = (gmem_n_b0 < (int)N && k_base + load_b_smem_k < (int)K);
    issue_cp_async(&s_b[OFFSET(load_b_smem_n, load_b_smem_k, BK + BPAD)],
                   v_b0 ? &B_q[gmem_n_b0 * K + k_base + load_b_smem_k] : B_q, v_b0);

    bool v_b1 = (gmem_n_b1 < (int)N && k_base + load_b_smem_k < (int)K);
    issue_cp_async(&s_b[OFFSET(load_b_smem_n + 64, load_b_smem_k, BK + BPAD)],
                   v_b1 ? &B_q[gmem_n_b1 * K + k_base + load_b_smem_k] : B_q, v_b1);

    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // ---- Main K-loop ----
    for (int bk = 1; bk <= num_k_tiles; bk++) {
        // Async load next tile
        if (bk < num_k_tiles) {
            int k_global = k_base + bk * BK;
            int next_idx = bk & 1;

            v_a = (gmem_m_a < (int)M && k_global + load_a_smem_k < (int)K);
            issue_cp_async(&s_a[OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset],
                           v_a ? &A_q[gmem_m_a * K + k_global + load_a_smem_k] : A_q, v_a);

            v_b0 = (gmem_n_b0 < (int)N && k_global + load_b_smem_k < (int)K);
            issue_cp_async(&s_b[OFFSET(load_b_smem_n, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset],
                           v_b0 ? &B_q[gmem_n_b0 * K + k_global + load_b_smem_k] : B_q, v_b0);

            v_b1 = (gmem_n_b1 < (int)N && k_global + load_b_smem_k < (int)K);
            issue_cp_async(&s_b[OFFSET(load_b_smem_n + 64, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset],
                           v_b1 ? &B_q[gmem_n_b1 * K + k_global + load_b_smem_k] : B_q, v_b1);
            asm("cp.async.commit_group;\n" ::);
        }

        // Compute current tile
        {
            int curr_idx = (bk - 1) & 1;
            for (int step = 0; step < 2; ++step) {
                int k_offset = step * 16;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    wmma::load_matrix_sync(
                        frag_a[step][i], &s_a[OFFSET(base_m + i * 16, k_offset, BK + APAD) + curr_idx * s_a_db_offset],
                        BK + APAD);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(
                        frag_b[step][j], &s_b[OFFSET(base_n + j * 16, k_offset, BK + BPAD) + curr_idx * s_b_db_offset],
                        BK + BPAD);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        wmma::mma_sync(frag_c_int32[i][j], frag_a[step][i], frag_b[step][j], frag_c_int32[i][j]);
                    }
            }
        }

        // Per-group scale folding IN the main loop
        // With per-group activation scales
        tile_count++;
        if (tile_count == k_tiles_per_group || bk == num_k_tiles) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int gm0 = by * BM + base_m + i * 16 + (lane_id / 4);
                float sa0_g = (gm0 < (int)M) ? to_float(A_scales[gm0 * num_groups + w_scale_idx]) : 0.0f;
                float sa1_g = (gm0 + 8 < (int)M) ? to_float(A_scales[(gm0 + 8) * num_groups + w_scale_idx]) : 0.0f;

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int gn0 = bx * BN + base_n + j * 16 + (lane_id % 4) * 2;
                    float sw0 = (gn0 < (int)N) ? to_float(B_scales[gn0 * num_groups + w_scale_idx]) : 0.0f;
                    float sw1 = (gn0 + 1 < (int)N) ? to_float(B_scales[(gn0 + 1) * num_groups + w_scale_idx]) : 0.0f;
                    float sw2 = (gn0 + 8 < (int)N) ? to_float(B_scales[(gn0 + 8) * num_groups + w_scale_idx]) : 0.0f;
                    float sw3 = (gn0 + 9 < (int)N) ? to_float(B_scales[(gn0 + 9) * num_groups + w_scale_idx]) : 0.0f;

                    frag_c_fp32[i][j].x[0] = fmaf((float)frag_c_int32[i][j].x[0], sa0_g * sw0, frag_c_fp32[i][j].x[0]);
                    frag_c_fp32[i][j].x[1] = fmaf((float)frag_c_int32[i][j].x[1], sa0_g * sw1, frag_c_fp32[i][j].x[1]);
                    frag_c_fp32[i][j].x[2] = fmaf((float)frag_c_int32[i][j].x[2], sa1_g * sw0, frag_c_fp32[i][j].x[2]);
                    frag_c_fp32[i][j].x[3] = fmaf((float)frag_c_int32[i][j].x[3], sa1_g * sw1, frag_c_fp32[i][j].x[3]);
                    frag_c_fp32[i][j].x[4] = fmaf((float)frag_c_int32[i][j].x[4], sa0_g * sw2, frag_c_fp32[i][j].x[4]);
                    frag_c_fp32[i][j].x[5] = fmaf((float)frag_c_int32[i][j].x[5], sa0_g * sw3, frag_c_fp32[i][j].x[5]);
                    frag_c_fp32[i][j].x[6] = fmaf((float)frag_c_int32[i][j].x[6], sa1_g * sw2, frag_c_fp32[i][j].x[6]);
                    frag_c_fp32[i][j].x[7] = fmaf((float)frag_c_int32[i][j].x[7], sa1_g * sw3, frag_c_fp32[i][j].x[7]);

                    wmma::fill_fragment(frag_c_int32[i][j], 0);
                }
            }
            tile_count = 0;
            w_scale_idx++;
        }

        if (bk < num_k_tiles) {
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();
        }
    }

    // Epilogue: just write fp32 from frag_c_fp32 to workspace
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    float* s_c_float = reinterpret_cast<float*>(smem_raw);
    float* ws_base = workspace + static_cast<size_t>(k_slice) * M * N;

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c_fp32[i][j], 16, wmma::mem_row_major);
            __syncwarp();

// Write float directly to workspace
#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;
                int gm = tile_m + local_m;
                int gn = tile_n + local_n;

                if (gm < (int)M && gn < (int)N) {
                    ws_base[gm * N + gn] = s_c_float[wid * 256 + idx];
                }
            }
            __syncwarp();
        }
    }
}


// ============================================================================
// W4A8 GEMM Split-K
// ============================================================================
template <typename T>
__global__ void w4a8_gemm_soa_64x128_splitk_kernel(float* __restrict__ workspace, const int8_t* __restrict__ A_q,
                                                   const T* __restrict__ A_scales,
                                                   const uint8_t* __restrict__ B_q_packed,
                                                   const T* __restrict__ B_scales, const size_t M, const size_t N,
                                                   const size_t K, const int group_size, const int split_k) {
    const int BM = 64, BN = 128, BK = 32;
    const int APAD = 16, BPAD = 16;
    const int A_STRIDE = BK + APAD;
    const int B_STRIDE = BK + BPAD;
    const int BK_PACKED = BK / 2;

    int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    int wid = tid >> 5, lane_id = tid & 31;

    int k_slice = blockIdx.z;
    int g_size = (group_size > 0) ? group_size : (int)K;
    int align = max(BK, g_size);
    int K_per_slice = ((int)K / split_k / align) * align;
    int k_base = k_slice * K_per_slice;
    int K_local = (k_slice == split_k - 1) ? ((int)K - k_base) : K_per_slice;
    int num_k_tiles = div_ceil(K_local, BK);
    if (num_k_tiles == 0) {
        return;
    }

    extern __shared__ int8_t smem_raw[];
    const size_t a_buf_size = BM * A_STRIDE;
    const size_t bp_buf_size = BN * BK_PACKED;

    int8_t* s_a = smem_raw;
    uint8_t* s_b_packed = reinterpret_cast<uint8_t*>(s_a + 2 * a_buf_size);
    int8_t* s_b = reinterpret_cast<int8_t*>(s_b_packed + 2 * bp_buf_size);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> frag_a[2][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> frag_c_int32[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c_fp32[2][4];

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_c_int32[i][j], 0);
            wmma::fill_fragment(frag_c_fp32[i][j], 0.0f);
        }

    int base_m = comp_c_frag_m * 32;
    int base_n = comp_c_frag_n * 64;

    // [FIX #1] REMOVED: pre-cached sa0, sa1. Loaded per-group in fold section.

    int num_groups = (int)K / g_size;
    int k_tiles_per_group = g_size / BK;
    int tile_count = 0;
    int w_scale_idx = k_base / g_size;

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 16;
    int gmem_m_a = by * BM + load_a_smem_m;
    int gmem_n_b = bx * BN + tid;

    auto cp_async_16b = [&](void* smem_ptr, const void* gmem_ptr, bool valid) {
        uint32_t sa = __cvta_generic_to_shared(smem_ptr);
        int bytes = valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem_ptr), "r"(bytes));
    };

    // Prologue: load + unpack tile 0
    {
        bool va = (gmem_m_a < (int)M && k_base + load_a_smem_k < (int)K);
        cp_async_16b(&s_a[load_a_smem_m * A_STRIDE + load_a_smem_k],
                     va ? &A_q[gmem_m_a * K + k_base + load_a_smem_k] : A_q, va);

        bool vb = (gmem_n_b < (int)N && k_base < (int)K);
        cp_async_16b(&s_b_packed[tid * BK_PACKED], vb ? &B_q_packed[gmem_n_b * (K / 2) + k_base / 2] : B_q_packed, vb);
    }
    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    detail::unpack_row_int4x32(&s_b[tid * B_STRIDE], &s_b_packed[tid * BK_PACKED]);
    __syncthreads();

    // Main loop
    for (int bk = 1; bk <= num_k_tiles; bk++) {
        if (bk < num_k_tiles) {
            int k_global = k_base + bk * BK;
            int buf = bk & 1;

            bool va = (gmem_m_a < (int)M && k_global + load_a_smem_k < (int)K);
            cp_async_16b(&s_a[buf * a_buf_size + load_a_smem_m * A_STRIDE + load_a_smem_k],
                         va ? &A_q[gmem_m_a * K + k_global + load_a_smem_k] : A_q, va);

            bool vb = (gmem_n_b < (int)N && k_global < (int)K);
            cp_async_16b(&s_b_packed[buf * bp_buf_size + tid * BK_PACKED],
                         vb ? &B_q_packed[gmem_n_b * (K / 2) + k_global / 2] : B_q_packed, vb);

            asm("cp.async.commit_group;\n" ::);
        }

        // Compute
        {
            int a_buf = (bk - 1) & 1;
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                int k_off = step * 16;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    wmma::load_matrix_sync(frag_a[step][i],
                                           &s_a[a_buf * a_buf_size + (base_m + i * 16) * A_STRIDE + k_off], A_STRIDE);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::load_matrix_sync(frag_b[step][j], &s_b[(base_n + j * 16) * B_STRIDE + k_off], B_STRIDE);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        wmma::mma_sync(frag_c_int32[i][j], frag_a[step][i], frag_b[step][j], frag_c_int32[i][j]);
                    }
            }
        }

        // Per-group scale fold with per-group activation scales
        tile_count++;
        if (tile_count == k_tiles_per_group || bk == num_k_tiles) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int gm0 = by * BM + base_m + i * 16 + (lane_id / 4);
                float sa0_g = (gm0 < (int)M) ? to_float(A_scales[gm0 * num_groups + w_scale_idx]) : 0.0f;
                float sa1_g = (gm0 + 8 < (int)M) ? to_float(A_scales[(gm0 + 8) * num_groups + w_scale_idx]) : 0.0f;

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int gn0 = bx * BN + base_n + j * 16 + (lane_id % 4) * 2;
                    float sw0 = (gn0 < (int)N) ? to_float(B_scales[gn0 * num_groups + w_scale_idx]) : 0.0f;
                    float sw1 = (gn0 + 1 < (int)N) ? to_float(B_scales[(gn0 + 1) * num_groups + w_scale_idx]) : 0.0f;
                    float sw2 = (gn0 + 8 < (int)N) ? to_float(B_scales[(gn0 + 8) * num_groups + w_scale_idx]) : 0.0f;
                    float sw3 = (gn0 + 9 < (int)N) ? to_float(B_scales[(gn0 + 9) * num_groups + w_scale_idx]) : 0.0f;

                    frag_c_fp32[i][j].x[0] = fmaf((float)frag_c_int32[i][j].x[0], sa0_g * sw0, frag_c_fp32[i][j].x[0]);
                    frag_c_fp32[i][j].x[1] = fmaf((float)frag_c_int32[i][j].x[1], sa0_g * sw1, frag_c_fp32[i][j].x[1]);
                    frag_c_fp32[i][j].x[2] = fmaf((float)frag_c_int32[i][j].x[2], sa1_g * sw0, frag_c_fp32[i][j].x[2]);
                    frag_c_fp32[i][j].x[3] = fmaf((float)frag_c_int32[i][j].x[3], sa1_g * sw1, frag_c_fp32[i][j].x[3]);
                    frag_c_fp32[i][j].x[4] = fmaf((float)frag_c_int32[i][j].x[4], sa0_g * sw2, frag_c_fp32[i][j].x[4]);
                    frag_c_fp32[i][j].x[5] = fmaf((float)frag_c_int32[i][j].x[5], sa0_g * sw3, frag_c_fp32[i][j].x[5]);
                    frag_c_fp32[i][j].x[6] = fmaf((float)frag_c_int32[i][j].x[6], sa1_g * sw2, frag_c_fp32[i][j].x[6]);
                    frag_c_fp32[i][j].x[7] = fmaf((float)frag_c_int32[i][j].x[7], sa1_g * sw3, frag_c_fp32[i][j].x[7]);
                    wmma::fill_fragment(frag_c_int32[i][j], 0);
                }
            }
            tile_count = 0;
            w_scale_idx++;
        }

        if (bk < num_k_tiles) {
            asm("cp.async.wait_group 0;\n" ::);
            __syncthreads();

            int buf = bk & 1;
            detail::unpack_row_int4x32(&s_b[tid * B_STRIDE], &s_b_packed[buf * bp_buf_size + tid * BK_PACKED]);
            __syncthreads();
        }
    }

    __syncthreads();

    // Epilogue: write fp32 from frag_c_fp32 to workspace (NO bias)
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    float* s_c_float = reinterpret_cast<float*>(smem_raw);
    float* ws_base = workspace + static_cast<size_t>(k_slice) * M * N;

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c_fp32[i][j], 16, wmma::mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;
                int gm = tile_m + local_m;
                int gn = tile_n + local_n;

                if (gm < (int)M && gn < (int)N) {
                    ws_base[gm * N + gn] = s_c_float[wid * 256 + idx];
                }
            }
            __syncwarp();
        }
    }
}

} // namespace zedinfer::ops::nvidia
