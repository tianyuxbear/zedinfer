#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "linear_bf16_kernel.cuh"
#include "linear_fp16_kernel.cuh"
#include "linear_fp32_kernel.cuh"
#include "matvec.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "zedinfer.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

// bf16 WMMA kernels require Ampere+ (sm_80). Guard the host-side dispatch
// function so that it is only compiled when the target arch supports bf16.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

/**
 * @brief Auto-tuned host function for dynamic Linear Kernel dispatch.
 * * Strategy derived from profiling data:
 * - Small M (<=128): Prefer 32x64 for low latency, 64x64 for larger N.
 * - Medium M (<=512): 128x128 is the workhorse. 128x256 for very large N.
 * - Large M (>=1024):
 * - Huge N: 128x256 is essential for throughput.
 * - Medium N: 64x128 is faster than 128x128 unless K is very large (compute bound).
 */
void launch_linear_bf16_kernel_autotuned(
    cuda_bfloat16 *output,
    const cuda_bfloat16 *input,
    const cuda_bfloat16 *weight,
    const cuda_bfloat16 *bias,
    const size_t M,
    const size_t N,
    const size_t K) {
    // === Kernel Config Constants ===
    constexpr int BK = 32;
    constexpr int PAD = 8;

    // === Shared Memory Calculation ===
    // Calculates dynamic shared memory needed for operands + pipeline stages
    auto calc_smem = [&](int bm, int bn) -> size_t {
        size_t operand_smem = 2 * (bm + bn) * (BK + PAD) * sizeof(cuda_bfloat16);
        size_t stage_smem = 8 * 16 * 16 * (sizeof(float) + sizeof(cuda_bfloat16));
        return operand_smem + stage_smem;
    };

    // === Dispatch Logic ===

    if (M <= 128) {
        // [Small M]
        // N=1536 -> 32x64 is best (0.016ms)
        // N=8960 -> 64x64 is best (0.047ms)
        if (N >= 4096) {
            constexpr int BM = 64;
            constexpr int BN = 64;
            size_t smem = calc_smem(BM, BN);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_64x64<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        } else {
            constexpr int BM = 32;
            constexpr int BN = 64;
            size_t smem = calc_smem(BM, BN);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_32x64<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        }
    } else if (M <= 512) {
        // [Medium M]
        // N=4096 -> 128x128 is best (0.138ms vs 0.22ms for 32x64)
        // N=12288 -> 128x256 is best (0.67ms)
        if (N >= 8192) {
            constexpr int BM = 128;
            constexpr int BN = 256;
            size_t smem = calc_smem(BM, BN);
            cudaFuncSetAttribute(linear_bf16_kernel_128x256, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_128x256<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        } else {
            constexpr int BM = 128;
            constexpr int BN = 128;
            size_t smem = calc_smem(BM, BN);
            cudaFuncSetAttribute(linear_bf16_kernel_128x128, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_128x128<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        }
    } else {
        // [Large M] (M >= 1024)
        // Case N=27648: 128x256 (4.19ms) >> 32x64 (18.45ms)
        // Case N=5120:
        //    If K is Small/Medium: 64x128 (0.40ms) > 128x128 (0.46ms)
        //    If K is Large (27648): 128x128 (2.61ms) > 64x128 (3.04ms)

        if (N >= 16384) {
            // Very large N -> Largest BN
            constexpr int BM = 128;
            constexpr int BN = 256;
            size_t smem = calc_smem(BM, BN);
            cudaFuncSetAttribute(linear_bf16_kernel_128x256, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_128x256<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        } else if (N <= 6144 && K < 10240) {
            // Medium N, Not compute bound -> 64x128 offers better wave quantization/occupancy
            constexpr int BM = 64;
            constexpr int BN = 128;
            size_t smem = calc_smem(BM, BN);
            // 64x128 might need smem adjustment depending on arch, safer to set it
            cudaFuncSetAttribute(linear_bf16_kernel_64x128, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_64x128<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        } else {
            // Default Large: 128x128 is robust for high compute (Large K)
            constexpr int BM = 128;
            constexpr int BN = 128;
            size_t smem = calc_smem(BM, BN);
            cudaFuncSetAttribute(linear_bf16_kernel_128x128, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
            linear_bf16_kernel_128x128<<<grid, BLOCK_SIZE, smem>>>(output, input, weight, bias, M, N, K);
        }
    }
}

#endif // __CUDA_ARCH__ >= 800

void linear(std::byte *output, const std::byte *input, const std::byte *weight, const std::byte *bias, zedinferDataType_t type, size_t M, size_t N, size_t K) {
    dim3 blockDim(BLOCK_SIZE);

    if (likely(M == 1)) {
        switch (type) {
        case ZEDINFER_DTYPE_F32: {
            dim3 gridDim(N);
            return matvec_kernel_warp_vec<<<gridDim, blockDim>>>(
                reinterpret_cast<float *>(output),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(weight),
                reinterpret_cast<const float *>(bias),
                N, K);
        } break;
        case ZEDINFER_DTYPE_F16: {
            dim3 gridDim(N);
            return matvec_kernel_warp_vec<<<gridDim, blockDim>>>(
                reinterpret_cast<half *>(output),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(weight),
                reinterpret_cast<const half *>(bias),
                N, K);
        } break;
        case ZEDINFER_DTYPE_BF16: {
            dim3 gridDim(N);
            return matvec_kernel_warp_vec<<<gridDim, blockDim>>>(
                reinterpret_cast<cuda_bfloat16 *>(output),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(weight),
                reinterpret_cast<const cuda_bfloat16 *>(bias),
                N, K);
        } break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    } else {
        switch (type) {
        case ZEDINFER_DTYPE_F32: {
            dim3 gridDim;
            gridDim.y = div_ceil(M, 128);
            gridDim.x = div_ceil(N, 128);
            return linear_fp32_kernel<<<gridDim, blockDim>>>(
                reinterpret_cast<float *>(output),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(weight),
                reinterpret_cast<const float *>(bias),
                M, N, K);

        } break;
        case ZEDINFER_DTYPE_F16: {
            constexpr size_t BM = 128;
            constexpr size_t BN = 256;
            constexpr size_t BK = 32;
            constexpr size_t PAD = 8;
            constexpr size_t smem_size = 2 * (BM + BN) * (BK + PAD) * sizeof(half) + 8 * 16 * 16 * sizeof(float) + 8 * 16 * 16 * sizeof(half);
            cudaFuncSetAttribute(linear_fp16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            dim3 gridDim;
            gridDim.y = div_ceil(M, BM);
            gridDim.x = div_ceil(N, BN);
            return linear_fp16_kernel<<<gridDim, blockDim, smem_size>>>(
                reinterpret_cast<half *>(output),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(weight),
                reinterpret_cast<const half *>(bias),
                M, N, K);
        } break;
        case ZEDINFER_DTYPE_BF16: {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
            launch_linear_bf16_kernel_autotuned(
                reinterpret_cast<cuda_bfloat16 *>(output),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(weight),
                reinterpret_cast<const cuda_bfloat16 *>(bias),
                M, N, K);
#else
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
#endif
        } break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}
} // namespace zedinfer::ops::nvidia