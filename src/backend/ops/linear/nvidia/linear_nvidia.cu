#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "backend/core/context/context.hpp"
#include "backend/core/storage/storage.hpp"
#include "backend/ops/linear/nvidia/linear_cublas.cuh"
#include "linear_quantized_kernel.cuh"
#include "quantize.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "zedinfer.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K) {
    // cuBLASLt handles all shapes (M=1 GEMV and M>1 GEMM) and all dtypes
    // (FP32, FP16, BF16) with automatic kernel selection and fused bias.
    cublas::linear(output, input, weight, bias, type, M, N, K);
}

// ============================================================================
// Autotuning helpers
// ============================================================================

// ============================================================================
// SM count cache
// ============================================================================
static int get_num_sms() {
    static int num_sms = 0;
    if (num_sms == 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    }
    return num_sms;
}

static int select_split_k(int base_blocks, int num_sms, int K, int BK, int group_size) {
    if (base_blocks >= num_sms) return 1;
    int align = BK;
    if (group_size > 0 && group_size < K) align = std::max(align, group_size);
    int max_splits = K / align;
    for (int sk = 2; sk <= std::min(max_splits, 16); sk++) {
        int k_per_slice = K / sk;
        if (k_per_slice % align != 0) continue;
        if (k_per_slice < BK) break;
        if (base_blocks * sk >= num_sms) return sk;
    }
    return 1;
}

// ============================================================================
// Main entry point
// ============================================================================
void launch_linear_quantized_autotuned(
    cuda_bfloat16 *output,
    const cuda_bfloat16 *input,
    const void *weight_packed,
    const cuda_bfloat16 *bias,
    const cuda_bfloat16 *scales,
    const size_t M,
    const size_t N,
    const size_t K,
    const int num_bits,
    int group_size)
{
    group_size = (group_size > 0) ? group_size : K;
    auto &runtime = zedinfer::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());

    // ==================================================================
    // Decode path (M == 1)
    // ==================================================================
    if (M == 1) {
        size_t q_size = K * sizeof(int8_t);
        size_t scale_size = 1 * sizeof(cuda_bfloat16);

        auto act_q_storage = runtime.allocateDeviceStorage(q_size);
        auto act_scale_storage = runtime.allocateDeviceStorage(scale_size);
        int8_t *act_q = reinterpret_cast<int8_t *>(act_q_storage->memory());
        cuda_bfloat16 *act_scales =
            reinterpret_cast<cuda_bfloat16 *>(act_scale_storage->memory());

        launch_quantize_q8_row<cuda_bfloat16>(
            input, act_q, act_scales, 1, K, stream);

        int threads_needed = K / 32;
        int block_size = std::min(256, (threads_needed + 31) / 32 * 32);
        dim3 grid(N);
        dim3 block(block_size);

        if (num_bits == 4) {
            matvec_q4_0_q8_row_soa_kernel<cuda_bfloat16><<<grid, block, 0, stream>>>(
                output, reinterpret_cast<const int32_t*>(weight_packed), scales,
                act_q, act_scales, bias, N, K, group_size);
        } else if (num_bits == 8) {
            matvec_q8_0_q8_row_soa_kernel<cuda_bfloat16><<<grid, block, 0, stream>>>(
                output, reinterpret_cast<const int8_t*>(weight_packed), scales,
                act_q, act_scales, bias, N, K, group_size);
        }
        return;
    }

    // ==================================================================
    // Prefill path (M > 1)
    // ==================================================================
    const int num_sms = get_num_sms();

    int act_group_size;
    int num_act_groups;

    if (num_bits == 4) {
        // Per-group: match weight group_size
        act_group_size = group_size;
        num_act_groups = (act_group_size > 0 && act_group_size < (int)K)
                         ? ((int)K / act_group_size) : 1;
    } else {
        // Default INT8 behavior is per-row. If group_size has been inferred
        // from an expanded scale tensor, reuse it for grouped activation
        // quantization during prefill.
        act_group_size = (group_size > 0 && group_size < (int)K)
                         ? group_size : (int)K;
        num_act_groups = (act_group_size > 0 && act_group_size < (int)K)
                         ? ((int)K / act_group_size) : 1;
    }

    size_t q_size = M * K * sizeof(int8_t);
    size_t scale_size = M * num_act_groups * sizeof(cuda_bfloat16);
    auto act_q_storage = runtime.allocateDeviceStorage(q_size);
    auto act_scale_storage = runtime.allocateDeviceStorage(scale_size);
    int8_t *act_q = reinterpret_cast<int8_t *>(act_q_storage->memory());
    cuda_bfloat16 *act_scales =
        reinterpret_cast<cuda_bfloat16 *>(act_scale_storage->memory());

    launch_quantize_q8_row_grouped<cuda_bfloat16>(
        input, act_q, act_scales, M, K, act_group_size, stream);

    int grid_large_x = (int)div_ceil(N, (size_t)256);
    int grid_large_y = (int)div_ceil(M, (size_t)128);
    int blocks_large = grid_large_x * grid_large_y;

    int grid_small_x = (int)div_ceil(N, (size_t)128);
    int grid_small_y = (int)div_ceil(M, (size_t)64);
    int blocks_small = grid_small_x * grid_small_y;

    bool can_use_bk64 = (K % 64 == 0) && (group_size % 64 == 0 || group_size >= (int)K);

    if (num_bits == 4) {
        constexpr size_t smem_large = 57344;
        constexpr size_t smem_small = 48128;

        if (can_use_bk64 && blocks_large >= num_sms) {
            cudaFuncSetAttribute(w4a8_gemm_soa_128x256_kernel<cuda_bfloat16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_large);
            w4a8_gemm_soa_128x256_kernel<cuda_bfloat16>
                <<<dim3(grid_large_x, grid_large_y), 256, smem_large, stream>>>(
                output, act_q, act_scales,
                reinterpret_cast<const uint8_t*>(weight_packed), scales, bias,
                M, N, K, group_size);

        } else if (can_use_bk64 && blocks_small >= num_sms) {
            cudaFuncSetAttribute(w4a8_gemm_soa_64x128_kernel<cuda_bfloat16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_small);
            w4a8_gemm_soa_64x128_kernel<cuda_bfloat16>
                <<<dim3(grid_small_x, grid_small_y), 128, smem_small, stream>>>(
                output, act_q, act_scales,
                reinterpret_cast<const uint8_t*>(weight_packed), scales, bias,
                M, N, K, group_size);

        } else {
            constexpr size_t smem_splitk = 16384;
            int split_k = select_split_k(blocks_small, num_sms, (int)K, 32, group_size);

            if (split_k <= 1) {
                if (can_use_bk64) {
                    cudaFuncSetAttribute(w4a8_gemm_soa_64x128_kernel<cuda_bfloat16>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_small);
                    w4a8_gemm_soa_64x128_kernel<cuda_bfloat16>
                        <<<dim3(grid_small_x, grid_small_y), 128, smem_small, stream>>>(
                        output, act_q, act_scales,
                        reinterpret_cast<const uint8_t*>(weight_packed), scales, bias,
                        M, N, K, group_size);
                } else {
                    cudaFuncSetAttribute(w4a8_gemm_soa_64x128_kernel<cuda_bfloat16>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
                    w4a8_gemm_soa_64x128_kernel<cuda_bfloat16>
                        <<<dim3(grid_small_x, grid_small_y), 128, 16384, stream>>>(
                        output, act_q, act_scales,
                        reinterpret_cast<const uint8_t*>(weight_packed), scales, bias,
                        M, N, K, group_size);
                }
            } else {
                size_t ws_size = (size_t)split_k * M * N * sizeof(float);
                auto workspace_storage = runtime.allocateDeviceStorage(ws_size);
                float *workspace =
                    reinterpret_cast<float *>(workspace_storage->memory());
                cudaMemsetAsync(workspace, 0, ws_size, stream);

                cudaFuncSetAttribute(w4a8_gemm_soa_64x128_splitk_kernel<cuda_bfloat16>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_splitk);
                w4a8_gemm_soa_64x128_splitk_kernel<cuda_bfloat16>
                    <<<dim3(grid_small_x, grid_small_y, split_k), 128, smem_splitk, stream>>>(
                    workspace, act_q, act_scales,
                    reinterpret_cast<const uint8_t*>(weight_packed), scales,
                    M, N, K, group_size, split_k);

                launch_splitk_reduce<cuda_bfloat16>(
                    output, workspace, bias, M, N, split_k, stream);
            }
        }

    } else if (num_bits == 8) {
        constexpr size_t smem_large_opt = 61440;
        constexpr size_t smem_small_opt = 30720;
        constexpr size_t smem_small_orig = 18432;

        if (can_use_bk64 && blocks_large >= num_sms) {
            cudaFuncSetAttribute(w8a8_gemm_soa_128x256_kernel<cuda_bfloat16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_large_opt);
            w8a8_gemm_soa_128x256_kernel<cuda_bfloat16>
                <<<dim3(grid_large_x, grid_large_y), 256, smem_large_opt, stream>>>(
                output, act_q, act_scales,
                reinterpret_cast<const int8_t*>(weight_packed), scales, bias,
                M, N, K, group_size);

        } else if (can_use_bk64 && blocks_small >= num_sms) {
            cudaFuncSetAttribute(w8a8_gemm_soa_64x128_kernel<cuda_bfloat16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_small_opt);
            w8a8_gemm_soa_64x128_kernel<cuda_bfloat16>
                <<<dim3(grid_small_x, grid_small_y), 128, smem_small_opt, stream>>>(
                output, act_q, act_scales,
                reinterpret_cast<const int8_t*>(weight_packed), scales, bias,
                M, N, K, group_size);

        } else {
            int split_k = select_split_k(blocks_small, num_sms, (int)K, 32, group_size);

            if (split_k <= 1) {
                if (can_use_bk64) {
                    cudaFuncSetAttribute(w8a8_gemm_soa_64x128_kernel<cuda_bfloat16>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_small_opt);
                    w8a8_gemm_soa_64x128_kernel<cuda_bfloat16>
                        <<<dim3(grid_small_x, grid_small_y), 128, smem_small_opt, stream>>>(
                        output, act_q, act_scales,
                        reinterpret_cast<const int8_t*>(weight_packed), scales, bias,
                        M, N, K, group_size);
                } else {
                    cudaFuncSetAttribute(w8a8_gemm_soa_64x128_kernel<cuda_bfloat16>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_small_orig);
                    w8a8_gemm_soa_64x128_kernel<cuda_bfloat16>
                        <<<dim3(grid_small_x, grid_small_y), 128, smem_small_orig, stream>>>(
                        output, act_q, act_scales,
                        reinterpret_cast<const int8_t*>(weight_packed), scales, bias,
                        M, N, K, group_size);
                }
            } else {
                size_t ws_size = (size_t)split_k * M * N * sizeof(float);
                auto workspace_storage = runtime.allocateDeviceStorage(ws_size);
                float *workspace =
                    reinterpret_cast<float *>(workspace_storage->memory());
                cudaMemsetAsync(workspace, 0, ws_size, stream);

                cudaFuncSetAttribute(w8a8_gemm_soa_64x128_splitk_kernel<cuda_bfloat16>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_small_orig);
                w8a8_gemm_soa_64x128_splitk_kernel<cuda_bfloat16>
                    <<<dim3(grid_small_x, grid_small_y, split_k), 128, smem_small_orig, stream>>>(
                    workspace, act_q, act_scales,
                    reinterpret_cast<const int8_t*>(weight_packed), scales,
                    M, N, K, group_size, split_k);

                launch_splitk_reduce<cuda_bfloat16>(
                    output, workspace, bias, M, N, split_k, stream);
            }
        }
    }
}

void linear_quantized(std::byte *output, const std::byte *input, const std::byte *weight, const std::byte *bias, const std::byte *scale, const std::byte *g_idx, zedinferDataType_t type, int num_bits, int group_size, size_t M, size_t N, size_t K) {
    (void)g_idx;

    if (num_bits != 4 && num_bits != 8) {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    if (type == ZEDINFER_DTYPE_BF16) {
        launch_linear_quantized_autotuned(
            reinterpret_cast<cuda_bfloat16 *>(output),
            reinterpret_cast<const cuda_bfloat16 *>(input),
            weight,
            reinterpret_cast<const cuda_bfloat16 *>(bias),
            reinterpret_cast<const cuda_bfloat16 *>(scale),
            M, N, K, num_bits, group_size);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
