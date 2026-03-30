#include "backend/core/context/context.hpp"
#include "backend/core/storage/storage.hpp"
#include "backend/ops/linear/nvidia/linear_cublas.cuh"
#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "linear_quantized_kernel.cuh"
#include "quantize.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "zedinfer.h"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K) {
    // cuBLASLt handles all shapes (M=1 GEMV and M>1 GEMM) and all dtypes
    // (FP32, FP16, BF16) with automatic kernel selection and fused bias.
    cublas::linear(output, input, weight, bias, type, M, N, K);
}

namespace {

enum class QuantizedLinearKernelFamily {
    SmallBatch,
    TensorOp64x128,
    TensorOp128x256,
    TensorOp64x128SplitK,
};

struct QuantizedLinearKernelConfig {
    QuantizedLinearKernelFamily family = QuantizedLinearKernelFamily::TensorOp64x128;
    int rows_per_block = 1;
    int split_k = 1;
    int threads = 128;
    size_t smem_bytes = 0;
};

template <typename T> struct QuantizedLinearWorkspace {
    zedinfer::core::storage_t activation_storage;
    zedinfer::core::storage_t splitk_storage;
    int8_t* act_q = nullptr;
    T* act_scales = nullptr;
    float* splitk_workspace = nullptr;
};

static int get_num_sms() {
    int device = 0;
    int num_sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    return num_sms;
}

static int get_compute_capability() {
    int device = 0;
    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major * 10 + minor;
}

static int round_up_to_warp(int value) {
    return std::clamp(((value + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, WARP_SIZE, BLOCK_SIZE);
}

static int select_split_k(int base_blocks, int num_sms, int K, int BK, int group_size, int cc) {
    if (base_blocks >= num_sms) {
        return 1;
    }

    int align = BK;
    if (group_size > 0 && group_size < K) {
        align = std::max(align, group_size);
    }

    const int max_splits = std::min(K / align, cc >= 90 ? 16 : 8);
    for (int sk = 2; sk <= max_splits; ++sk) {
        const int k_per_slice = K / sk;
        if (k_per_slice % align != 0) {
            continue;
        }
        if (k_per_slice < BK) {
            break;
        }
        if (base_blocks * sk >= num_sms) {
            return sk;
        }
    }
    return 1;
}

static double tile_coverage(size_t extent, int tile) {
    if (extent == 0) {
        return 0.0;
    }
    return static_cast<double>(extent) / static_cast<double>(div_ceil(extent, static_cast<size_t>(tile)) * tile);
}

static int select_small_batch_rows_per_block(size_t M) {
    return static_cast<int>(std::min(M, static_cast<size_t>(detail::kMaxSmallBatchRows)));
}

static int select_small_batch_threads(size_t K, int rows_per_block) {
    const int chunks = static_cast<int>(div_ceil(K, static_cast<size_t>(32)));
    const int max_threads = rows_per_block >= 4 ? 128 : 256;
    return std::min(round_up_to_warp(chunks), max_threads);
}

static QuantizedLinearKernelConfig select_quantized_linear_kernel(size_t M, size_t N, size_t K, int num_bits,
                                                                  int group_size, int cc, int num_sms) {
    QuantizedLinearKernelConfig config{};

    if (M >= 2 && M <= static_cast<size_t>(detail::kMaxSmallBatchRows)) {
        config.family = QuantizedLinearKernelFamily::SmallBatch;
        config.rows_per_block = select_small_batch_rows_per_block(M);
        config.threads = select_small_batch_threads(K, config.rows_per_block);
        return config;
    }

    const int grid_large_x = static_cast<int>(div_ceil(N, static_cast<size_t>(256)));
    const int grid_large_y = static_cast<int>(div_ceil(M, static_cast<size_t>(128)));
    const int blocks_large = grid_large_x * grid_large_y;

    const int grid_small_x = static_cast<int>(div_ceil(N, static_cast<size_t>(128)));
    const int grid_small_y = static_cast<int>(div_ceil(M, static_cast<size_t>(64)));
    const int blocks_small = grid_small_x * grid_small_y;

    const double occ_small = std::min(1.0, static_cast<double>(blocks_small) / std::max(1, num_sms));
    const double occ_large = std::min(1.0, static_cast<double>(blocks_large) / std::max(1, num_sms));

    const double small_score
        = tile_coverage(M, 64) * tile_coverage(N, 128) * (0.65 + 0.35 * occ_small);

    double large_arch_bonus = cc >= 90 ? 1.18 : (cc >= 89 ? 1.10 : 1.0);
    if (num_bits == 4 && cc < 89) {
        large_arch_bonus *= 0.93;
    }

    double large_score
        = tile_coverage(M, 128) * tile_coverage(N, 256) * (0.65 + 0.35 * occ_large) * large_arch_bonus;

    if (M < 96) {
        large_score *= 0.88;
    }
    if (N < 192) {
        large_score *= 0.82;
    }

    const int split_k = select_split_k(blocks_small, num_sms, static_cast<int>(K), 32, group_size, cc);
    const bool prefer_large_tile = blocks_large > 0 && large_score >= small_score * 1.05;
    const bool prefer_splitk = split_k > 1 && blocks_small < std::max(1, num_sms * 3 / 4);

    if (prefer_large_tile) {
        config.family = QuantizedLinearKernelFamily::TensorOp128x256;
        config.threads = 256;
        config.smem_bytes = num_bits == 4 ? 57344 : 61440;
        return config;
    }

    if (prefer_splitk) {
        config.family = QuantizedLinearKernelFamily::TensorOp64x128SplitK;
        config.split_k = split_k;
        config.threads = 128;
        config.smem_bytes = num_bits == 4 ? 16384 : 18432;
        return config;
    }

    config.family = QuantizedLinearKernelFamily::TensorOp64x128;
    config.threads = 128;
    config.smem_bytes = num_bits == 4 ? 48128 : 30720;
    return config;
}

template <typename T>
QuantizedLinearWorkspace<T> allocate_quantized_linear_workspace(zedinfer::core::Runtime& runtime, size_t M, size_t K,
                                                                int num_act_groups, int split_k, size_t N,
                                                                cudaStream_t stream) {
    QuantizedLinearWorkspace<T> workspace{};

    const size_t q_bytes = M * K * sizeof(int8_t);
    const size_t scale_offset = (q_bytes + 15u) & ~size_t(15u);
    const size_t scale_bytes = M * static_cast<size_t>(num_act_groups) * sizeof(T);

    workspace.activation_storage = runtime.allocateDeviceStorage(scale_offset + scale_bytes);
    auto* base = reinterpret_cast<std::byte*>(workspace.activation_storage->memory());
    workspace.act_q = reinterpret_cast<int8_t*>(base);
    workspace.act_scales = reinterpret_cast<T*>(base + scale_offset);

    if (split_k > 1) {
        const size_t splitk_bytes = static_cast<size_t>(split_k) * M * N * sizeof(float);
        workspace.splitk_storage = runtime.allocateDeviceStorage(splitk_bytes);
        workspace.splitk_workspace = reinterpret_cast<float*>(workspace.splitk_storage->memory());
        CUDA_CHECK(cudaMemsetAsync(workspace.splitk_workspace, 0, splitk_bytes, stream));
    }

    return workspace;
}

template <typename T>
void launch_decode_matvec_quantized(T* output, const T* input, const void* weight_packed, const T* bias, const T* scales,
                                    int num_bits, int group_size, size_t N, size_t K, zedinfer::core::Runtime& runtime,
                                    cudaStream_t stream) {
    auto workspace = allocate_quantized_linear_workspace<T>(runtime, 1, K, 1, 1, 0, stream);
    launch_quantize_q8_row<T>(input, workspace.act_q, workspace.act_scales, 1, K, stream);

    const int chunks = static_cast<int>(div_ceil(K, static_cast<size_t>(32)));
    const dim3 grid(static_cast<unsigned>(N));
    const dim3 block(static_cast<unsigned>(std::min(round_up_to_warp(chunks), BLOCK_SIZE)));

    if (num_bits == 4) {
        matvec_q4_0_q8_row_soa_kernel<T><<<grid, block, 0, stream>>>(
            output, reinterpret_cast<const int32_t*>(weight_packed), scales, workspace.act_q, workspace.act_scales,
            bias, N, K, group_size);
    } else {
        if (group_size >= static_cast<int>(K)) {
            matvec_q8_0_q8_row_soa_kernel<true, T><<<grid, block, 0, stream>>>(
                output, reinterpret_cast<const int8_t*>(weight_packed), scales, workspace.act_q, workspace.act_scales,
                bias, N, K, group_size);
        } else {
            matvec_q8_0_q8_row_soa_kernel<false, T><<<grid, block, 0, stream>>>(
                output, reinterpret_cast<const int8_t*>(weight_packed), scales, workspace.act_q, workspace.act_scales,
                bias, N, K, group_size);
        }
    }
}

template <int RowsPerBlock, typename T>
void launch_small_batch_quantized_linear_case(const dim3& grid, const dim3& block, T* output, const int8_t* act_q,
                                              const T* act_scales, const void* weight_packed, const T* scales,
                                              const T* bias, size_t M, size_t N, size_t K, int num_bits,
                                              int group_size, cudaStream_t stream) {
    if (num_bits == 4) {
        matmul_q4_0_q8_small_batch_soa_kernel<RowsPerBlock, T>
            <<<grid, block, 0, stream>>>(output, reinterpret_cast<const int32_t*>(weight_packed), scales, act_q,
                                         act_scales, bias, M, N, K, group_size);
        return;
    }

    if (group_size >= static_cast<int>(K)) {
        matmul_q8_0_q8_small_batch_soa_kernel<RowsPerBlock, true, T>
            <<<grid, block, 0, stream>>>(output, reinterpret_cast<const int8_t*>(weight_packed), scales, act_q,
                                         act_scales, bias, M, N, K, group_size);
    } else {
        matmul_q8_0_q8_small_batch_soa_kernel<RowsPerBlock, false, T>
            <<<grid, block, 0, stream>>>(output, reinterpret_cast<const int8_t*>(weight_packed), scales, act_q,
                                         act_scales, bias, M, N, K, group_size);
    }
}

template <typename T>
void launch_small_batch_quantized_linear(T* output, const int8_t* act_q, const T* act_scales, const void* weight_packed,
                                         const T* scales, const T* bias, size_t M, size_t N, size_t K, int num_bits,
                                         int group_size, int rows_per_block, int threads, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(div_ceil(M, static_cast<size_t>(rows_per_block))));
    const dim3 block(static_cast<unsigned>(threads));

    switch (rows_per_block) {
        case 8:
            launch_small_batch_quantized_linear_case<8>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        case 7:
            launch_small_batch_quantized_linear_case<7>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        case 6:
            launch_small_batch_quantized_linear_case<6>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        case 5:
            launch_small_batch_quantized_linear_case<5>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        case 4:
            launch_small_batch_quantized_linear_case<4>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        case 3:
            launch_small_batch_quantized_linear_case<3>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        case 2:
            launch_small_batch_quantized_linear_case<2>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
        default:
            launch_small_batch_quantized_linear_case<1>(grid, block, output, act_q, act_scales, weight_packed, scales,
                                                        bias, M, N, K, num_bits, group_size, stream);
            break;
    }
}

template <typename T>
void launch_tensorop_quantized_linear(const QuantizedLinearKernelConfig& config, T* output, const int8_t* act_q,
                                      const T* act_scales, const void* weight_packed, const T* scales, const T* bias,
                                      size_t M, size_t N, size_t K, int num_bits, int group_size, cudaStream_t stream) {
    if (config.family == QuantizedLinearKernelFamily::TensorOp128x256) {
        const dim3 grid(static_cast<unsigned>(div_ceil(N, static_cast<size_t>(256))),
                        static_cast<unsigned>(div_ceil(M, static_cast<size_t>(128))));
        if (num_bits == 4) {
            CUDA_CHECK(cudaFuncSetAttribute(w4a8_gemm_soa_128x256_kernel<T>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(config.smem_bytes)));
            w4a8_gemm_soa_128x256_kernel<T><<<grid, 256, config.smem_bytes, stream>>>(
                output, act_q, act_scales, reinterpret_cast<const uint8_t*>(weight_packed), scales, bias, M, N, K,
                group_size);
        } else {
            CUDA_CHECK(cudaFuncSetAttribute(w8a8_gemm_soa_128x256_kernel<T>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(config.smem_bytes)));
            w8a8_gemm_soa_128x256_kernel<T><<<grid, 256, config.smem_bytes, stream>>>(
                output, act_q, act_scales, reinterpret_cast<const int8_t*>(weight_packed), scales, bias, M, N, K,
                group_size);
        }
        return;
    }

    const dim3 grid(static_cast<unsigned>(div_ceil(N, static_cast<size_t>(128))),
                    static_cast<unsigned>(div_ceil(M, static_cast<size_t>(64))));

    if (num_bits == 4) {
        CUDA_CHECK(cudaFuncSetAttribute(w4a8_gemm_soa_64x128_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(config.smem_bytes)));
        w4a8_gemm_soa_64x128_kernel<T><<<grid, 128, config.smem_bytes, stream>>>(
            output, act_q, act_scales, reinterpret_cast<const uint8_t*>(weight_packed), scales, bias, M, N, K,
            group_size);
    } else {
        CUDA_CHECK(cudaFuncSetAttribute(w8a8_gemm_soa_64x128_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(config.smem_bytes)));
        w8a8_gemm_soa_64x128_kernel<T><<<grid, 128, config.smem_bytes, stream>>>(
            output, act_q, act_scales, reinterpret_cast<const int8_t*>(weight_packed), scales, bias, M, N, K,
            group_size);
    }
}

template <typename T>
void launch_splitk_quantized_linear(const QuantizedLinearKernelConfig& config, T* output, const int8_t* act_q,
                                    const T* act_scales, const void* weight_packed, const T* scales, const T* bias,
                                    float* splitk_workspace, size_t M, size_t N, size_t K, int num_bits,
                                    int group_size, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(div_ceil(N, static_cast<size_t>(128))),
                    static_cast<unsigned>(div_ceil(M, static_cast<size_t>(64))),
                    static_cast<unsigned>(config.split_k));

    if (num_bits == 4) {
        CUDA_CHECK(cudaFuncSetAttribute(w4a8_gemm_soa_64x128_splitk_kernel<T>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(config.smem_bytes)));
        w4a8_gemm_soa_64x128_splitk_kernel<T><<<grid, 128, config.smem_bytes, stream>>>(
            splitk_workspace, act_q, act_scales, reinterpret_cast<const uint8_t*>(weight_packed), scales, M, N, K,
            group_size, config.split_k);
    } else {
        CUDA_CHECK(cudaFuncSetAttribute(w8a8_gemm_soa_64x128_splitk_kernel<T>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(config.smem_bytes)));
        w8a8_gemm_soa_64x128_splitk_kernel<T><<<grid, 128, config.smem_bytes, stream>>>(
            splitk_workspace, act_q, act_scales, reinterpret_cast<const int8_t*>(weight_packed), scales, M, N, K,
            group_size, config.split_k);
    }

    launch_splitk_reduce<T>(output, splitk_workspace, bias, M, N, config.split_k, stream);
}

template <typename T>
void launch_linear_quantized_autotuned(T* output, const T* input, const void* weight_packed, const T* bias,
                                       const T* scales, size_t M, size_t N, size_t K, int num_bits, int group_size) {
    group_size = (group_size > 0) ? group_size : static_cast<int>(K);

    auto& runtime = zedinfer::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());

    if (M == 1) {
        launch_decode_matvec_quantized(output, input, weight_packed, bias, scales, num_bits, group_size, N, K,
                                       runtime, stream);
        return;
    }

    const int num_sms = get_num_sms();
    const int cc = get_compute_capability();

    const int act_group_size
        = (num_bits == 4 || (group_size > 0 && group_size < static_cast<int>(K))) ? group_size : static_cast<int>(K);
    const int num_act_groups = (act_group_size > 0 && act_group_size < static_cast<int>(K))
                                   ? static_cast<int>(K) / act_group_size
                                   : 1;

    const auto config = select_quantized_linear_kernel(M, N, K, num_bits, act_group_size, cc, num_sms);
    auto workspace = allocate_quantized_linear_workspace<T>(runtime, M, K, num_act_groups,
                                                            config.family == QuantizedLinearKernelFamily::TensorOp64x128SplitK
                                                                ? config.split_k
                                                                : 1,
                                                            N, stream);

    launch_quantize_q8_row_grouped<T>(input, workspace.act_q, workspace.act_scales, M, K, act_group_size, stream);

    switch (config.family) {
        case QuantizedLinearKernelFamily::SmallBatch:
            launch_small_batch_quantized_linear(output, workspace.act_q, workspace.act_scales, weight_packed, scales,
                                                bias, M, N, K, num_bits, act_group_size, config.rows_per_block,
                                                config.threads, stream);
            break;
        case QuantizedLinearKernelFamily::TensorOp128x256:
        case QuantizedLinearKernelFamily::TensorOp64x128:
            launch_tensorop_quantized_linear(config, output, workspace.act_q, workspace.act_scales, weight_packed,
                                             scales, bias, M, N, K, num_bits, act_group_size, stream);
            break;
        case QuantizedLinearKernelFamily::TensorOp64x128SplitK:
            launch_splitk_quantized_linear(config, output, workspace.act_q, workspace.act_scales, weight_packed, scales,
                                           bias, workspace.splitk_workspace, M, N, K, num_bits, act_group_size,
                                           stream);
            break;
    }
}

} // namespace

void linear_quantized(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                      const std::byte* scale, const std::byte* g_idx, zedinferDataType_t type, int num_bits,
                      int group_size, size_t M, size_t N, size_t K) {
    (void)g_idx;

    if (num_bits != 4 && num_bits != 8) {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    if (type == ZEDINFER_DTYPE_BF16) {
        launch_linear_quantized_autotuned(reinterpret_cast<cuda_bfloat16*>(output),
                                          reinterpret_cast<const cuda_bfloat16*>(input), weight,
                                          reinterpret_cast<const cuda_bfloat16*>(bias),
                                          reinterpret_cast<const cuda_bfloat16*>(scale), M, N, K, num_bits, group_size);
    } else if (type == ZEDINFER_DTYPE_F16) {
        launch_linear_quantized_autotuned(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(input), weight,
                                          reinterpret_cast<const half*>(bias), reinterpret_cast<const half*>(scale), M,
                                          N, K, num_bits, group_size);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
