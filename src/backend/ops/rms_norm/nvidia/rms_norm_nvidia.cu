#include "backend/ops/rms_norm/nvidia/rms_norm_nvidia.cuh"
#include "backend/core/context/context.hpp"
#include "utils/check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <flashinfer/norm.cuh>

namespace zedinfer::ops::nvidia {

// FlashInfer's RMSNorm computes output = x / rms(x) * (weight + weight_bias).
//   add_one_to_weight=false  -> weight_bias=0 (standard RMSNorm)        -> flashinfer::RMSNorm
//   add_one_to_weight=true   -> weight_bias=1 (Qwen3.5/Gemma (1+w))     -> flashinfer::GemmaRMSNorm
//
// FlashInfer's API takes non-const T* for input and weight; we const_cast since the
// kernel is read-only on those buffers. The compute stream is fetched from the runtime
// API so kernels enqueue in FIFO order with the surrounding compute.
template <typename T>
static void launch_rms_norm(T* output, const T* input, const T* weight, float eps, size_t batch_size,
                            size_t hidden_size, bool add_one_to_weight, cudaStream_t stream) {
    auto* in  = const_cast<T*>(input);
    auto* w   = const_cast<T*>(weight);
    const uint32_t bs = static_cast<uint32_t>(batch_size);
    const uint32_t d  = static_cast<uint32_t>(hidden_size);
    cudaError_t err;
    if (add_one_to_weight) {
        err = flashinfer::norm::GemmaRMSNorm<T>(in, w, output, bs, d, /*stride_input=*/d,
                                          /*stride_output=*/d, eps, /*enable_pdl=*/false, stream);
    } else {
        err = flashinfer::norm::RMSNorm<T>(in, w, output, bs, d, /*stride_input=*/d,
                                      /*stride_output=*/d, eps, /*enable_pdl=*/false, stream);
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("FlashInfer RMSNorm failed: ") + cudaGetErrorString(err));
    }
}

void rms_norm(std::byte* output, const std::byte* input, const std::byte* weight, float eps, zedinferDataType_t type,
              size_t seq_len, size_t hidden_size, bool add_one_to_weight) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return launch_rms_norm<float>(reinterpret_cast<float*>(output),
                                          reinterpret_cast<const float*>(input),
                                          reinterpret_cast<const float*>(weight), eps, seq_len,
                                          hidden_size, add_one_to_weight, stream);
        case ZEDINFER_DTYPE_F16:
            return launch_rms_norm<half>(reinterpret_cast<half*>(output),
                                         reinterpret_cast<const half*>(input),
                                         reinterpret_cast<const half*>(weight), eps, seq_len,
                                         hidden_size, add_one_to_weight, stream);
        case ZEDINFER_DTYPE_BF16:
            return launch_rms_norm<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(output),
                                                   reinterpret_cast<const __nv_bfloat16*>(input),
                                                   reinterpret_cast<const __nv_bfloat16*>(weight), eps,
                                                   seq_len, hidden_size, add_one_to_weight, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

// In-place fused: residual = residual + input; input = norm(residual) * (weight + weight_bias)
template <typename T>
static void launch_fused_add_rms_norm(T* input, T* residual, const T* weight, float eps, size_t batch_size,
                                      size_t hidden_size, bool add_one_to_weight, cudaStream_t stream) {
    auto* w = const_cast<T*>(weight);
    const uint32_t bs = static_cast<uint32_t>(batch_size);
    const uint32_t d  = static_cast<uint32_t>(hidden_size);
    cudaError_t err;
    if (add_one_to_weight) {
        err = flashinfer::norm::GemmaFusedAddRMSNorm<T>(input, residual, w, bs, d, /*stride_input=*/d,
                                                  /*stride_residual=*/d, eps, /*enable_pdl=*/false,
                                                  stream);
    } else {
        err = flashinfer::norm::FusedAddRMSNorm<T>(input, residual, w, bs, d, /*stride_input=*/d,
                                              /*stride_residual=*/d, eps, /*enable_pdl=*/false,
                                              stream);
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("FlashInfer FusedAddRMSNorm failed: ")
                                 + cudaGetErrorString(err));
    }
}

void fused_add_rms_norm(std::byte* input, std::byte* residual, const std::byte* weight, float eps,
                        zedinferDataType_t type, size_t seq_len, size_t hidden_size, bool add_one_to_weight) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return launch_fused_add_rms_norm<float>(reinterpret_cast<float*>(input),
                                                    reinterpret_cast<float*>(residual),
                                                    reinterpret_cast<const float*>(weight), eps, seq_len,
                                                    hidden_size, add_one_to_weight, stream);
        case ZEDINFER_DTYPE_F16:
            return launch_fused_add_rms_norm<half>(reinterpret_cast<half*>(input),
                                                    reinterpret_cast<half*>(residual),
                                                    reinterpret_cast<const half*>(weight), eps, seq_len,
                                                    hidden_size, add_one_to_weight, stream);
        case ZEDINFER_DTYPE_BF16:
            return launch_fused_add_rms_norm<__nv_bfloat16>(
                reinterpret_cast<__nv_bfloat16*>(input), reinterpret_cast<__nv_bfloat16*>(residual),
                reinterpret_cast<const __nv_bfloat16*>(weight), eps, seq_len, hidden_size,
                add_one_to_weight, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
