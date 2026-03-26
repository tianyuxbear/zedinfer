#include "backend/core/context/context.hpp"
#include "backend/ops/linear/nvidia/permute.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

template <typename T>
__global__ void permute_cols_kernel(const T* __restrict__ input, const int32_t* __restrict__ indices,
                                    T* __restrict__ output, int total_elements, int K) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_elements) {
        return;
    }

    const int row = tid / K;
    const int col = tid % K;
    output[tid] = input[row * K + indices[col]];
}

tensor_t permute_cols(tensor_t input, tensor_t indices) {
    if (!input || !indices) {
        throw std::runtime_error("permute_cols requires input and indices.");
    }
    if (input->shape().size() != 2 || indices->shape().size() != 1) {
        throw std::runtime_error("permute_cols expects input [M, K] and indices [K].");
    }
    if (indices->dtype() != ZEDINFER_DTYPE_I32) {
        throw std::runtime_error("permute_cols indices must be INT32.");
    }
    if (input->dim(1) != indices->dim(0)) {
        throw std::runtime_error("permute_cols indices length must match input columns.");
    }

    const int M = static_cast<int>(input->dim(0));
    const int K = static_cast<int>(input->dim(1));
    const int total_elements = M * K;

    auto output = Tensor::create(input->shape(), input->dtype(), ZEDINFER_DEVICE_NVIDIA, input->deviceId());

    auto& runtime = zedinfer::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    constexpr int kBlockSize = 256;
    const int grid_size = (total_elements + kBlockSize - 1) / kBlockSize;
    const int32_t* idx_ptr = reinterpret_cast<const int32_t*>(indices->data());

    switch (input->dtype()) {
        case ZEDINFER_DTYPE_F32:
            permute_cols_kernel<float>
                <<<grid_size, kBlockSize, 0, stream>>>(reinterpret_cast<const float*>(input->data()), idx_ptr,
                                                       reinterpret_cast<float*>(output->data()), total_elements, K);
            break;
        case ZEDINFER_DTYPE_F16:
            permute_cols_kernel<half>
                <<<grid_size, kBlockSize, 0, stream>>>(reinterpret_cast<const half*>(input->data()), idx_ptr,
                                                       reinterpret_cast<half*>(output->data()), total_elements, K);
            break;
        case ZEDINFER_DTYPE_BF16:
            permute_cols_kernel<__nv_bfloat16><<<grid_size, kBlockSize, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input->data()), idx_ptr,
                reinterpret_cast<__nv_bfloat16*>(output->data()), total_elements, K);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for CUDA permute_cols.");
    }

    return output;
}

} // namespace zedinfer::ops::nvidia
