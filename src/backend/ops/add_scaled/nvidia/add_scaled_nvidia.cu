#include "backend/core/context/context.hpp"
#include "backend/ops/add_scaled/nvidia/add_scaled_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "utils/nvidia/memory.cuh"

#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

__global__ void add_scaled_kernel_f32(float* out, const float* a, float alpha, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = out[idx] + alpha * a[idx];
    }
}

__global__ void add_scaled_kernel_f16(half* out, const half* a, float alpha, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __float2half(__half2float(out[idx]) + alpha * __half2float(a[idx]));
    }
}

__global__ void add_scaled_kernel_bf16(cuda_bfloat16* out, const cuda_bfloat16* a, float alpha, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + alpha * __bfloat162float(a[idx]));
    }
}

} // namespace

void add_scaled(std::byte* out, const std::byte* a, float alpha, zedinferDataType_t type, size_t numel) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_ceil(numel, static_cast<size_t>(BLOCK_SIZE)));
    auto stream = reinterpret_cast<cudaStream_t>(zedinfer::core::context().runtime().stream());

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            add_scaled_kernel_f32<<<grid, block, 0, stream>>>(reinterpret_cast<float*>(out),
                                                              reinterpret_cast<const float*>(a), alpha, numel);
            break;
        case ZEDINFER_DTYPE_F16:
            add_scaled_kernel_f16<<<grid, block, 0, stream>>>(reinterpret_cast<half*>(out),
                                                              reinterpret_cast<const half*>(a), alpha, numel);
            break;
        case ZEDINFER_DTYPE_BF16:
            add_scaled_kernel_bf16<<<grid, block, 0, stream>>>(reinterpret_cast<cuda_bfloat16*>(out),
                                                               reinterpret_cast<const cuda_bfloat16*>(a), alpha, numel);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
