#include "backend/core/context/context.hpp"
#include "backend/ops/scatter_add_rows/nvidia/scatter_add_rows_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "utils/nvidia/memory.cuh"

#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

// One block per src row r; the block writes to out[indices[r], :] with weight weights[r].
// Within a kernel invocation, indices[*] are guaranteed distinct (one expert's tokens are
// distinct), so there is no intra-kernel write race. Across invocations (different experts
// targeting overlapping tokens), stream-ordering serializes the accumulations.
__global__ void scatter_add_rows_kernel_f32(float* out, const float* src, const int* indices, const float* weights,
                                            size_t row_elements) {
    const size_t r = blockIdx.x;
    const size_t out_row = static_cast<size_t>(indices[r]);
    const float w = weights[r];
    const float* sp = src + r * row_elements;
    float* op = out + out_row * row_elements;
    for (size_t c = threadIdx.x; c < row_elements; c += blockDim.x) {
        op[c] = op[c] + w * sp[c];
    }
}

__global__ void scatter_add_rows_kernel_f16(half* out, const half* src, const int* indices, const float* weights,
                                            size_t row_elements) {
    const size_t r = blockIdx.x;
    const size_t out_row = static_cast<size_t>(indices[r]);
    const float w = weights[r];
    const half* sp = src + r * row_elements;
    half* op = out + out_row * row_elements;
    for (size_t c = threadIdx.x; c < row_elements; c += blockDim.x) {
        op[c] = __float2half(__half2float(op[c]) + w * __half2float(sp[c]));
    }
}

__global__ void scatter_add_rows_kernel_bf16(cuda_bfloat16* out, const cuda_bfloat16* src, const int* indices,
                                             const float* weights, size_t row_elements) {
    const size_t r = blockIdx.x;
    const size_t out_row = static_cast<size_t>(indices[r]);
    const float w = weights[r];
    const cuda_bfloat16* sp = src + r * row_elements;
    cuda_bfloat16* op = out + out_row * row_elements;
    for (size_t c = threadIdx.x; c < row_elements; c += blockDim.x) {
        op[c] = __float2bfloat16(__bfloat162float(op[c]) + w * __bfloat162float(sp[c]));
    }
}

} // namespace

void scatter_add_rows(std::byte* out, const std::byte* src, const std::int32_t* indices_device,
                      const float* weights_device, zedinferDataType_t type, std::size_t num_rows,
                      std::size_t row_elements) {
    if (num_rows == 0) {
        return;
    }
    auto stream = reinterpret_cast<cudaStream_t>(zedinfer::core::context().runtime().stream());
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_rows);

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            scatter_add_rows_kernel_f32<<<grid, block, 0, stream>>>(
                reinterpret_cast<float*>(out), reinterpret_cast<const float*>(src),
                reinterpret_cast<const int*>(indices_device), weights_device, row_elements);
            break;
        case ZEDINFER_DTYPE_F16:
            scatter_add_rows_kernel_f16<<<grid, block, 0, stream>>>(
                reinterpret_cast<half*>(out), reinterpret_cast<const half*>(src),
                reinterpret_cast<const int*>(indices_device), weights_device, row_elements);
            break;
        case ZEDINFER_DTYPE_BF16:
            scatter_add_rows_kernel_bf16<<<grid, block, 0, stream>>>(
                reinterpret_cast<cuda_bfloat16*>(out), reinterpret_cast<const cuda_bfloat16*>(src),
                reinterpret_cast<const int*>(indices_device), weights_device, row_elements);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
