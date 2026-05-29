#include "backend/core/context/context.hpp"
#include "backend/ops/scatter_image_embeds/nvidia/scatter_image_embeds_nvidia.cuh"
#include "utils/check.hpp"

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

namespace {

__global__ void build_mask_kernel(int* mask, const int* ids, int N, int target) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        mask[i] = (ids[i] == target) ? 1 : 0;
    }
}

template <typename T>
__global__ void scatter_kernel(T* hidden, const int* ids, const T* image_embeds, const int* prefix, int N, int H,
                                int target) {
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;
    if (n >= N) {
        return;
    }
    if (ids[n] != target) {
        return;
    }
    const int img_idx = prefix[n];
    for (int d = tid; d < H; d += blockDim.x) {
        hidden[static_cast<size_t>(n) * H + d] = image_embeds[static_cast<size_t>(img_idx) * H + d];
    }
}

template <typename T>
void launch(T* hidden, const int* ids, const T* image_embeds, int N, int H, int target, cudaStream_t stream) {
    int* mask    = nullptr;
    int* prefix  = nullptr;
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&mask), N * sizeof(int), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("scatter_image_embeds mask alloc: ") + cudaGetErrorString(err));
    }
    err = cudaMallocAsync(reinterpret_cast<void**>(&prefix), N * sizeof(int), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(mask, stream);
        throw std::runtime_error(std::string("scatter_image_embeds prefix alloc: ") + cudaGetErrorString(err));
    }

    const int  bm_block = 256;
    const dim3 bm_grid((N + bm_block - 1) / bm_block);
    build_mask_kernel<<<bm_grid, bm_block, 0, stream>>>(mask, ids, N, target);

    // CUB exclusive scan: prefix[i] = sum_{k<i} mask[k]
    size_t       tmp_bytes = 0;
    void*        tmp       = nullptr;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, mask, prefix, N, stream);
    err = cudaMallocAsync(&tmp, tmp_bytes, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(mask, stream);
        cudaFreeAsync(prefix, stream);
        throw std::runtime_error(std::string("scatter_image_embeds tmp alloc: ") + cudaGetErrorString(err));
    }
    cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, mask, prefix, N, stream);

    const int  sc_block = 128;
    const dim3 sc_grid(static_cast<unsigned int>(N));
    scatter_kernel<T><<<sc_grid, sc_block, 0, stream>>>(hidden, ids, image_embeds, prefix, N, H, target);

    cudaFreeAsync(tmp, stream);
    cudaFreeAsync(prefix, stream);
    cudaFreeAsync(mask, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("scatter_image_embeds kernel: ") + cudaGetErrorString(err));
    }
}

} // namespace

void scatter_image_embeds(std::byte* hidden, const std::byte* input_ids, const std::byte* image_embeds,
                          zedinferDataType_t type, int N, int H, int target) {
    auto       stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    const int* ids    = reinterpret_cast<const int*>(input_ids);
    switch (type) {
        case ZEDINFER_DTYPE_BF16:
            return launch<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(hidden), ids,
                                         reinterpret_cast<const __nv_bfloat16*>(image_embeds), N, H, target, stream);
        case ZEDINFER_DTYPE_F16:
            return launch<half>(reinterpret_cast<half*>(hidden), ids, reinterpret_cast<const half*>(image_embeds), N, H,
                                target, stream);
        case ZEDINFER_DTYPE_F32:
            return launch<float>(reinterpret_cast<float*>(hidden), ids, reinterpret_cast<const float*>(image_embeds),
                                 N, H, target, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
