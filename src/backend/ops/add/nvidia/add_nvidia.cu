#include "backend/ops/add/nvidia/add_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/memory.cuh"

namespace zedinfer::ops::nvidia {

template <typename T> __global__ void vector_add_scalar_kernel(T* c, const T* a, const T* b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if constexpr (std::is_same_v<T, float>) {
            c[idx] = a[idx] + b[idx];
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            c[idx] = __hadd(a[idx], b[idx]); // Intrinsics for half/bf16
        }
    }
}

template <typename T> __global__ void vector_add_packed_kernel(T* c, const T* a, const T* b, size_t numel) {
    // Determine elements processed per thread (e.g., 4 for float, 8 for half/bf16)
    constexpr int PackSize = PackedTraits<T>::size;
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * PackSize;

    if (idx + PackSize <= numel) {
        // 128-bit vectorized load
        float4 av = load_128b(&a[idx]);
        float4 bv = load_128b(&b[idx]);
        float4 cv;

        if constexpr (std::is_same_v<T, float>) {
            // Manual SIMD for FP32
            cv.x = av.x + bv.x;
            cv.y = av.y + bv.y;
            cv.z = av.z + bv.z;
            cv.w = av.w + bv.w;
        } else if constexpr (std::is_same_v<T, half>) {
            // Reinterpret 128-bit as 4x half2
            auto* ah = reinterpret_cast<half2*>(&av);
            auto* bh = reinterpret_cast<half2*>(&bv);
            auto* ch = reinterpret_cast<half2*>(&cv);

#pragma unroll
            for (int i = 0; i < 4; ++i) { ch[i] = __hadd2(ah[i], bh[i]); }
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Reinterpret 128-bit as 4x nv_bfloat162
            auto* ab = reinterpret_cast<nv_bfloat162*>(&av);
            auto* bb = reinterpret_cast<nv_bfloat162*>(&bv);
            auto* cb = reinterpret_cast<nv_bfloat162*>(&cv);

#pragma unroll
            for (int i = 0; i < 4; ++i) { cb[i] = __hadd2(ab[i], bb[i]); }
        }
        store_128b(&c[idx], cv);
    } else if (idx < numel) {
        // Handle scalar tail
        for (size_t i = idx; i < numel; ++i) {
            if constexpr (std::is_same_v<T, float>) {
                c[i] = a[i] + b[i];
            } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
                c[i] = __hadd(a[i], b[i]);
            }
        }
    }
}

template <typename T> void launch_vector_add(T* c, const T* a, const T* b, size_t numel) {
    constexpr size_t ElemsPerThread = PackedTraits<T>::size;

    dim3 block(BLOCK_SIZE);
    dim3 grid(div_ceil(numel, BLOCK_SIZE * ElemsPerThread));

    vector_add_packed_kernel<<<grid, block>>>(c, a, b, numel);
}

void add(std::byte* c, const std::byte* a, const std::byte* b, zedinferDataType_t type, size_t numel) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return launch_vector_add(reinterpret_cast<float*>(c), reinterpret_cast<const float*>(a),
                                     reinterpret_cast<const float*>(b), numel);
        case ZEDINFER_DTYPE_F16:
            return launch_vector_add(reinterpret_cast<half*>(c), reinterpret_cast<const half*>(a),
                                     reinterpret_cast<const half*>(b), numel);
        case ZEDINFER_DTYPE_BF16:
            return launch_vector_add(reinterpret_cast<cuda_bfloat16*>(c), reinterpret_cast<const cuda_bfloat16*>(a),
                                     reinterpret_cast<const cuda_bfloat16*>(b), numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia