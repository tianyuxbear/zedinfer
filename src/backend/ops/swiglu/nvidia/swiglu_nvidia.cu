#include "backend/ops/swiglu/nvidia/swiglu_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/memory.cuh"

namespace zedinfer::ops::nvidia {

template <typename T> __global__ void swiglu_scalar_kernel(T* output, const T* gate, const T* up, const size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float gate_val = to_float(gate[idx]);
        float up_val = to_float(up[idx]);
        output[idx] = from_float<T>(up_val * silu(gate_val));
    }
}

template <typename T>
__global__ void swiglu_packed_kernel(T* __restrict__ output, const T* __restrict__ gate, const T* __restrict__ up,
                                     size_t numel) {
    // 1. Setup: Determine elements per thread based on type
    // (4 for float, 8 for half/bf16)
    constexpr int PackSize = PackedTraits<T>::size;
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * PackSize;

    // 2. Vectorized Path (128-bit processing)
    if (idx + PackSize <= numel) {
        // Unified 128-bit Loads
        float4 gate_vec = load_128b(&gate[idx]);
        float4 up_vec = load_128b(&up[idx]);
        float4 res_vec;

        // 3. Math Dispatch based on type
        if constexpr (std::is_same_v<T, float>) {
            // ---------------------------------------------------
            // Float32: Manual SIMD unrolling
            // ---------------------------------------------------
            res_vec.x = up_vec.x * silu(gate_vec.x);
            res_vec.y = up_vec.y * silu(gate_vec.y);
            res_vec.z = up_vec.z * silu(gate_vec.z);
            res_vec.w = up_vec.w * silu(gate_vec.w);

        } else if constexpr (std::is_same_v<T, half>) {
            // ---------------------------------------------------
            // Half: Reinterpret as 4x half2
            // ---------------------------------------------------
            const half2* gate_h2 = reinterpret_cast<const half2*>(&gate_vec);
            const half2* up_h2 = reinterpret_cast<const half2*>(&up_vec);
            half2* res_h2 = reinterpret_cast<half2*>(&res_vec);

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Convert to float2 for high-precision SiLU math
                float2 gate_f2 = __half22float2(gate_h2[i]);
                float2 up_f2 = __half22float2(up_h2[i]);

                float2 res_f2;
                res_f2.x = up_f2.x * silu(gate_f2.x);
                res_f2.y = up_f2.y * silu(gate_f2.y);

                res_h2[i] = __float22half2_rn(res_f2);
            }

        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // ---------------------------------------------------
            // BFloat16: Reinterpret as 4x cuda_bfloat162
            // ---------------------------------------------------
            const cuda_bfloat162* gate_bf2 = reinterpret_cast<const cuda_bfloat162*>(&gate_vec);
            const cuda_bfloat162* up_bf2 = reinterpret_cast<const cuda_bfloat162*>(&up_vec);
            cuda_bfloat162* res_bf2 = reinterpret_cast<cuda_bfloat162*>(&res_vec);

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Convert to float2
                float2 gate_f2 = __bfloat1622float2(gate_bf2[i]);
                float2 up_f2 = __bfloat1622float2(up_bf2[i]);

                float2 res_f2;
                res_f2.x = up_f2.x * silu(gate_f2.x);
                res_f2.y = up_f2.y * silu(gate_f2.y);

                res_bf2[i] = __float22bfloat162_rn(res_f2);
            }
        }

        // Unified 128-bit Store
        store_128b(&output[idx], res_vec);

    }
    // 4. Scalar Tail Handling
    else if (idx < numel) {
        for (size_t i = idx; i < numel; ++i) {
            float gate_val = to_float(gate[i]);
            float up_val = to_float(up[i]);
            output[i] = from_float<T>(up_val * silu(gate_val));
        }
    }
}

// ----------------------------------------------------------------------
// Host Dispatcher
// ----------------------------------------------------------------------
template <typename T> void launch_swiglu(T* output, const T* gate, const T* up, const size_t numel) {
    constexpr size_t PackSize = PackedTraits<T>::size; // 4 for float, 8 for half/bf16

    dim3 block(BLOCK_SIZE);
    dim3 grid(div_ceil(numel, BLOCK_SIZE * PackSize));

    swiglu_packed_kernel<<<grid, block>>>(output, gate, up, numel);
}

// Public API Interface
void swiglu(std::byte* output, const std::byte* gate, const std::byte* up, zedinferDataType_t type, size_t numel) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            launch_swiglu(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(gate),
                          reinterpret_cast<const float*>(up), numel);
            break;
        case ZEDINFER_DTYPE_F16:
            launch_swiglu(reinterpret_cast<half*>(output), reinterpret_cast<const half*>(gate),
                          reinterpret_cast<const half*>(up), numel);
            break;
        case ZEDINFER_DTYPE_BF16:
            launch_swiglu(reinterpret_cast<cuda_bfloat16*>(output), reinterpret_cast<const cuda_bfloat16*>(gate),
                          reinterpret_cast<const cuda_bfloat16*>(up), numel);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
