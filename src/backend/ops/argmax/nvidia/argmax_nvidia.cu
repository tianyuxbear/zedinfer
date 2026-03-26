#include "backend/ops/argmax/nvidia/argmax_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/types.cuh"

#include <cstring>
#include <math_constants.h>

namespace zedinfer::ops::nvidia {

// Returns type-specific negative infinity for max-reduction initialization.
template <typename T> DEVICE_INLINE T get_lowest_value() {
    if constexpr (std::is_same_v<T, float>) {
        return -CUDART_INF_F;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
        return from_float<T>(-CUDART_INF_F);
    }
    // Note: Ensure T is one of the supported types.
}

// Packs value (high 32-bit) and index (low 32-bit) into a 64-bit integer for atomic CAS.
HOST_DEVICE unsigned long long pack(float val, uint32_t idx) {
#ifdef __CUDA_ARCH__
    unsigned int vb = __float_as_uint(val);
#else
    unsigned int vb;
    std::memcpy(&vb, &val, sizeof(vb)); // Portable bit-cast on host
#endif
    return (static_cast<unsigned long long>(vb) << 32) | static_cast<unsigned long long>(idx);
}

// Extracts the float value from the upper 32 bits of the packed data.
HOST_DEVICE float unpack_val(unsigned long long packed) {
    unsigned int vb = static_cast<unsigned int>(packed >> 32);
#ifdef __CUDA_ARCH__
    return __uint_as_float(vb);
#else
    float f;
    std::memcpy(&f, &vb, sizeof(f));
    return f;
#endif
}

// Extracts the index from the lower 32 bits of the packed data.
HOST_DEVICE uint32_t unpack_idx(unsigned long long packed) {
    return static_cast<uint32_t>(packed & 0xffffffffu);
}

// ----------------------------------------------------------------------
// Kernel 1: Warp-level Atomic Reduction
// High contention: Each WARP leader attempts to update global memory.
// ----------------------------------------------------------------------
template <typename T>
__global__ void argmax_reduce_warp_scope(unsigned long long* global_packed, const T* input, size_t numel,
                                         std::byte* max_idx, std::byte* max_val, zedinferDataType_t type) {
    const unsigned mask = 0xffffffffu;
    const size_t tid = threadIdx.x;
    const size_t gidx = blockIdx.x * blockDim.x + tid;
    const size_t lane_id = tid & 31;

    // -------------------------------------------------------
    // Phase 1: Thread-local Reduction (Grid-Stride Loop)
    // -------------------------------------------------------
    T thread_max_val = get_lowest_value<T>();
    uint32_t thread_max_idx = UINT32_MAX;

    for (size_t i = gidx; i < numel; i += blockDim.x * gridDim.x) {
        T val = input[i];
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = static_cast<uint32_t>(i);
        }
    }

    // -------------------------------------------------------
    // Phase 2: Warp-level Reduction (Shuffle)
    // -------------------------------------------------------
    // Convert to float for bitwise operations/comparisons
    float warp_max_val = to_float<T>(thread_max_val);
    uint32_t warp_max_idx = thread_max_idx;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float peer_val = __shfl_down_sync(mask, warp_max_val, offset);
        uint32_t peer_idx = __shfl_down_sync(mask, warp_max_idx, offset);

        // Keep local max or take peer's max
        if (peer_val > warp_max_val || (peer_val == warp_max_val && peer_idx < warp_max_idx)) {
            warp_max_val = peer_val;
            warp_max_idx = peer_idx;
        }
    }

    // -------------------------------------------------------
    // Phase 3: Global Update (Atomic CAS by Warp Leader)
    // -------------------------------------------------------
    if (lane_id == 0) {
        unsigned long long candidate_packed = pack(warp_max_val, warp_max_idx);
        unsigned long long assumed_packed = *global_packed;

        while (true) {
            float global_val = unpack_val(assumed_packed);
            uint32_t global_idx = unpack_idx(assumed_packed);

            // Optimization: Abort if global state is already better
            if (warp_max_val < global_val || (warp_max_val == global_val && warp_max_idx >= global_idx)) {
                break;
            }

            unsigned long long actual_packed = atomicCAS(global_packed, assumed_packed, candidate_packed);

            if (actual_packed == assumed_packed) {
                // Winner: Write final output to device memory
                *reinterpret_cast<int64_t*>(max_idx) = static_cast<int64_t>(warp_max_idx);

                switch (type) {
                    case ZEDINFER_DTYPE_F32:
                        *reinterpret_cast<float*>(max_val) = warp_max_val;
                        break;
                    case ZEDINFER_DTYPE_F16:
                    case ZEDINFER_DTYPE_BF16:
                        *reinterpret_cast<T*>(max_val) = from_float<T>(warp_max_val);
                        break;
                    default:
                        break;
                }
                break;                      // Done
            }
            assumed_packed = actual_packed; // Retry with updated state
        }
    }
}

// ----------------------------------------------------------------------
// Kernel 2: Block-level Atomic Reduction (via Shared Memory)
// Low contention: Only BLOCK leader attempts to update global memory.
// ----------------------------------------------------------------------
template <typename T>
__global__ void argmax_reduce_block_scope(unsigned long long* global_packed, const T* input, size_t numel,
                                          std::byte* max_idx, std::byte* max_val, zedinferDataType_t type) {
    const unsigned mask = 0xffffffffu;
    const size_t tid = threadIdx.x;
    const size_t lane_id = tid & 31;
    const size_t warp_id = tid >> 5;
    const size_t gidx = blockIdx.x * blockDim.x + tid;

    // Shared memory to store results from each warp (max 32 warps per block)
    __shared__ float smem_warp_max_vals[MAX_NUM_WARPS];
    __shared__ uint32_t smem_warp_max_idxs[MAX_NUM_WARPS];

    // -------------------------------------------------------
    // Phase 1: Thread-local Reduction
    // -------------------------------------------------------
    T thread_max_val = get_lowest_value<T>();
    uint32_t thread_max_idx = UINT32_MAX;

    for (size_t i = gidx; i < numel; i += blockDim.x * gridDim.x) {
        T val = input[i];
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = static_cast<uint32_t>(i);
        }
    }

    // -------------------------------------------------------
    // Phase 2: Warp-level Reduction
    // -------------------------------------------------------
    float warp_max_val = to_float<T>(thread_max_val);
    uint32_t warp_max_idx = thread_max_idx;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float peer_val = __shfl_down_sync(mask, warp_max_val, offset);
        uint32_t peer_idx = __shfl_down_sync(mask, warp_max_idx, offset);

        if (peer_val > warp_max_val || (peer_val == warp_max_val && peer_idx < warp_max_idx)) {
            warp_max_val = peer_val;
            warp_max_idx = peer_idx;
        }
    }

    // Warp leader stores result to shared memory
    if (lane_id == 0) {
        smem_warp_max_vals[warp_id] = warp_max_val;
        smem_warp_max_idxs[warp_id] = warp_max_idx;
    }
    __syncthreads();

    // -------------------------------------------------------
    // Phase 3: Block-level Reduction (by First Warp)
    // -------------------------------------------------------
    // Only the first 32 threads (Warp 0) participate
    if (tid < 32) {
        unsigned num_active_warps = (blockDim.x + 31) / 32;

        // Initialize block accumulator from shared memory
        float block_max_val = to_float<T>(get_lowest_value<T>());
        uint32_t block_max_idx = UINT32_MAX;

        if (tid < num_active_warps) {
            block_max_val = smem_warp_max_vals[tid];
            block_max_idx = smem_warp_max_idxs[tid];
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float peer_val = __shfl_down_sync(mask, block_max_val, offset);
            uint32_t peer_idx = __shfl_down_sync(mask, block_max_idx, offset);

            if (peer_val > block_max_val || (peer_val == block_max_val && peer_idx < block_max_idx)) {
                block_max_val = peer_val;
                block_max_idx = peer_idx;
            }
        }

        // ---------------------------------------------------
        // Phase 4: Global Update (Atomic CAS by Block Leader)
        // ---------------------------------------------------
        if (tid == 0) {
            unsigned long long candidate_packed = pack(block_max_val, block_max_idx);
            unsigned long long assumed_packed = *global_packed;

            while (true) {
                float global_val = unpack_val(assumed_packed);
                uint32_t global_idx = unpack_idx(assumed_packed);

                if (block_max_val < global_val || (block_max_val == global_val && block_max_idx >= global_idx)) {
                    break;
                }

                unsigned long long actual_packed = atomicCAS(global_packed, assumed_packed, candidate_packed);

                if (actual_packed == assumed_packed) {
                    // Winner: Write final output
                    *reinterpret_cast<int64_t*>(max_idx) = static_cast<int64_t>(block_max_idx);

                    switch (type) {
                        case ZEDINFER_DTYPE_F32:
                            *reinterpret_cast<float*>(max_val) = block_max_val;
                            break;
                        case ZEDINFER_DTYPE_F16:
                        case ZEDINFER_DTYPE_BF16:
                            *reinterpret_cast<T*>(max_val) = from_float<T>(block_max_val);
                            break;
                        default:
                            break;
                    }
                    break;
                }
                assumed_packed = actual_packed;
            }
        }
    }
}

// Pre-allocated device buffer for argmax reduction (8 bytes, allocated once)
static unsigned long long* get_packed_res_buf() {
    static unsigned long long* buf = []() {
        unsigned long long* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, sizeof(unsigned long long)));
        return p;
    }();
    return buf;
}

void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals, zedinferDataType_t type, size_t numel) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(BLOCK_SIZE);

    unsigned long long h_packed_res = pack(-std::numeric_limits<float>::infinity(), UINT32_MAX);
    unsigned long long* d_packed_res = get_packed_res_buf();
    CUDA_CHECK(cudaMemcpy(d_packed_res, &h_packed_res, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            argmax_reduce_block_scope<<<grid, block>>>(d_packed_res, reinterpret_cast<const float*>(vals), numel,
                                                       max_idx, max_val, type);
            break;
        case ZEDINFER_DTYPE_F16:
            argmax_reduce_block_scope<<<grid, block>>>(d_packed_res, reinterpret_cast<const half*>(vals), numel,
                                                       max_idx, max_val, type);
            break;
        case ZEDINFER_DTYPE_BF16:
            argmax_reduce_block_scope<<<grid, block>>>(d_packed_res, reinterpret_cast<const cuda_bfloat16*>(vals),
                                                       numel, max_idx, max_val, type);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::nvidia
