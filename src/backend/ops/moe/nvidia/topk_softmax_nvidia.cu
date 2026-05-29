// GPU implementation of MoE router top-K softmax.
//
// One CTA per token row; block size is rounded up to the next power of two ≥
// num_experts (capped at 1024). Each thread owns one expert position; threads
// beyond num_experts hold sentinel -inf so they never win selection.
//
// Algorithm per row:
//   1. Each thread loads its logit, converts to fp32.
//   2. Block-reduce max (numerical stability anchor).
//   3. Each thread computes exp(logit - max), block-reduce sum.
//   4. Each thread divides by sum -> per-expert softmax probability.
//   5. Top-K selection: K rounds of block-reduce max over a thread-local prob
//      register, with the winning thread masking itself to -inf for the next
//      round. K rounds × O(log BLOCK_DIM) work; for K=8 and BLOCK_DIM=256
//      this is well under a microsecond.
//   6. Optional renormalization of the K selected weights to sum to 1.
//
// The kernel only allocates O(1) shared memory (scalars for reductions); the
// per-expert probabilities live in registers. NUM_EXPERTS templated to let
// the compiler hoist comparisons against a compile-time bound when the model
// uses one of the common router widths (128 / 256 / 512). For now we ship the
// single specialization at BLOCK_THREADS=256 (Qwen3.5 has num_experts=256
// exactly); other widths fall back to the runtime path.

#include "backend/core/context/context.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "utils/check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>

namespace zedinfer::ops::moe {

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float to_float_bf16(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(float));
    return f;
}

__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ float to_float(half v) {
    return __half2float(v);
}
__device__ __forceinline__ float to_float(float v) {
    return v;
}

// Block reduction helpers using two-stage warp shuffle + shared memory.
// BLOCK_THREADS is assumed ≤ 1024 (32 warps max).
template <int BLOCK_THREADS> __device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    static_assert(BLOCK_THREADS <= 1024, "block reduce max supports up to 1024 threads");
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    constexpr int kNumWarps = (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, val, off);
        val = fmaxf(val, other);
    }
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        val = (lane < kNumWarps) ? smem[lane] : -FLT_MAX;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, val, off);
            val = fmaxf(val, other);
        }
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    return smem[0];
}

template <int BLOCK_THREADS> __device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    static_assert(BLOCK_THREADS <= 1024, "block reduce sum supports up to 1024 threads");
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    constexpr int kNumWarps = (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) { val += __shfl_xor_sync(0xffffffffu, val, off); }
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        val = (lane < kNumWarps) ? smem[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) { val += __shfl_xor_sync(0xffffffffu, val, off); }
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    return smem[0];
}

// Find both max value and argmax id across the block. Uses the same shuffle
// pattern as block_reduce_max but threads the index alongside, picking the
// smaller index on ties (matches CPU std::partial_sort's stable order).
template <int BLOCK_THREADS>
__device__ __forceinline__ void block_argmax(float val, int idx, float& out_val, int& out_idx, float* smem_val,
                                             int* smem_idx) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    constexpr int kNumWarps = (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;

#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffffu, val, off);
        int other_idx = __shfl_xor_sync(0xffffffffu, idx, off);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    if (lane == 0) {
        smem_val[warp] = val;
        smem_idx[warp] = idx;
    }
    __syncthreads();
    if (warp == 0) {
        if (lane < kNumWarps) {
            val = smem_val[lane];
            idx = smem_idx[lane];
        } else {
            val = -FLT_MAX;
            idx = -1;
        }
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffffu, val, off);
            int other_idx = __shfl_xor_sync(0xffffffffu, idx, off);
            if (other_val > val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (lane == 0) {
            smem_val[0] = val;
            smem_idx[0] = idx;
        }
    }
    __syncthreads();
    out_val = smem_val[0];
    out_idx = smem_idx[0];
}

template <int BLOCK_THREADS, int MAX_TOP_K, typename DType>
__global__ void topk_softmax_kernel(const DType* __restrict__ logits, int32_t* __restrict__ expert_ids,
                                    float* __restrict__ expert_weights, int num_experts, int top_k,
                                    bool norm_topk_prob) {
    constexpr int kNumWarps = (BLOCK_THREADS + kWarpSize - 1) / kWarpSize;
    __shared__ float smem_red[kNumWarps]; // shared by block_reduce_max / sum
    __shared__ int smem_red_idx[kNumWarps];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const DType* row_logits = logits + row * num_experts;

    // 1. Load + convert. Threads beyond num_experts hold -inf so they
    // never win argmax.
    float my_logit = (tid < num_experts) ? to_float(row_logits[tid]) : -FLT_MAX;

    // 2. Block-reduce max for numerical stability.
    float max_logit = block_reduce_max<BLOCK_THREADS>(my_logit, smem_red);

    // 3. exp(logit - max).
    float my_exp = (tid < num_experts) ? __expf(my_logit - max_logit) : 0.0f;

    // 4. Block-reduce sum.
    float sum_exp = block_reduce_sum<BLOCK_THREADS>(my_exp, smem_red);
    float inv_sum = 1.0f / sum_exp;

    // 5. Per-expert prob (in register only, not stored to smem — we mask via
    //    a local copy `mask_val` so other threads still see their own).
    float my_prob = my_exp * inv_sum;
    float mask_val = my_prob;

    // 6. K rounds of (find argmax, record, mask).
    float topk_sum = 0.0f;
#pragma unroll 1
    for (int k = 0; k < MAX_TOP_K; ++k) {
        if (k >= top_k) {
            break;
        }
        float winner_val;
        int winner_idx;
        block_argmax<BLOCK_THREADS>(mask_val, tid, winner_val, winner_idx, smem_red, smem_red_idx);

        if (tid == 0) {
            expert_ids[row * top_k + k] = winner_idx;
            expert_weights[row * top_k + k] = winner_val;
        }
        topk_sum += winner_val;

        // Mask this thread out for subsequent rounds. Each thread keeps its
        // own register; only the winning thread updates.
        if (tid == winner_idx) {
            mask_val = -FLT_MAX;
        }
    }

    // 7. Optional renormalization of the K weights to sum to 1.
    if (norm_topk_prob && tid == 0 && topk_sum > 0.0f) {
        const float inv_topk = 1.0f / topk_sum;
        for (int k = 0; k < top_k; ++k) { expert_weights[row * top_k + k] *= inv_topk; }
    }
}

template <typename DType>
void launch_topk_softmax(const DType* logits, int32_t* expert_ids, float* expert_weights, int N, int num_experts,
                         int top_k, bool norm_topk_prob, cudaStream_t stream) {
    // Round block size up to the next warp multiple, capped at 1024.
    auto round_to_block = [](int n) -> int {
        if (n <= 32) {
            return 32;
        }
        if (n <= 64) {
            return 64;
        }
        if (n <= 128) {
            return 128;
        }
        if (n <= 256) {
            return 256;
        }
        if (n <= 512) {
            return 512;
        }
        return 1024;
    };
    const int block = round_to_block(num_experts);
    dim3 grid(N);

    // Template specialize on (block_size, max_top_k=16) — covers all known
    // Qwen3-MoE / Qwen3.5 / DeepSeek-V3 router widths.
    if (block == 256) {
        topk_softmax_kernel<256, 16, DType>
            <<<grid, dim3(256), 0, stream>>>(logits, expert_ids, expert_weights, num_experts, top_k, norm_topk_prob);
    } else if (block == 128) {
        topk_softmax_kernel<128, 16, DType>
            <<<grid, dim3(128), 0, stream>>>(logits, expert_ids, expert_weights, num_experts, top_k, norm_topk_prob);
    } else if (block == 512) {
        topk_softmax_kernel<512, 16, DType>
            <<<grid, dim3(512), 0, stream>>>(logits, expert_ids, expert_weights, num_experts, top_k, norm_topk_prob);
    } else if (block == 1024) {
        topk_softmax_kernel<1024, 16, DType>
            <<<grid, dim3(1024), 0, stream>>>(logits, expert_ids, expert_weights, num_experts, top_k, norm_topk_prob);
    } else if (block == 64) {
        topk_softmax_kernel<64, 16, DType>
            <<<grid, dim3(64), 0, stream>>>(logits, expert_ids, expert_weights, num_experts, top_k, norm_topk_prob);
    } else {
        topk_softmax_kernel<32, 16, DType>
            <<<grid, dim3(32), 0, stream>>>(logits, expert_ids, expert_weights, num_experts, top_k, norm_topk_prob);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("topk_softmax_kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace

void topk_softmax_gpu(tensor_t expert_ids, tensor_t expert_weights, tensor_t logits, size_t top_k,
                      bool norm_topk_prob) {
    if (top_k > 16) {
        throw std::runtime_error("topk_softmax_gpu: top_k > 16 not supported (got " + std::to_string(top_k) + ")");
    }
    if (logits->deviceType() != ZEDINFER_DEVICE_NVIDIA) {
        throw std::runtime_error("topk_softmax_gpu: logits must be on NVIDIA device");
    }
    if (expert_ids->dtype() != ZEDINFER_DTYPE_I32) {
        throw std::runtime_error("topk_softmax_gpu: expert_ids must be int32");
    }
    if (expert_weights->dtype() != ZEDINFER_DTYPE_F32) {
        throw std::runtime_error("topk_softmax_gpu: expert_weights must be float32");
    }
    const size_t N = logits->shape()[0];
    const size_t num_experts = logits->shape()[1];

    zedinfer::core::context().setDevice(logits->deviceType(), logits->deviceId());
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    auto* eid_p = reinterpret_cast<int32_t*>(expert_ids->data());
    auto* ew_p = reinterpret_cast<float*>(expert_weights->data());

    switch (logits->dtype()) {
        case ZEDINFER_DTYPE_BF16:
            return launch_topk_softmax<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(logits->data()), eid_p,
                                                      ew_p, static_cast<int>(N), static_cast<int>(num_experts),
                                                      static_cast<int>(top_k), norm_topk_prob, stream);
        case ZEDINFER_DTYPE_F16:
            return launch_topk_softmax<half>(reinterpret_cast<const half*>(logits->data()), eid_p, ew_p,
                                             static_cast<int>(N), static_cast<int>(num_experts),
                                             static_cast<int>(top_k), norm_topk_prob, stream);
        case ZEDINFER_DTYPE_F32:
            return launch_topk_softmax<float>(reinterpret_cast<const float*>(logits->data()), eid_p, ew_p,
                                              static_cast<int>(N), static_cast<int>(num_experts),
                                              static_cast<int>(top_k), norm_topk_prob, stream);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(logits->dtype());
    }
}

} // namespace zedinfer::ops::moe
