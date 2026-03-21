#include "backend/ops/self_attention/nvidia/self_attention_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "utils/nvidia/types.cuh"
#include "zedinfer.h"

#ifdef USE_CUDNN_FLASH
#include "backend/ops/self_attention/nvidia/flash_attention_cudnn.cuh"
#endif

#include <cfloat>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

// ============================================================================
// Warp/Block Reduction helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float *smem, int tid, int lane_id, int warp_id) {
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    float result = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < NUM_WARPS; ++w) result += smem[w];
        smem[0] = result;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float *smem, int tid, int lane_id, int warp_id) {
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    float result = -FLT_MAX;
    if (tid == 0) {
        for (int w = 0; w < NUM_WARPS; ++w) result = fmaxf(result, smem[w]);
        smem[0] = result;
    }
    __syncthreads();
    return smem[0];
}

// ============================================================================
// Decode Kernel (seqlen=1) — kept as custom kernel
// FlashAttention doesn't help for single-query decode (memory-bound).
// ============================================================================

constexpr int TILE_KV = 256;

template <typename T>
__global__ void self_attention_decode_kernel(
    T *__restrict__ attn_out,
    const T *__restrict__ Q,
    const T *__restrict__ K,
    const T *__restrict__ V,
    const float scale,
    const int nhead, const int nkvhead,
    const int d, const int dv, const int total_len) {

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_scores = s_q + d;
    float *s_reduce = s_scores + TILE_KV;

    const T *q_ptr = Q + h * d;
    for (int i = tid; i < d; i += blockDim.x) s_q[i] = to_float(q_ptr[i]);
    __syncthreads();

    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < total_len; kv_start += TILE_KV) {
        const int tile_len = min(TILE_KV, total_len - kv_start);
        float warp_max = -FLT_MAX;

        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const int j_global = kv_start + j_local;
                const T *k_ptr = K + j_global * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE)
                    dot += s_q[dim] * to_float(k_ptr[dim]);
                dot = warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) s_scores[j_local] = score;
            }
            if (lane_id == 0 && j_local < tile_len)
                warp_max = fmaxf(warp_max, score);
        }

        float tile_max = block_reduce_max(warp_max, s_reduce, tid, lane_id, warp_id);
        float local_exp = 0.0f;
        if (tid < tile_len) {
            float p = expf(s_scores[tid] - tile_max);
            s_scores[tid] = p;
            local_exp = p;
        }
        __syncthreads();
        float tile_sum = block_reduce_sum(local_exp, s_reduce, tid, lane_id, warp_id);

        float new_max = fmaxf(prev_max, tile_max);
        float alpha = expf(prev_max - new_max);
        float beta = expf(tile_max - new_max);
        prev_sum = prev_sum * alpha + tile_sum * beta;
        acc_out *= alpha;

        if (tid < dv) {
            float weighted_v = 0.0f;
#pragma unroll 4
            for (int j = 0; j < tile_len; ++j) {
                weighted_v += s_scores[j] * to_float(V[(kv_start + j) * nkvhead * dv + kvh * dv + tid]);
            }
            acc_out += beta * weighted_v;
        }
        prev_max = new_max;
        __syncthreads();
    }

    if (tid < dv)
        attn_out[h * dv + tid] = from_float<T>(acc_out / prev_sum);
}

// ============================================================================
// Prefill Kernel (fallback for FP32 or when cuDNN is not available)
// ============================================================================

template <typename T>
__global__ void self_attention_prefill_kernel(
    T *__restrict__ attn_out,
    const T *__restrict__ Q,
    const T *__restrict__ K,
    const T *__restrict__ V,
    const float scale,
    const int nhead, const int nkvhead,
    const int d, const int dv,
    const int seqlen, const int total_len) {

    const int query_pos = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);
    const int past_len = total_len - seqlen;
    const int causal_end = past_len + query_pos + 1;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_scores = s_q + d;
    float *s_reduce = s_scores + TILE_KV;

    const T *q_ptr = Q + query_pos * nhead * d + h * d;
    for (int i = tid; i < d; i += blockDim.x) s_q[i] = to_float(q_ptr[i]);
    __syncthreads();

    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < causal_end; kv_start += TILE_KV) {
        const int tile_len = min(TILE_KV, causal_end - kv_start);
        float warp_max = -FLT_MAX;

        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const int j_global = kv_start + j_local;
                const T *k_ptr = K + j_global * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE)
                    dot += s_q[dim] * to_float(k_ptr[dim]);
                dot = warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) s_scores[j_local] = score;
            }
            if (lane_id == 0 && j_local < tile_len)
                warp_max = fmaxf(warp_max, score);
        }

        float tile_max = block_reduce_max(warp_max, s_reduce, tid, lane_id, warp_id);
        float local_exp = 0.0f;
        if (tid < tile_len) {
            float p = expf(s_scores[tid] - tile_max);
            s_scores[tid] = p;
            local_exp = p;
        }
        __syncthreads();
        float tile_sum = block_reduce_sum(local_exp, s_reduce, tid, lane_id, warp_id);

        float new_max = fmaxf(prev_max, tile_max);
        float alpha = expf(prev_max - new_max);
        float beta = expf(tile_max - new_max);
        prev_sum = prev_sum * alpha + tile_sum * beta;
        acc_out *= alpha;

        if (tid < dv) {
            float weighted_v = 0.0f;
#pragma unroll 4
            for (int j = 0; j < tile_len; ++j)
                weighted_v += s_scores[j] * to_float(V[(kv_start + j) * nkvhead * dv + kvh * dv + tid]);
            acc_out += beta * weighted_v;
        }
        prev_max = new_max;
        __syncthreads();
    }

    if (tid < dv)
        attn_out[query_pos * nhead * dv + h * dv + tid] = from_float<T>(acc_out / prev_sum);
}

// ============================================================================
// Dispatch
// ============================================================================

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, float scale, zedinferDataType_t type,
                    size_t seqlen, size_t nhead, size_t dv, size_t total_len,
                    size_t nkvhead, size_t d) {

    // Decode: always use custom kernel (memory-bound, FlashAttention doesn't help)
    if (likely(seqlen == 1)) {
        dim3 block(BLOCK_SIZE);
        dim3 grid(nhead);
        size_t smem_size = (d + TILE_KV + NUM_WARPS) * sizeof(float);

        switch (type) {
        case ZEDINFER_DTYPE_F32:
            return self_attention_decode_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                scale, nhead, nkvhead, d, dv, total_len);
        case ZEDINFER_DTYPE_F16:
            return self_attention_decode_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<half *>(attn_val), reinterpret_cast<const half *>(q),
                reinterpret_cast<const half *>(k), reinterpret_cast<const half *>(v),
                scale, nhead, nkvhead, d, dv, total_len);
        case ZEDINFER_DTYPE_BF16:
            return self_attention_decode_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<cuda_bfloat16 *>(attn_val), reinterpret_cast<const cuda_bfloat16 *>(q),
                reinterpret_cast<const cuda_bfloat16 *>(k), reinterpret_cast<const cuda_bfloat16 *>(v),
                scale, nhead, nkvhead, d, dv, total_len);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
        return;
    }

    // Prefill: try cuDNN FlashAttention for FP16/BF16 (2-4x faster)
    // Falls back to custom kernel if cuDNN doesn't support this GPU/config
#ifdef USE_CUDNN_FLASH
    if (type == ZEDINFER_DTYPE_F16 || type == ZEDINFER_DTYPE_BF16) {
        if (cudnn_flash::flash_attention_prefill(
                attn_val, q, k, v, scale, type,
                seqlen, nhead, d, total_len, nkvhead)) {
            return;  // cuDNN succeeded
        }
        // cuDNN unsupported on this GPU — fall through to custom kernel
    }
#endif

    // Prefill fallback: custom kernel (FP32, unsupported GPU, or no cuDNN)
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid(seqlen, nhead);
        size_t smem_size = (d + TILE_KV + NUM_WARPS) * sizeof(float);

        switch (type) {
        case ZEDINFER_DTYPE_F32:
            return self_attention_prefill_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                scale, nhead, nkvhead, d, dv, seqlen, total_len);
        case ZEDINFER_DTYPE_F16:
            return self_attention_prefill_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<half *>(attn_val), reinterpret_cast<const half *>(q),
                reinterpret_cast<const half *>(k), reinterpret_cast<const half *>(v),
                scale, nhead, nkvhead, d, dv, seqlen, total_len);
        case ZEDINFER_DTYPE_BF16:
            return self_attention_prefill_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<cuda_bfloat16 *>(attn_val), reinterpret_cast<const cuda_bfloat16 *>(q),
                reinterpret_cast<const cuda_bfloat16 *>(k), reinterpret_cast<const cuda_bfloat16 *>(v),
                scale, nhead, nkvhead, d, dv, seqlen, total_len);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}

} // namespace zedinfer::ops::nvidia
