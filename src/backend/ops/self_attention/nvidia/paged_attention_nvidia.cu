#include "backend/ops/self_attention/nvidia/paged_attention_nvidia.cuh"
#include "utils/check.hpp"
#include "utils/nvidia/common.cuh"
#include "utils/nvidia/types.cuh"

#include <cfloat>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

// ============================================================================
// Warp/Block Reduction helpers (same as self_attention_nvidia.cu)
// ============================================================================

__device__ __forceinline__ float pa_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float pa_warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float pa_block_reduce_sum(float val, float *smem, int tid, int lane_id, int warp_id) {
    val = pa_warp_reduce_sum(val);
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

__device__ __forceinline__ float pa_block_reduce_max(float val, float *smem, int tid, int lane_id, int warp_id) {
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
// Paged Attention Decode Kernel
//
// vLLM-style: one CUDA block per attention head. K/V accessed via block table.
// Uses online softmax with tiled KV iteration (same algorithm as the non-paged
// decode kernel, but with block-table-indexed memory access).
//
// Key difference from non-paged: instead of K[j * nkvhead * d + kvh * d + dim],
// we compute the physical address via:
//   physical_token = block_table[j / block_size] * block_size + j % block_size
//   K[physical_token * nkvhead * d + kvh * d + dim]
// ============================================================================

constexpr int PA_TILE_KV = 256;

template <typename T>
__global__ void paged_attention_decode_kernel(
    T *__restrict__ attn_out,
    const T *__restrict__ Q,
    const T *__restrict__ pool_base,
    const int *__restrict__ k_block_table,
    const int *__restrict__ v_block_table,
    const int seq_len,
    const float scale,
    const int nhead, const int nkvhead,
    const int d, const int block_size) {

    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_scores = s_q + d;
    float *s_reduce = s_scores + PA_TILE_KV;

    // Load Q into shared memory
    const T *q_ptr = Q + h * d;
    for (int i = tid; i < d; i += blockDim.x) s_q[i] = to_float(q_ptr[i]);
    __syncthreads();

    // V aggregation: all threads participate, strided over dv dimensions
    const int dv = d; // head_dim == d for standard transformers
    const int dv_idx = tid % dv;
    const int dv_group = blockDim.x / dv;
    const int dv_rank = tid / dv;

    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < seq_len; kv_start += PA_TILE_KV) {
        const int tile_len = min(PA_TILE_KV, seq_len - kv_start);
        float warp_max = -FLT_MAX;

        // Q*K scoring with block-table-indexed K access
        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const int j_global = kv_start + j_local;
                // Block-table lookup: logical token -> physical token
                const int block_idx = j_global / block_size;
                const int block_offset = j_global % block_size;
                const int k_physical = k_block_table[block_idx] * block_size + block_offset;

                const T *k_ptr = pool_base + k_physical * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE)
                    dot += s_q[dim] * to_float(k_ptr[dim]);
                dot = pa_warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) s_scores[j_local] = score;
            }
            if (lane_id == 0 && j_local < tile_len)
                warp_max = fmaxf(warp_max, score);
        }

        float tile_max = pa_block_reduce_max(warp_max, s_reduce, tid, lane_id, warp_id);
        float local_exp = 0.0f;
        if (tid < tile_len) {
            float p = expf(s_scores[tid] - tile_max);
            s_scores[tid] = p;
            local_exp = p;
        }
        __syncthreads();
        float tile_sum = pa_block_reduce_sum(local_exp, s_reduce, tid, lane_id, warp_id);

        float new_max = fmaxf(prev_max, tile_max);
        float alpha = expf(prev_max - new_max);
        float beta = expf(tile_max - new_max);
        prev_sum = prev_sum * alpha + tile_sum * beta;
        acc_out *= alpha;

        // V aggregation with block-table-indexed V access
        {
            float weighted_v = 0.0f;
#pragma unroll 4
            for (int j = dv_rank; j < tile_len; j += dv_group) {
                const int j_global = kv_start + j;
                const int block_idx = j_global / block_size;
                const int block_offset = j_global % block_size;
                const int v_physical = v_block_table[block_idx] * block_size + block_offset;

                weighted_v += s_scores[j] * to_float(pool_base[v_physical * nkvhead * dv + kvh * dv + dv_idx]);
            }
            acc_out += beta * weighted_v;
        }

        prev_max = new_max;
        __syncthreads();
    }

    // Reduce across threads that share the same dv_idx
    if (dv_group > 1) {
        s_scores[tid] = acc_out;
        __syncthreads();
        if (dv_rank == 0) {
            float sum = s_scores[tid];
            for (int g = 1; g < dv_group; ++g) {
                sum += s_scores[g * dv + dv_idx];
            }
            attn_out[h * dv + dv_idx] = from_float<T>(sum / prev_sum);
        }
    } else {
        if (tid < dv)
            attn_out[h * dv + tid] = from_float<T>(acc_out / prev_sum);
    }
}

// ============================================================================
// Batched Paged Attention Decode Kernel
//
// Grid = (num_requests, nhead). One CUDA block per (request, head).
// Each request has its own block table and seq_len.
// ============================================================================

template <typename T>
__global__ void paged_attention_decode_batched_kernel(
    T *__restrict__ attn_out,        // [num_reqs, nhead, head_dim]
    const T *__restrict__ Q,          // [num_reqs, nhead, head_dim]
    const T *__restrict__ pool_base,
    const int *__restrict__ k_block_tables,  // [num_reqs, max_blocks_per_seq]
    const int *__restrict__ v_block_tables,
    const int *__restrict__ seq_lens,        // [num_reqs]
    const float scale,
    const int nhead, const int nkvhead,
    const int d, const int block_size,
    const int max_blocks_per_seq) {

    const int req_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);
    const int seq_len = seq_lens[req_idx];

    // Per-request block tables
    const int *k_bt = k_block_tables + req_idx * max_blocks_per_seq;
    const int *v_bt = v_block_tables + req_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_scores = s_q + d;
    float *s_reduce = s_scores + PA_TILE_KV;

    // Load this request's Q into shared memory
    const T *q_ptr = Q + req_idx * nhead * d + h * d;
    for (int i = tid; i < d; i += blockDim.x) s_q[i] = to_float(q_ptr[i]);
    __syncthreads();

    const int dv = d;
    const int dv_idx = tid % dv;
    const int dv_group = blockDim.x / dv;
    const int dv_rank = tid / dv;

    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < seq_len; kv_start += PA_TILE_KV) {
        const int tile_len = min(PA_TILE_KV, seq_len - kv_start);
        float warp_max = -FLT_MAX;

        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const int j_global = kv_start + j_local;
                const int blk = j_global / block_size;
                const int off = j_global % block_size;
                const int k_phys = k_bt[blk] * block_size + off;

                const T *k_ptr = pool_base + k_phys * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE)
                    dot += s_q[dim] * to_float(k_ptr[dim]);
                dot = pa_warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) s_scores[j_local] = score;
            }
            if (lane_id == 0 && j_local < tile_len)
                warp_max = fmaxf(warp_max, score);
        }

        float tile_max = pa_block_reduce_max(warp_max, s_reduce, tid, lane_id, warp_id);
        float local_exp = 0.0f;
        if (tid < tile_len) {
            float p = expf(s_scores[tid] - tile_max);
            s_scores[tid] = p;
            local_exp = p;
        }
        __syncthreads();
        float tile_sum = pa_block_reduce_sum(local_exp, s_reduce, tid, lane_id, warp_id);

        float new_max = fmaxf(prev_max, tile_max);
        float alpha = expf(prev_max - new_max);
        float beta = expf(tile_max - new_max);
        prev_sum = prev_sum * alpha + tile_sum * beta;
        acc_out *= alpha;

        {
            float weighted_v = 0.0f;
#pragma unroll 4
            for (int j = dv_rank; j < tile_len; j += dv_group) {
                const int j_global = kv_start + j;
                const int blk = j_global / block_size;
                const int off = j_global % block_size;
                const int v_phys = v_bt[blk] * block_size + off;
                weighted_v += s_scores[j] * to_float(pool_base[v_phys * nkvhead * dv + kvh * dv + dv_idx]);
            }
            acc_out += beta * weighted_v;
        }

        prev_max = new_max;
        __syncthreads();
    }

    if (dv_group > 1) {
        s_scores[tid] = acc_out;
        __syncthreads();
        if (dv_rank == 0) {
            float sum = s_scores[tid];
            for (int g = 1; g < dv_group; ++g)
                sum += s_scores[g * dv + dv_idx];
            attn_out[req_idx * nhead * dv + h * dv + dv_idx] = from_float<T>(sum / prev_sum);
        }
    } else {
        if (tid < dv)
            attn_out[req_idx * nhead * dv + h * dv + tid] = from_float<T>(acc_out / prev_sum);
    }
}

// ============================================================================
// Dispatch
// ============================================================================

void paged_attention_decode(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seq_len,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size) {

    dim3 block(BLOCK_SIZE);
    dim3 grid(nhead);
    size_t smem_size = (head_dim + PA_TILE_KV + NUM_WARPS) * sizeof(float);

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return paged_attention_decode_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, nhead, nkvhead, head_dim, block_size);
    case ZEDINFER_DTYPE_F16:
        return paged_attention_decode_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<half *>(attn_val), reinterpret_cast<const half *>(q),
            reinterpret_cast<const half *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, nhead, nkvhead, head_dim, block_size);
    case ZEDINFER_DTYPE_BF16:
        return paged_attention_decode_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<cuda_bfloat16 *>(attn_val), reinterpret_cast<const cuda_bfloat16 *>(q),
            reinterpret_cast<const cuda_bfloat16 *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, nhead, nkvhead, head_dim, block_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

// ============================================================================
// Paged Attention Prefill Kernel
//
// Grid = (seqlen_q, nhead). One CUDA block per (query_position, head).
// Same online-softmax tiled algorithm as the non-paged prefill kernel,
// but K/V accessed via block table from the shared pool.
//
// Causal mask: query at position (past_len + query_idx) attends to
// KV positions 0..past_len+query_idx (inclusive).
// ============================================================================

template <typename T>
__global__ void paged_attention_prefill_kernel(
    T *__restrict__ attn_out,
    const T *__restrict__ Q,
    const T *__restrict__ pool_base,
    const int *__restrict__ k_block_table,
    const int *__restrict__ v_block_table,
    const float scale,
    const int nhead, const int nkvhead,
    const int d, const int past_len,
    const int seqlen_q, const int block_size) {

    const int query_pos = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);
    const int causal_end = past_len + query_pos + 1;

    extern __shared__ float smem[];
    float *s_q = smem;
    float *s_scores = s_q + d;
    float *s_reduce = s_scores + PA_TILE_KV;

    // Load Q into shared memory
    const T *q_ptr = Q + query_pos * nhead * d + h * d;
    for (int i = tid; i < d; i += blockDim.x) s_q[i] = to_float(q_ptr[i]);
    __syncthreads();

    const int dv = d;
    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < causal_end; kv_start += PA_TILE_KV) {
        const int tile_len = min(PA_TILE_KV, causal_end - kv_start);
        float warp_max = -FLT_MAX;

        // Q*K scoring with block-table-indexed K
        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const int j_global = kv_start + j_local;
                const int blk = j_global / block_size;
                const int off = j_global % block_size;
                const int k_phys = k_block_table[blk] * block_size + off;

                const T *k_ptr = pool_base + k_phys * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE)
                    dot += s_q[dim] * to_float(k_ptr[dim]);
                dot = pa_warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) s_scores[j_local] = score;
            }
            if (lane_id == 0 && j_local < tile_len)
                warp_max = fmaxf(warp_max, score);
        }

        float tile_max = pa_block_reduce_max(warp_max, s_reduce, tid, lane_id, warp_id);
        float local_exp = 0.0f;
        if (tid < tile_len) {
            float p = expf(s_scores[tid] - tile_max);
            s_scores[tid] = p;
            local_exp = p;
        }
        __syncthreads();
        float tile_sum = pa_block_reduce_sum(local_exp, s_reduce, tid, lane_id, warp_id);

        float new_max = fmaxf(prev_max, tile_max);
        float alpha = expf(prev_max - new_max);
        float beta = expf(tile_max - new_max);
        prev_sum = prev_sum * alpha + tile_sum * beta;
        acc_out *= alpha;

        // V aggregation with block-table-indexed V
        if (tid < dv) {
            float weighted_v = 0.0f;
#pragma unroll 4
            for (int j = 0; j < tile_len; ++j) {
                const int j_global = kv_start + j;
                const int blk = j_global / block_size;
                const int off = j_global % block_size;
                const int v_phys = v_block_table[blk] * block_size + off;

                weighted_v += s_scores[j] * to_float(pool_base[v_phys * nkvhead * dv + kvh * dv + tid]);
            }
            acc_out += beta * weighted_v;
        }
        prev_max = new_max;
        __syncthreads();
    }

    if (tid < dv)
        attn_out[query_pos * nhead * dv + h * dv + tid] = from_float<T>(acc_out / prev_sum);
}

// ============================================================================
// Prefill Dispatch
// ============================================================================

void paged_attention_prefill(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seqlen_q, int past_len,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size) {

    dim3 block(BLOCK_SIZE);
    dim3 grid(seqlen_q, nhead);
    size_t smem_size = (head_dim + PA_TILE_KV + NUM_WARPS) * sizeof(float);

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return paged_attention_prefill_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(pool_base),
            k_block_table, v_block_table,
            scale, nhead, nkvhead, head_dim, past_len, seqlen_q, block_size);
    case ZEDINFER_DTYPE_F16:
        return paged_attention_prefill_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<half *>(attn_val), reinterpret_cast<const half *>(q),
            reinterpret_cast<const half *>(pool_base),
            k_block_table, v_block_table,
            scale, nhead, nkvhead, head_dim, past_len, seqlen_q, block_size);
    case ZEDINFER_DTYPE_BF16:
        return paged_attention_prefill_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<cuda_bfloat16 *>(attn_val), reinterpret_cast<const cuda_bfloat16 *>(q),
            reinterpret_cast<const cuda_bfloat16 *>(pool_base),
            k_block_table, v_block_table,
            scale, nhead, nkvhead, head_dim, past_len, seqlen_q, block_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

// ============================================================================
// Batched Decode Dispatch
// ============================================================================

void paged_attention_decode_batched(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_tables, const int *v_block_tables,
    const int *seq_lens,
    int num_reqs, int max_blocks_per_seq,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size) {

    dim3 block(BLOCK_SIZE);
    dim3 grid(num_reqs, nhead);
    size_t smem_size = (head_dim + PA_TILE_KV + NUM_WARPS) * sizeof(float);

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return paged_attention_decode_batched_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(pool_base),
            k_block_tables, v_block_tables, seq_lens,
            scale, nhead, nkvhead, head_dim, block_size, max_blocks_per_seq);
    case ZEDINFER_DTYPE_F16:
        return paged_attention_decode_batched_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<half *>(attn_val), reinterpret_cast<const half *>(q),
            reinterpret_cast<const half *>(pool_base),
            k_block_tables, v_block_tables, seq_lens,
            scale, nhead, nkvhead, head_dim, block_size, max_blocks_per_seq);
    case ZEDINFER_DTYPE_BF16:
        return paged_attention_decode_batched_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<cuda_bfloat16 *>(attn_val), reinterpret_cast<const cuda_bfloat16 *>(q),
            reinterpret_cast<const cuda_bfloat16 *>(pool_base),
            k_block_tables, v_block_tables, seq_lens,
            scale, nhead, nkvhead, head_dim, block_size, max_blocks_per_seq);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
