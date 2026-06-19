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
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) { val += __shfl_down_sync(0xffffffff, val, offset); }
    return val;
}

__device__ __forceinline__ float pa_warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float pa_block_reduce_sum(float val, float* smem, int tid, int lane_id, int warp_id) {
    val = pa_warp_reduce_sum(val);
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    float result = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < NUM_WARPS; ++w) { result += smem[w]; }
        smem[0] = result;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float pa_block_reduce_max(float val, float* smem, int tid, int lane_id, int warp_id) {
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    float result = -FLT_MAX;
    if (tid == 0) {
        for (int w = 0; w < NUM_WARPS; ++w) { result = fmaxf(result, smem[w]); }
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
__global__ void paged_attention_decode_kernel(T* __restrict__ attn_out, const T* __restrict__ Q,
                                              const T* __restrict__ k_pool_base, const T* __restrict__ v_pool_base,
                                              const int* __restrict__ page_table, const int seq_len, const float scale,
                                              const int nhead, const int nkvhead, const int d, const int block_size) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);

    // Shared memory layout:
    //   s_q[d]              - query vector
    //   s_scores[PA_TILE_KV]- attention scores per tile
    //   s_reduce[NUM_WARPS] - warp reduction scratch
    //   s_phys[PA_TILE_KV]  - precomputed physical token indices per tile
    //
    // s_phys breaks the dependent-load chain in K scoring and V aggregation:
    //   Before: block_table[global_load] → compute addr → K/V[global_load]
    //   After:  s_phys[smem_load] → K/V[global_load]  (1 hop instead of 2)
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_scores = s_q + d;
    float* s_reduce = s_scores + PA_TILE_KV;
    int* s_phys = reinterpret_cast<int*>(s_reduce + NUM_WARPS);

    // Load Q into shared memory
    const T* q_ptr = Q + h * d;
    for (int i = tid; i < d; i += blockDim.x) { s_q[i] = to_float(q_ptr[i]); }
    __syncthreads();

    const int dv = d;
    const int dv_idx = tid % dv;
    const int dv_group = blockDim.x / dv;
    const int dv_rank = tid / dv;

    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < seq_len; kv_start += PA_TILE_KV) {
        const int tile_len = min(PA_TILE_KV, seq_len - kv_start);

        // Precompute physical token addresses for this tile (all threads cooperate)
        for (int j = tid; j < tile_len; j += blockDim.x) {
            const int j_global = kv_start + j;
            const int blk = j_global / block_size;
            const int off = j_global % block_size;
            s_phys[j] = page_table[blk] * block_size + off;
        }
        __syncthreads();

        // Q*K scoring — block table lookup is now a fast smem read
        float warp_max = -FLT_MAX;
        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const T* k_ptr = k_pool_base + s_phys[j_local] * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE) { dot += s_q[dim] * to_float(k_ptr[dim]); }
                dot = pa_warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) {
                    s_scores[j_local] = score;
                }
            }
            if (lane_id == 0 && j_local < tile_len) {
                warp_max = fmaxf(warp_max, score);
            }
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

        // V aggregation — single-hop global load (address from smem)
        {
            float weighted_v = 0.0f;
#pragma unroll 4
            for (int j = dv_rank; j < tile_len; j += dv_group) {
                weighted_v += s_scores[j] * to_float(v_pool_base[s_phys[j] * nkvhead * dv + kvh * dv + dv_idx]);
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
            for (int g = 1; g < dv_group; ++g) { sum += s_scores[g * dv + dv_idx]; }
            attn_out[h * dv + dv_idx] = from_float<T>(sum / prev_sum);
        }
    } else {
        if (tid < dv) {
            attn_out[h * dv + tid] = from_float<T>(acc_out / prev_sum);
        }
    }
}

// ============================================================================
// Batched Paged Attention Decode Kernel
//
// Grid = (num_requests, nhead). One CUDA block per (request, head).
// Each request has its own block table and seq_len.
// ============================================================================

template <typename T>
__global__ void
paged_attention_decode_batched_kernel(T* __restrict__ attn_out,            // [num_reqs, nhead, head_dim]
                                      const T* __restrict__ Q,             // [num_reqs, nhead, head_dim]
                                      const T* __restrict__ k_pool_base, const T* __restrict__ v_pool_base,
                                      const int* __restrict__ page_tables, // [num_reqs, max_blocks_per_seq]
                                      const int* __restrict__ seq_lens,    // [num_reqs]
                                      const float scale, const int nhead, const int nkvhead, const int d,
                                      const int block_size, const int max_blocks_per_seq) {
    const int req_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);
    const int seq_len = seq_lens[req_idx];

    // Per-request page table
    const int* page_table = page_tables + req_idx * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_scores = s_q + d;
    float* s_reduce = s_scores + PA_TILE_KV;
    int* s_phys = reinterpret_cast<int*>(s_reduce + NUM_WARPS);

    const T* q_ptr = Q + req_idx * nhead * d + h * d;
    for (int i = tid; i < d; i += blockDim.x) { s_q[i] = to_float(q_ptr[i]); }
    __syncthreads();

    const int dv = d;
    const int dv_idx = tid % dv;
    const int dv_group = blockDim.x / dv;
    const int dv_rank = tid / dv;

    float prev_max = -FLT_MAX, prev_sum = 0.0f, acc_out = 0.0f;

    for (int kv_start = 0; kv_start < seq_len; kv_start += PA_TILE_KV) {
        const int tile_len = min(PA_TILE_KV, seq_len - kv_start);

        // Precompute physical token addresses for this tile
        for (int j = tid; j < tile_len; j += blockDim.x) {
            const int j_global = kv_start + j;
            const int blk = j_global / block_size;
            const int off = j_global % block_size;
            s_phys[j] = page_table[blk] * block_size + off;
        }
        __syncthreads();

        float warp_max = -FLT_MAX;
        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            float score = -FLT_MAX;
            if (j_local < tile_len) {
                const T* k_ptr = k_pool_base + s_phys[j_local] * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE) { dot += s_q[dim] * to_float(k_ptr[dim]); }
                dot = pa_warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) {
                    s_scores[j_local] = score;
                }
            }
            if (lane_id == 0 && j_local < tile_len) {
                warp_max = fmaxf(warp_max, score);
            }
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
                weighted_v += s_scores[j] * to_float(v_pool_base[s_phys[j] * nkvhead * dv + kvh * dv + dv_idx]);
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
            for (int g = 1; g < dv_group; ++g) { sum += s_scores[g * dv + dv_idx]; }
            attn_out[req_idx * nhead * dv + h * dv + dv_idx] = from_float<T>(sum / prev_sum);
        }
    } else {
        if (tid < dv) {
            attn_out[req_idx * nhead * dv + h * dv + tid] = from_float<T>(acc_out / prev_sum);
        }
    }
}

// ============================================================================
// Dispatch
// ============================================================================

void paged_attention_decode(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                            const std::byte* v_pool_base, const int* page_table, int seq_len, float scale,
                            zedinferDataType_t type, int nhead, int nkvhead, int head_dim, int block_size) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nhead);
    // smem: s_q[d] + s_scores[PA_TILE_KV] + s_reduce[NUM_WARPS] + s_phys[PA_TILE_KV]
    size_t smem_size = (head_dim + PA_TILE_KV + NUM_WARPS) * sizeof(float) + PA_TILE_KV * sizeof(int);

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return paged_attention_decode_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(k_pool_base), reinterpret_cast<const float*>(v_pool_base), page_table,
                seq_len, scale, nhead, nkvhead, head_dim, block_size);
        case ZEDINFER_DTYPE_F16:
            return paged_attention_decode_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<half*>(attn_val), reinterpret_cast<const half*>(q),
                reinterpret_cast<const half*>(k_pool_base), reinterpret_cast<const half*>(v_pool_base), page_table,
                seq_len, scale, nhead, nkvhead, head_dim, block_size);
        case ZEDINFER_DTYPE_BF16:
            return paged_attention_decode_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<cuda_bfloat16*>(attn_val), reinterpret_cast<const cuda_bfloat16*>(q),
                reinterpret_cast<const cuda_bfloat16*>(k_pool_base),
                reinterpret_cast<const cuda_bfloat16*>(v_pool_base), page_table, seq_len, scale, nhead, nkvhead,
                head_dim, block_size);
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
__global__ void paged_attention_prefill_kernel(T* __restrict__ attn_out, const T* __restrict__ Q,
                                               const T* __restrict__ k_pool_base, const T* __restrict__ v_pool_base,
                                               const int* __restrict__ page_table, const float scale, const int nhead,
                                               const int nkvhead, const int d, const int past_len, const int seqlen_q,
                                               const int block_size) {
    const int query_pos = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);
    const int causal_end = past_len + query_pos + 1;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_scores = s_q + d;
    float* s_reduce = s_scores + PA_TILE_KV;

    // Load Q into shared memory
    const T* q_ptr = Q + query_pos * nhead * d + h * d;
    for (int i = tid; i < d; i += blockDim.x) { s_q[i] = to_float(q_ptr[i]); }
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
                const int k_phys = page_table[blk] * block_size + off;

                const T* k_ptr = k_pool_base + k_phys * nkvhead * d + kvh * d;
                float dot = 0.0f;
                for (int dim = lane_id; dim < d; dim += WARP_SIZE) { dot += s_q[dim] * to_float(k_ptr[dim]); }
                dot = pa_warp_reduce_sum(dot);
                score = dot * scale;
                if (lane_id == 0) {
                    s_scores[j_local] = score;
                }
            }
            if (lane_id == 0 && j_local < tile_len) {
                warp_max = fmaxf(warp_max, score);
            }
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
                const int v_phys = page_table[blk] * block_size + off;

                weighted_v += s_scores[j] * to_float(v_pool_base[v_phys * nkvhead * dv + kvh * dv + tid]);
            }
            acc_out += beta * weighted_v;
        }
        prev_max = new_max;
        __syncthreads();
    }

    if (tid < dv) {
        attn_out[query_pos * nhead * dv + h * dv + tid] = from_float<T>(acc_out / prev_sum);
    }
}

// ============================================================================
// Prefill Dispatch
// ============================================================================

void paged_attention_prefill(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                             const std::byte* v_pool_base, const int* page_table, int seqlen_q, int past_len,
                             float scale, zedinferDataType_t type, int nhead, int nkvhead, int head_dim,
                             int block_size) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(seqlen_q, nhead);
    size_t smem_size = (head_dim + PA_TILE_KV + NUM_WARPS) * sizeof(float);

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return paged_attention_prefill_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(k_pool_base), reinterpret_cast<const float*>(v_pool_base), page_table,
                scale, nhead, nkvhead, head_dim, past_len, seqlen_q, block_size);
        case ZEDINFER_DTYPE_F16:
            return paged_attention_prefill_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<half*>(attn_val), reinterpret_cast<const half*>(q),
                reinterpret_cast<const half*>(k_pool_base), reinterpret_cast<const half*>(v_pool_base), page_table,
                scale, nhead, nkvhead, head_dim, past_len, seqlen_q, block_size);
        case ZEDINFER_DTYPE_BF16:
            return paged_attention_prefill_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<cuda_bfloat16*>(attn_val), reinterpret_cast<const cuda_bfloat16*>(q),
                reinterpret_cast<const cuda_bfloat16*>(k_pool_base),
                reinterpret_cast<const cuda_bfloat16*>(v_pool_base), page_table, scale, nhead, nkvhead, head_dim,
                past_len, seqlen_q, block_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

// ============================================================================
// Paged Attention for Small n_q (MTP spec-decode verify)
//
// Grid = (nhead,). One CUDA block per attention head.
// One block processes ALL N_Q queries against ALL KV positions in a single
// streaming pass — K/V is read once and scored against every query in the
// inner loop. This eliminates the 1/n_q reduction in K/V reuse that makes
// paged_attention_prefill slow at small n_q (it issues n_q separate blocks
// per head, each rereading the full KV cache).
//
// Online softmax state (prev_max, prev_sum) is kept per-query in registers.
// Output accumulation is per (query × head_dim_position) in registers, with
// thread-block layout: each thread owns one head_dim element and N_Q outputs.
//
// Causal mask: query q (q ∈ [0, N_Q)) attends to KV positions [0, past_len + q].
// Applied per-element when constructing the per-query score row.
//
// BLOCK_SIZE = head_dim (one thread per output element). For Qwen3.5
// head_dim=128 we get 128 threads/CTA (4 warps), enough for the inner
// reductions and well within register/smem budget.
// ============================================================================

template <typename T, int N_Q>
__global__ void paged_attention_small_nq_kernel(T* __restrict__ attn_out, const T* __restrict__ Q,
                                                const T* __restrict__ k_pool_base, const T* __restrict__ v_pool_base,
                                                const int* __restrict__ page_table, const float scale, const int nhead,
                                                const int nkvhead, const int d, const int past_len,
                                                const int block_size) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int kvh = h / (nhead / nkvhead);
    const int total_kv = past_len + N_Q;

    // BLOCK_SIZE=256 (8 warps) matches the pa_block_reduce_* helpers' hardcoded
    // NUM_WARPS=8. For head_dim=128 this gives dv_group=2 — V aggregation work
    // is split between two thread groups, then reduced via shared memory before
    // the final write.
    const int dv = d;
    const int dv_idx = tid % dv;
    const int dv_group = blockDim.x / dv;
    const int dv_rank = tid / dv;

    // Smem layout:
    //   s_q[N_Q * d]                 — N_Q query vectors loaded once
    //   s_scores[N_Q * PA_TILE_KV]   — per-query scores (raw, then post-exp)
    //   s_reduce[NUM_WARPS]          — warp reduction scratch (shared across queries)
    //   s_phys[PA_TILE_KV]           — physical token indices (shared across queries)
    // s_scores is also reused as scratch for the dv_group reduction at the end.
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_scores = s_q + N_Q * d;
    float* s_reduce = s_scores + N_Q * PA_TILE_KV;
    int* s_phys = reinterpret_cast<int*>(s_reduce + NUM_WARPS);

    // Load all N_Q query vectors into smem. Q is laid out [N_Q, nhead, d].
#pragma unroll
    for (int q = 0; q < N_Q; ++q) {
        const T* q_ptr = Q + q * nhead * d + h * d;
        for (int i = tid; i < d; i += blockDim.x) { s_q[q * d + i] = to_float(q_ptr[i]); }
    }
    __syncthreads();

    // Per-query running state (registers).
    float prev_max[N_Q];
    float prev_sum[N_Q];
    float acc_out[N_Q]; // each thread accumulates the output for its (dv_rank, dv_idx) slot, per query.
#pragma unroll
    for (int q = 0; q < N_Q; ++q) {
        prev_max[q] = -FLT_MAX;
        prev_sum[q] = 0.0f;
        acc_out[q] = 0.0f;
    }

    for (int kv_start = 0; kv_start < total_kv; kv_start += PA_TILE_KV) {
        const int tile_len = min(PA_TILE_KV, total_kv - kv_start);

        // Precompute physical token addresses (shared across all queries).
        for (int j = tid; j < tile_len; j += blockDim.x) {
            const int j_global = kv_start + j;
            const int blk = j_global / block_size;
            const int off = j_global % block_size;
            s_phys[j] = page_table[blk] * block_size + off;
        }
        __syncthreads();

        // Q*K scoring: each warp handles one KV position, computes N_Q dots.
        for (int j_base = 0; j_base < tile_len; j_base += NUM_WARPS) {
            const int j_local = j_base + warp_id;
            if (j_local < tile_len) {
                const T* k_ptr = k_pool_base + s_phys[j_local] * nkvhead * d + kvh * d;
                const int j_global = kv_start + j_local;

                // For each query, compute Q·K, apply scale, apply causal mask.
#pragma unroll
                for (int q = 0; q < N_Q; ++q) {
                    float dot = 0.0f;
                    for (int dim = lane_id; dim < d; dim += WARP_SIZE) {
                        dot += s_q[q * d + dim] * to_float(k_ptr[dim]);
                    }
                    dot = pa_warp_reduce_sum(dot);
                    // Causal: query q sees KV positions <= past_len + q.
                    float score = (j_global <= past_len + q) ? dot * scale : -FLT_MAX;
                    if (lane_id == 0) {
                        s_scores[q * PA_TILE_KV + j_local] = score;
                    }
                }
            }
        }
        __syncthreads();

        // Per-query online softmax: block-reduce max + exp + block-reduce sum
        // for each query. After this loop s_scores holds the exp'd probabilities
        // for each query, and we have per-query beta to merge into acc_out.
        float beta[N_Q];
#pragma unroll
        for (int q = 0; q < N_Q; ++q) {
            // Block-reduce max for query q.
            float my_max = -FLT_MAX;
            for (int j = tid; j < tile_len; j += blockDim.x) { my_max = fmaxf(my_max, s_scores[q * PA_TILE_KV + j]); }
            float warp_max = pa_warp_reduce_max(my_max);
            float tile_max = pa_block_reduce_max(warp_max, s_reduce, tid, lane_id, warp_id);

            // Block-reduce sum of exp(score - tile_max). Threads cooperatively
            // exponentiate, write back to smem (overwriting raw scores), and
            // reduce the partial sum. pa_block_reduce_sum takes per-thread
            // partial sums and does warp + block reduction internally — DO NOT
            // pre-warp-reduce here (that would double-shuffle and give a wrong
            // total). Asymmetric with pa_block_reduce_max which expects
            // pre-warp-reduced input on lane 0.
            float my_sum = 0.0f;
            for (int j = tid; j < tile_len; j += blockDim.x) {
                float p = expf(s_scores[q * PA_TILE_KV + j] - tile_max);
                s_scores[q * PA_TILE_KV + j] = p;
                my_sum += p;
            }
            float tile_sum = pa_block_reduce_sum(my_sum, s_reduce, tid, lane_id, warp_id);

            // Rescale prior accumulators and stash the new merge factor.
            float new_max = fmaxf(prev_max[q], tile_max);
            float alpha = expf(prev_max[q] - new_max);
            beta[q] = expf(tile_max - new_max);
            prev_sum[q] = prev_sum[q] * alpha + tile_sum * beta[q];
            acc_out[q] *= alpha;
            prev_max[q] = new_max;
        }

        // V aggregation: single streaming pass over the tile, reading each V
        // element ONCE and contributing it to all N_Q queries. This is the
        // critical loop the small_nq kernel exists for — paged_attention_prefill
        // launches n_q separate blocks and rereads V every time.
        for (int j = dv_rank; j < tile_len; j += dv_group) {
            float v_val = to_float(v_pool_base[s_phys[j] * nkvhead * d + kvh * d + dv_idx]);
#pragma unroll
            for (int q = 0; q < N_Q; ++q) { acc_out[q] += beta[q] * s_scores[q * PA_TILE_KV + j] * v_val; }
        }
        __syncthreads();
    }

    // Final write: reduce dv_group partial sums per query, then write to attn_out.
    // attn_out is [N_Q, nhead, d].
    if (dv_group > 1) {
#pragma unroll
        for (int q = 0; q < N_Q; ++q) {
            s_scores[tid] = acc_out[q];
            __syncthreads();
            if (dv_rank == 0) {
                float sum = s_scores[tid];
                for (int g = 1; g < dv_group; ++g) { sum += s_scores[g * dv + dv_idx]; }
                attn_out[q * nhead * d + h * d + dv_idx] = from_float<T>(sum / prev_sum[q]);
            }
            __syncthreads();
        }
    } else {
        if (tid < d) {
#pragma unroll
            for (int q = 0; q < N_Q; ++q) {
                attn_out[q * nhead * d + h * d + tid] = from_float<T>(acc_out[q] / prev_sum[q]);
            }
        }
    }
}

// Dispatch helper templated on n_q. Returns the smem byte size needed.
template <typename T>
static bool launch_small_nq_kernel(int n_q, std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                                   const std::byte* v_pool_base, const int* page_table, int past_len, float scale,
                                   int nhead, int nkvhead, int head_dim, int block_size) {
    // Match the pa_block_reduce helpers' assumption (NUM_WARPS=8 = BLOCK_SIZE/32).
    // BLOCK_SIZE=256 with head_dim=128 gives dv_group=2; the V aggregation work
    // is split across two thread groups whose partial sums are reduced at the
    // end. head_dim=64 → dv_group=4; head_dim=256 → dv_group=1. head_dim must
    // divide BLOCK_SIZE.
    if (BLOCK_SIZE % head_dim != 0) {
        return false;
    }
    dim3 grid(nhead);
    dim3 block(BLOCK_SIZE);

    auto run = [&](auto n_q_const) {
        constexpr int N_Q = decltype(n_q_const)::value;
        // s_q + s_scores + s_reduce + s_phys
        size_t smem_size = static_cast<size_t>(N_Q) * head_dim * sizeof(float)
                         + static_cast<size_t>(N_Q) * PA_TILE_KV * sizeof(float) + NUM_WARPS * sizeof(float)
                         + PA_TILE_KV * sizeof(int);
        paged_attention_small_nq_kernel<T, N_Q><<<grid, block, smem_size>>>(
            reinterpret_cast<T*>(attn_val), reinterpret_cast<const T*>(q), reinterpret_cast<const T*>(k_pool_base),
            reinterpret_cast<const T*>(v_pool_base), page_table, scale, nhead, nkvhead, head_dim, past_len, block_size);
    };

    switch (n_q) {
        case 2:
            run(std::integral_constant<int, 2>{});
            return true;
        case 3:
            run(std::integral_constant<int, 3>{});
            return true;
        case 4:
            run(std::integral_constant<int, 4>{});
            return true;
        default:
            return false;
    }
}

bool paged_attention_small_nq(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                              const std::byte* v_pool_base, const int* page_table, int seqlen_q, int past_len,
                              float scale, zedinferDataType_t type, int nhead, int nkvhead, int head_dim,
                              int block_size) {
    if (seqlen_q < 2 || seqlen_q > 4) {
        return false;
    }
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return launch_small_nq_kernel<float>(seqlen_q, attn_val, q, k_pool_base, v_pool_base, page_table, past_len,
                                                 scale, nhead, nkvhead, head_dim, block_size);
        case ZEDINFER_DTYPE_F16:
            return launch_small_nq_kernel<half>(seqlen_q, attn_val, q, k_pool_base, v_pool_base, page_table, past_len,
                                                scale, nhead, nkvhead, head_dim, block_size);
        case ZEDINFER_DTYPE_BF16:
            return launch_small_nq_kernel<cuda_bfloat16>(seqlen_q, attn_val, q, k_pool_base, v_pool_base, page_table,
                                                         past_len, scale, nhead, nkvhead, head_dim, block_size);
        default:
            return false;
    }
}

// ============================================================================
// Batched Decode Dispatch
// ============================================================================

void paged_attention_decode_batched(std::byte* attn_val, const std::byte* q, const std::byte* k_pool_base,
                                    const std::byte* v_pool_base, const int* page_tables, const int* seq_lens,
                                    int num_reqs, int max_blocks_per_seq, float scale, zedinferDataType_t type,
                                    int nhead, int nkvhead, int head_dim, int block_size) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_reqs, nhead);
    size_t smem_size = (head_dim + PA_TILE_KV + NUM_WARPS) * sizeof(float) + PA_TILE_KV * sizeof(int);

    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return paged_attention_decode_batched_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<float*>(attn_val), reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(k_pool_base), reinterpret_cast<const float*>(v_pool_base), page_tables,
                seq_lens, scale, nhead, nkvhead, head_dim, block_size, max_blocks_per_seq);
        case ZEDINFER_DTYPE_F16:
            return paged_attention_decode_batched_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<half*>(attn_val), reinterpret_cast<const half*>(q),
                reinterpret_cast<const half*>(k_pool_base), reinterpret_cast<const half*>(v_pool_base), page_tables,
                seq_lens, scale, nhead, nkvhead, head_dim, block_size, max_blocks_per_seq);
        case ZEDINFER_DTYPE_BF16:
            return paged_attention_decode_batched_kernel<<<grid, block, smem_size>>>(
                reinterpret_cast<cuda_bfloat16*>(attn_val), reinterpret_cast<const cuda_bfloat16*>(q),
                reinterpret_cast<const cuda_bfloat16*>(k_pool_base),
                reinterpret_cast<const cuda_bfloat16*>(v_pool_base), page_tables, seq_lens, scale, nhead, nkvhead,
                head_dim, block_size, max_blocks_per_seq);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
