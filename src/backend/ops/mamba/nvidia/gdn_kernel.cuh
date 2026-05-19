#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace zedinfer::ops::mamba::gdn_device {

// Per-head scalar prep (called once per token per V-head, by lane 0).
// Returns (beta, decay).  All scalar arithmetic in fp32.
__device__ inline float2 prepare_scalars(float b_raw, float a_raw,
                                         float A_log, float dt_bias) {
    // beta = sigmoid(b)
    float beta  = 1.0f / (1.0f + __expf(-b_raw));
    // softplus(x) = log(1 + exp(x));  use stable form via log1pf(exp(...))
    float sp    = (a_raw + dt_bias) > 20.0f ? (a_raw + dt_bias)
                                            : __logf(1.0f + __expf(a_raw + dt_bias));
    // decay = exp(-exp(A_log) * softplus(a+dt_bias))
    float decay = __expf(-__expf(A_log) * sp);
    return make_float2(beta, decay);
}

// Cooperatively compute Sk[d] = dot(S[d, :], k_vec) for one V-head.
//   S        : [Dv, Dk] in GMEM (row-major; row d is row d of state)
//   k_vec    : [Dk] in registers, replicated across threads (or in SMEM)
//   d        : which Dv row each thread is responsible for (caller maps)
//   Dk_tile  : columns per thread (Dk / blockDim.x typically)
//   Returns scalar dot product for this thread's (d, j-slice).
// Caller composes the full [Dv] vector by collecting one thread's result per d.
__device__ inline float dot_row(const float* S_row, const __nv_bfloat16* k_vec,
                                int Dk, int lane_id, int lanes_per_row) {
    float acc = 0.0f;
    for (int j = lane_id; j < Dk; j += lanes_per_row) {
        acc += S_row[j] * __bfloat162float(k_vec[j]);
    }
    // Warp reduction across lanes (warp-cooperative).
    for (int off = lanes_per_row / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    return acc;
}

// In-place update of one row d of S:
//   S[d, j] = decay * S[d, j] + beta * delta_d * k[j]
// for j in [0, Dk).  delta_d already computed by caller.
__device__ inline void update_row(float* S_row, const __nv_bfloat16* k_vec,
                                   float decay, float beta, float delta_d,
                                   int Dk, int lane_id, int lanes_per_row) {
    const float coef = beta * delta_d;
    for (int j = lane_id; j < Dk; j += lanes_per_row) {
        S_row[j] = decay * S_row[j] + coef * __bfloat162float(k_vec[j]);
    }
}

// Read-back: y[d] = dot(S[d, :], q_vec).  Same shape as dot_row.
__device__ inline float readout_row(const float* S_row,
                                     const __nv_bfloat16* q_vec,
                                     int Dk, int lane_id, int lanes_per_row) {
    float acc = 0.0f;
    for (int j = lane_id; j < Dk; j += lanes_per_row) {
        acc += S_row[j] * __bfloat162float(q_vec[j]);
    }
    for (int off = lanes_per_row / 2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    return acc;
}

} // namespace zedinfer::ops::mamba::gdn_device
