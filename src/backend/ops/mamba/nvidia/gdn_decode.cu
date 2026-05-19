#include "backend/ops/mamba/gdn.hpp"
#include "gdn_kernel.cuh"

#include "backend/core/context/context.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops::mamba {

namespace {

// One CTA = one V-head.  32 threads (1 warp).  N=1 (decode).
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,        // [Hk * Dk]
    const __nv_bfloat16* __restrict__ k,        // [Hk * Dk]
    const __nv_bfloat16* __restrict__ v,        // [Hv * Dv]
    const __nv_bfloat16* __restrict__ b,        // [Hv]
    const __nv_bfloat16* __restrict__ a,        // [Hv]
    const float*          __restrict__ A_log,   // [Hv]
    const __nv_bfloat16* __restrict__ dt_bias,  // [Hv]
    float*                __restrict__ S_base,  // [Hv, Dv, Dk] — pre-offset to (slot, layer)
    __nv_bfloat16*        __restrict__ out,     // [Hv * Dv]
    int Hv, int Hk, int Dv, int Dk) {

    const int vh   = blockIdx.x;
    const int lane = threadIdx.x;
    if (vh >= Hv) return;
    const int rep  = Hv / Hk;
    const int kh   = vh / rep;

    // Scalars: beta, decay (lane 0 computes, broadcast).
    float beta, decay;
    if (lane == 0) {
        float b_raw = __bfloat162float(b[vh]);
        float a_raw = __bfloat162float(a[vh]);
        float Alog  = A_log[vh];
        float dtb   = __bfloat162float(dt_bias[vh]);
        float2 s    = gdn_device::prepare_scalars(b_raw, a_raw, Alog, dtb);
        beta = s.x; decay = s.y;
    }
    beta  = __shfl_sync(0xffffffff, beta,  0);
    decay = __shfl_sync(0xffffffff, decay, 0);

    const __nv_bfloat16* k_vec = k + kh * Dk;
    const __nv_bfloat16* q_vec = q + kh * Dk;
    const __nv_bfloat16* v_vec = v + vh * Dv;
    float*               S_vh  = S_base + (size_t)vh * Dv * Dk;
    __nv_bfloat16*       y_vec = out + vh * Dv;

    // Iterate over Dv rows; one warp handles one row at a time.
    for (int d = 0; d < Dv; ++d) {
        float* S_row = S_vh + (size_t)d * Dk;

        // 1) Sk[d] = dot(S[d, :], k)
        float Sk = gdn_device::dot_row(S_row, k_vec, Dk, lane, 32);

        // 2) delta_d = v[d] - Sk
        float delta_d;
        if (lane == 0) delta_d = __bfloat162float(v_vec[d]) - Sk;
        delta_d = __shfl_sync(0xffffffff, delta_d, 0);

        // 3) S[d, :] = decay * S[d, :] + beta * delta_d * k[:]
        gdn_device::update_row(S_row, k_vec, decay, beta, delta_d, Dk, lane, 32);

        // 4) y[d] = dot(S_new[d, :], q)
        float y_d = gdn_device::readout_row(S_row, q_vec, Dk, lane, 32);
        if (lane == 0) y_vec[d] = __float2bfloat16(y_d);
    }
}

} // namespace

void gdn_decode_launch(const GDNParams& p) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    const int Hv = (int)p.state_view.num_v_heads;
    const int Dv = (int)p.state_view.value_head_dim;
    const int Dk = (int)p.state_view.d_state;          // == Dk in Qwen3.5
    const int Hk_Dk = (int)(p.k->numel() / (p.num_tokens));
    const int Hk    = Hk_Dk / Dk;

    float* S_base = reinterpret_cast<float*>(
        reinterpret_cast<char*>(p.state_view.ssm_base)
        + (int64_t)p.slot_idx  * p.state_view.ssm_stride_slot
        + (int64_t)p.layer_idx * p.state_view.ssm_stride_layer);

    dim3 grid(Hv);
    dim3 block(32);
    gdn_decode_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(p.q->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.k->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.v->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.b->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.a->data()),
        reinterpret_cast<const float*>(p.A_log->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.dt_bias->data()),
        S_base,
        reinterpret_cast<__nv_bfloat16*>(p.out->data()),
        Hv, Hk, Dv, Dk);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("gdn_decode_launch: ") + cudaGetErrorString(e));
    }
}

} // namespace zedinfer::ops::mamba
