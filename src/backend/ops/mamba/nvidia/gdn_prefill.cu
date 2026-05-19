#include "backend/ops/mamba/gdn.hpp"
#include "gdn_kernel.cuh"

#include "backend/core/context/context.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops::mamba {

namespace {

__global__ void gdn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,        // [N, Hk * Dk]
    const __nv_bfloat16* __restrict__ k,        // [N, Hk * Dk]
    const __nv_bfloat16* __restrict__ v,        // [N, Hv * Dv]
    const __nv_bfloat16* __restrict__ b,        // [N, Hv]
    const __nv_bfloat16* __restrict__ a,        // [N, Hv]
    const float*          __restrict__ A_log,   // [Hv]
    const __nv_bfloat16* __restrict__ dt_bias,  // [Hv]
    float*                __restrict__ S_base,  // [Hv, Dv, Dk] (slot, layer pre-offset)
    __nv_bfloat16*        __restrict__ out,     // [N, Hv * Dv]
    int N, int Hv, int Hk, int Dv, int Dk) {

    const int vh   = blockIdx.x;
    const int lane = threadIdx.x;
    if (vh >= Hv) return;
    const int rep  = Hv / Hk;
    const int kh   = vh / rep;

    float Alog_v   = A_log[vh];
    float dtb_v    = __bfloat162float(dt_bias[vh]);
    float*  S_vh   = S_base + (size_t)vh * Dv * Dk;

    for (int t = 0; t < N; ++t) {
        // Per-token scalars
        float beta, decay;
        if (lane == 0) {
            float b_raw = __bfloat162float(b[(size_t)t * Hv + vh]);
            float a_raw = __bfloat162float(a[(size_t)t * Hv + vh]);
            float2 s    = gdn_device::prepare_scalars(b_raw, a_raw, Alog_v, dtb_v);
            beta = s.x; decay = s.y;
        }
        beta  = __shfl_sync(0xffffffff, beta,  0);
        decay = __shfl_sync(0xffffffff, decay, 0);

        const __nv_bfloat16* k_vec = k + (size_t)t * Hk * Dk + kh * Dk;
        const __nv_bfloat16* q_vec = q + (size_t)t * Hk * Dk + kh * Dk;
        const __nv_bfloat16* v_vec = v + (size_t)t * Hv * Dv + vh * Dv;
        __nv_bfloat16*       y_vec = out + (size_t)t * Hv * Dv + vh * Dv;

        for (int d = 0; d < Dv; ++d) {
            float* S_row = S_vh + (size_t)d * Dk;
            float Sk     = gdn_device::dot_row(S_row, k_vec, Dk, lane, 32);
            float delta_d;
            if (lane == 0) delta_d = __bfloat162float(v_vec[d]) - Sk;
            delta_d = __shfl_sync(0xffffffff, delta_d, 0);
            gdn_device::update_row(S_row, k_vec, decay, beta, delta_d, Dk, lane, 32);
            float y_d = gdn_device::readout_row(S_row, q_vec, Dk, lane, 32);
            if (lane == 0) y_vec[d] = __float2bfloat16(y_d);
        }
    }
}

} // namespace

void gdn_prefill_launch(const GDNParams& p) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    const int N  = p.num_tokens;
    const int Hv = (int)p.state_view.num_v_heads;
    const int Dv = (int)p.state_view.value_head_dim;
    const int Dk = (int)p.state_view.d_state;
    const int Hk_Dk = (int)(p.k->numel() / N);
    const int Hk    = Hk_Dk / Dk;

    float* S_base = reinterpret_cast<float*>(
        reinterpret_cast<char*>(p.state_view.ssm_base)
        + (int64_t)p.slot_idx  * p.state_view.ssm_stride_slot
        + (int64_t)p.layer_idx * p.state_view.ssm_stride_layer);

    dim3 grid(Hv);
    dim3 block(32);
    gdn_prefill_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(p.q->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.k->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.v->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.b->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.a->data()),
        reinterpret_cast<const float*>(p.A_log->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.dt_bias->data()),
        S_base,
        reinterpret_cast<__nv_bfloat16*>(p.out->data()),
        N, Hv, Hk, Dv, Dk);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("gdn_prefill_launch: ") + cudaGetErrorString(e));
    }
}

} // namespace zedinfer::ops::mamba
