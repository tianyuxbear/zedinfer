#include "backend/ops/mamba/gdn.hpp"
#include "gdn_kernel.cuh"

#include "backend/core/context/context.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops::mamba {

namespace {

// One CTA = one V-head, multiple warps per CTA. The outer token loop is
// sequential (state carries across tokens), but within each token the per-row
// math is independent so warps share that token's k_vec/q_vec via shared
// memory and stride over the Dv axis.
template <int WARPS_PER_CTA>
__global__ void gdn_prefill_kernel(const __nv_bfloat16* __restrict__ q,       // [N, Hk * Dk]
                                   const __nv_bfloat16* __restrict__ k,       // [N, Hk * Dk]
                                   const __nv_bfloat16* __restrict__ v,       // [N, Hv * Dv]
                                   const __nv_bfloat16* __restrict__ b,       // [N, Hv]
                                   const __nv_bfloat16* __restrict__ a,       // [N, Hv]
                                   const float* __restrict__ A_log,           // [Hv]
                                   const __nv_bfloat16* __restrict__ dt_bias, // [Hv]
                                   float* __restrict__ S_base,                // [Hv, Dv, Dk] (slot, layer pre-offset)
                                   __nv_bfloat16* __restrict__ out,           // [N, Hv * Dv]
                                   int N, int Hv, int Hk, int Dv, int Dk) {
    extern __shared__ __nv_bfloat16 smem_bf16[];
    __nv_bfloat16* sk_vec = smem_bf16;
    __nv_bfloat16* sq_vec = sk_vec + Dk;
    float* s_scalars = reinterpret_cast<float*>(sq_vec + Dk); // [beta, decay]

    const int vh = blockIdx.x;
    if (vh >= Hv) {
        return;
    }
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5; // tid / 32
    const int lane = tid & 0x1f;  // tid % 32
    const int rep = Hv / Hk;
    const int kh = vh / rep;

    const float Alog_v = A_log[vh];
    const float dtb_v = __bfloat162float(dt_bias[vh]);
    float* S_vh = S_base + (size_t)vh * Dv * Dk;

    for (int t = 0; t < N; ++t) {
        // Stage this token's k_vec/q_vec into shared memory + compute per-token
        // beta/decay (warp 0 lane 0, written through shared for the rest).
        const __nv_bfloat16* k_glob = k + (size_t)t * Hk * Dk + (size_t)kh * Dk;
        const __nv_bfloat16* q_glob = q + (size_t)t * Hk * Dk + (size_t)kh * Dk;
        for (int j = tid; j < Dk; j += WARPS_PER_CTA * 32) {
            sk_vec[j] = k_glob[j];
            sq_vec[j] = q_glob[j];
        }
        if (warp_id == 0 && lane == 0) {
            float b_raw = __bfloat162float(b[(size_t)t * Hv + vh]);
            float a_raw = __bfloat162float(a[(size_t)t * Hv + vh]);
            float2 s = gdn_device::prepare_scalars(b_raw, a_raw, Alog_v, dtb_v);
            s_scalars[0] = s.x; // beta
            s_scalars[1] = s.y; // decay
        }
        __syncthreads();

        const float beta = s_scalars[0];
        const float decay = s_scalars[1];

        const __nv_bfloat16* v_vec = v + (size_t)t * Hv * Dv + (size_t)vh * Dv;
        __nv_bfloat16* y_vec = out + (size_t)t * Hv * Dv + (size_t)vh * Dv;

        // Per-row recurrence within this token: warps stride over Dv.
        for (int d = warp_id; d < Dv; d += WARPS_PER_CTA) {
            float* S_row = S_vh + (size_t)d * Dk;
            for (int j = lane; j < Dk; j += 32) { S_row[j] *= decay; }
            float Sk = gdn_device::dot_row(S_row, sk_vec, Dk, lane, 32);
            float delta_d;
            if (lane == 0) {
                delta_d = __bfloat162float(v_vec[d]) - Sk;
            }
            delta_d = __shfl_sync(0xffffffff, delta_d, 0);
            const float coef = beta * delta_d;
            for (int j = lane; j < Dk; j += 32) { S_row[j] += coef * __bfloat162float(sk_vec[j]); }
            float y_d = gdn_device::readout_row(S_row, sq_vec, Dk, lane, 32);
            if (lane == 0) {
                y_vec[d] = __float2bfloat16(y_d);
            }
        }
        // Next token depends on the just-written state; sync before reusing
        // the shared scratch for the new k_vec/q_vec/scalars.
        __syncthreads();
    }
}

} // namespace

void gdn_prefill_launch(const GDNParams& p) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    const int N = p.num_tokens;
    const int Hv = (int)p.state_view.num_v_heads;
    const int Dv = (int)p.state_view.value_head_dim;
    const int Dk = (int)p.state_view.d_state;
    const int Hk_Dk = (int)(p.k->numel() / N);
    const int Hk = Hk_Dk / Dk;

    float* S_base = reinterpret_cast<float*>(reinterpret_cast<char*>(p.state_view.ssm_base)
                                             + (int64_t)p.slot_idx * p.state_view.ssm_stride_slot
                                             + (int64_t)p.layer_idx * p.state_view.ssm_stride_layer);

    constexpr int WARPS_PER_CTA = 32;
    dim3 grid(Hv);
    dim3 block(WARPS_PER_CTA * 32);
    const size_t smem_bytes = static_cast<size_t>(2 * Dk) * sizeof(__nv_bfloat16) + 2 * sizeof(float);
    gdn_prefill_kernel<WARPS_PER_CTA><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(p.q->data()), reinterpret_cast<const __nv_bfloat16*>(p.k->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.v->data()), reinterpret_cast<const __nv_bfloat16*>(p.b->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.a->data()), reinterpret_cast<const float*>(p.A_log->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.dt_bias->data()), S_base,
        reinterpret_cast<__nv_bfloat16*>(p.out->data()), N, Hv, Hk, Dv, Dk);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("gdn_prefill_launch: ") + cudaGetErrorString(e));
    }
}

} // namespace zedinfer::ops::mamba
