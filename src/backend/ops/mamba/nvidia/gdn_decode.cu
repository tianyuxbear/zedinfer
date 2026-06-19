#include "backend/ops/mamba/gdn.hpp"
#include "gdn_kernel.cuh"

#include "backend/core/context/context.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops::mamba {

namespace {

// One CTA = one V-head. Block has WARPS_PER_CTA warps; each warp owns a
// stride-`WARPS_PER_CTA` slice of the Dv axis (so for Dv=128, WARPS=32 the
// warp handles 4 rows; for Dv=64, WARPS=32 it handles 2 rows; etc.). The
// per-row math (decay → Sk → delta → S += beta*delta*k → y = S @ q) is
// independent across `d`, so all warps run their slices concurrently.
//
// Shared memory holds k_vec/q_vec (Dk bf16 each) so every warp reads the
// projection vectors once at CTA scope instead of once per d iteration.
// beta/decay are computed by warp 0 and broadcast via a 2-float SMEM slot.
template <int WARPS_PER_CTA>
__global__ void gdn_decode_kernel(const __nv_bfloat16* __restrict__ q,       // [Hk * Dk]
                                  const __nv_bfloat16* __restrict__ k,       // [Hk * Dk]
                                  const __nv_bfloat16* __restrict__ v,       // [Hv * Dv]
                                  const __nv_bfloat16* __restrict__ b,       // [Hv]
                                  const __nv_bfloat16* __restrict__ a,       // [Hv]
                                  const float* __restrict__ A_log,           // [Hv]
                                  const __nv_bfloat16* __restrict__ dt_bias, // [Hv]
                                  float* __restrict__ S_base,      // [Hv, Dv, Dk] — pre-offset to (slot, layer)
                                  __nv_bfloat16* __restrict__ out, // [Hv * Dv]
                                  int Hv, int Hk, int Dv, int Dk) {
    extern __shared__ __nv_bfloat16 smem_bf16[];
    __nv_bfloat16* sk_vec = smem_bf16;
    __nv_bfloat16* sq_vec = sk_vec + Dk;
    // 2-float scratch for beta/decay broadcast lives right after the bf16 vectors.
    float* s_scalars = reinterpret_cast<float*>(sq_vec + Dk);

    const int vh = blockIdx.x;
    if (vh >= Hv) {
        return;
    }
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5; // tid / 32
    const int lane = tid & 0x1f;  // tid % 32
    const int rep = Hv / Hk;
    const int kh = vh / rep;

    // Stage k_vec and q_vec (Dk bf16 each) into shared memory.
    const __nv_bfloat16* k_glob = k + (size_t)kh * Dk;
    const __nv_bfloat16* q_glob = q + (size_t)kh * Dk;
    for (int j = tid; j < Dk; j += WARPS_PER_CTA * 32) {
        sk_vec[j] = k_glob[j];
        sq_vec[j] = q_glob[j];
    }

    // beta/decay: warp 0 lane 0 computes, write to shared, all warps read.
    if (warp_id == 0 && lane == 0) {
        float b_raw = __bfloat162float(b[vh]);
        float a_raw = __bfloat162float(a[vh]);
        float Alog = A_log[vh];
        float dtb = __bfloat162float(dt_bias[vh]);
        float2 s = gdn_device::prepare_scalars(b_raw, a_raw, Alog, dtb);
        s_scalars[0] = s.x; // beta
        s_scalars[1] = s.y; // decay
    }
    __syncthreads();

    const float beta = s_scalars[0];
    const float decay = s_scalars[1];

    const __nv_bfloat16* v_vec = v + (size_t)vh * Dv;
    float* S_vh = S_base + (size_t)vh * Dv * Dk;
    __nv_bfloat16* y_vec = out + (size_t)vh * Dv;

    // Each warp processes d's strided by WARPS_PER_CTA (so warp 0 -> d=0, WARPS,
    // 2*WARPS, ...; warp 1 -> d=1, WARPS+1, ...). Per-d math is independent.
    for (int d = warp_id; d < Dv; d += WARPS_PER_CTA) {
        float* S_row = S_vh + (size_t)d * Dk;

        // 1) Decay: S[d, :] *= decay.
        for (int j = lane; j < Dk; j += 32) { S_row[j] *= decay; }

        // 2) Sk = dot(decayed S[d, :], k).
        float Sk = gdn_device::dot_row(S_row, sk_vec, Dk, lane, 32);

        // 3) delta_d = v[d] - Sk  (lane 0 computes, warp-broadcast).
        float delta_d;
        if (lane == 0) {
            delta_d = __bfloat162float(v_vec[d]) - Sk;
        }
        delta_d = __shfl_sync(0xffffffff, delta_d, 0);

        // 4) S[d, :] += beta * delta_d * k[:].
        const float coef = beta * delta_d;
        for (int j = lane; j < Dk; j += 32) { S_row[j] += coef * __bfloat162float(sk_vec[j]); }

        // 5) y[d] = dot(S_new[d, :], q).
        float y_d = gdn_device::readout_row(S_row, sq_vec, Dk, lane, 32);
        if (lane == 0) {
            y_vec[d] = __float2bfloat16(y_d);
        }
    }
}

} // namespace

void gdn_decode_launch(const GDNParams& p) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    const int Hv = (int)p.state_view.num_v_heads;
    const int Dv = (int)p.state_view.value_head_dim;
    const int Dk = (int)p.state_view.d_state; // == Dk in Qwen3.5
    const int Hk_Dk = (int)(p.k->numel() / (p.num_tokens));
    const int Hk = Hk_Dk / Dk;

    float* S_base = reinterpret_cast<float*>(reinterpret_cast<char*>(p.state_view.ssm_base)
                                             + (int64_t)p.slot_idx * p.state_view.ssm_stride_slot
                                             + (int64_t)p.layer_idx * p.state_view.ssm_stride_layer);

    // 32 warps (1024 threads) per CTA: max occupancy on Ampere/Ada SMs without
    // exceeding the 1024-thread block limit. For Qwen3.5 (Dv=128) each warp owns
    // exactly 4 rows; smaller Dv still works (the strided d-loop covers all rows
    // and extra warps just exit the loop).
    constexpr int WARPS_PER_CTA = 32;
    dim3 grid(Hv);
    dim3 block(WARPS_PER_CTA * 32);
    // Shared memory: 2 * Dk bf16 for k_vec/q_vec, plus 2 floats for beta/decay.
    const size_t smem_bytes = static_cast<size_t>(2 * Dk) * sizeof(__nv_bfloat16) + 2 * sizeof(float);
    gdn_decode_kernel<WARPS_PER_CTA><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(p.q->data()), reinterpret_cast<const __nv_bfloat16*>(p.k->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.v->data()), reinterpret_cast<const __nv_bfloat16*>(p.b->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.a->data()), reinterpret_cast<const float*>(p.A_log->data()),
        reinterpret_cast<const __nv_bfloat16*>(p.dt_bias->data()), S_base,
        reinterpret_cast<__nv_bfloat16*>(p.out->data()), Hv, Hk, Dv, Dk);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("gdn_decode_launch: ") + cudaGetErrorString(e));
    }
}

} // namespace zedinfer::ops::mamba
