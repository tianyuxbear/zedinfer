#include "backend/core/context/context.hpp"
#include "backend/ops/mrope/mrope_3d.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace zedinfer::ops {

namespace {

// One thread per (token n, head hd, dim-pair pi). Each thread fully rotates
// one bf16 pair (pi, pi+half) — the HF "rotate_half" form used by
// Qwen3_5MoeTextRotaryEmbedding.apply_rotary_pos_emb (see modeling_qwen3_5_moe.py
// lines 558-585). HF builds cos/sin of shape (..., Dh_rot) by concatenating
// freqs with itself along the last axis, so cos[..., i] == cos[..., i+half]
// for i in [0, half); the rotation pairs slot i with slot i+half, both rotated
// by the same angle freq[i] = theta^(-(2*i)/(2*half)).
//
// The pi-axis assignment (t/h/w) is derived from the (s0, s1, s2) section
// split. `interleaved` selects between two layouts:
//   - chunked   (interleaved=false): [T..., H..., W...] each chunk of length section[axis]
//   - interleaved (interleaved=true):  matches HF Qwen3_5MoeTextRotaryEmbedding.
//     apply_interleaved_mrope: T defaults everywhere, H overwrites pi in
//     {1, 4, 7, ...} up to section[1]*3, W overwrites pi in {2, 5, 8, ...} up
//     to section[2]*3. See modeling_qwen3_5_moe.py:165-180.
//
// NOTE: An earlier version of this kernel paired (2*pi, 2*pi+1) — the
// GPT-J interleaved rotation form. That is mathematically NOT equivalent to
// HF's rotate_half (it rotates different pairs), so when applied to q/k
// projections trained for HF's half-rotation it produced subtly wrong q.k
// dot products. The error compounded across all full-attention layers and
// caused token-level divergence from HF after the very first generated token
// (the Qwen3.5 alignment bug investigated in
// docs/debug/qwen3_5_sampler_and_thinking_drift.md).
//
// freq = exp(-theta_log * (2*pi) / (2*half)) — equivalent to theta^(-(2*pi)/(2*half))
// but cheaper because half is small and theta_log is precomputed on the host.
__global__ void mrope_3d_kernel(__nv_bfloat16* __restrict__ x, const int32_t* __restrict__ pos_t,
                                const int32_t* __restrict__ pos_h, const int32_t* __restrict__ pos_w, int N, int H,
                                int Dh, int half, int s0, int s1, int s2, int interleaved, float theta_log) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * H * half;
    if (idx >= total) {
        return;
    }

    const int pi = idx % half;
    const int tmp = idx / half;
    const int hd = tmp % H;
    const int n = tmp / H;

    int axis;
    if (interleaved) {
        // HF interleaving: H overwrites pi % 3 == 1 (offset 1) up to s1*3;
        //                  W overwrites pi % 3 == 2 (offset 2) up to s2*3;
        //                  otherwise T owns the slot.
        if ((pi % 3) == 1 && pi < s1 * 3) {
            axis = 1;
        } else if ((pi % 3) == 2 && pi < s2 * 3) {
            axis = 2;
        } else {
            axis = 0;
        }
    } else {
        axis = (pi < s0) ? 0 : (pi < s0 + s1) ? 1 : 2;
    }
    const int pos_val = (axis == 0) ? pos_t[n] : (axis == 1) ? pos_h[n] : pos_w[n];

    // freq = exp(-theta_log * (2*pi) / (2*half))
    const float exponent = -theta_log * static_cast<float>(2 * pi) / static_cast<float>(2 * half);
    const float freq = __expf(exponent);
    const float angle = static_cast<float>(pos_val) * freq;
    float c, s;
    __sincosf(angle, &s, &c);

    // HF rotate_half pair: slot pi (low half) and slot pi+half (high half),
    // both rotated by the same angle. Matches q_embed = q*cos + rotate_half(q)*sin.
    const int row_base = (n * H + hd) * Dh;
    const int idx_lo = row_base + pi;
    const int idx_hi = row_base + pi + half;
    const float a = __bfloat162float(x[idx_lo]);
    const float b = __bfloat162float(x[idx_hi]);
    x[idx_lo] = __float2bfloat16(a * c - b * s);
    x[idx_hi] = __float2bfloat16(a * s + b * c);
}

} // namespace

void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const model::MRoPEConfig& cfg) {
    if (!x || !pos_ids_thw) {
        throw std::runtime_error("ops::mrope_3d: null tensor input");
    }
    if (x->ndim() != 3) {
        throw std::runtime_error("ops::mrope_3d: x must be 3-D [N, H, Dh]");
    }
    if (pos_ids_thw->ndim() != 2 || pos_ids_thw->shape()[0] != 3) {
        throw std::runtime_error("ops::mrope_3d: pos_ids_thw must be [3, N]");
    }
    if (x->dtype() != ZEDINFER_DTYPE_BF16) {
        throw std::runtime_error("ops::mrope_3d: only BF16 x is supported on NVIDIA");
    }
    if (pos_ids_thw->dtype() != ZEDINFER_DTYPE_I32) {
        throw std::runtime_error("ops::mrope_3d: pos_ids_thw must be int32");
    }

    const int N = static_cast<int>(x->shape()[0]);
    const int H = static_cast<int>(x->shape()[1]);
    const int Dh = static_cast<int>(x->shape()[2]);
    int Dh_rot = static_cast<int>(static_cast<float>(Dh) * cfg.partial_factor);
    if (Dh_rot % 2 != 0) {
        --Dh_rot;
    }
    if (Dh_rot <= 0 || N <= 0 || H <= 0) {
        return; // Nothing to rotate; pass-through dims stay untouched.
    }
    const int half = Dh_rot / 2;

    auto* x_ptr = reinterpret_cast<__nv_bfloat16*>(x->data());
    auto* pos = reinterpret_cast<const int32_t*>(pos_ids_thw->data());
    const int32_t* pos_t = pos + 0 * N;
    const int32_t* pos_h = pos + 1 * N;
    const int32_t* pos_w = pos + 2 * N;

    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    constexpr int kBlock = 128;
    const int total = N * H * half;
    const dim3 block(kBlock);
    const dim3 grid(static_cast<unsigned int>((total + kBlock - 1) / kBlock));

    // logf(theta) is float-only; host std::log gives the same result for
    // theta <= 1e7, but using logf keeps the float pipeline consistent.
    const float theta_log = std::log(cfg.theta);
    mrope_3d_kernel<<<grid, block, 0, stream>>>(x_ptr, pos_t, pos_h, pos_w, N, H, Dh, half, cfg.section[0],
                                                cfg.section[1], cfg.section[2], cfg.interleaved ? 1 : 0, theta_log);

    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("ops::mrope_3d: kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace zedinfer::ops
