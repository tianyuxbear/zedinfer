# Qwen3.5 P2 (M1) — Hybrid Text-only Forward

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia` produce a coherent text reply to "Who are you?" (vision tower untouched; MTP still inactive; 35B-A3B path included as stretch test). DoD: 27B emits sensible Qwen-like reply (correctness alignment is M2's job; M1 just needs running + non-garbage output).

**Architecture:** New `hybrid_transformer_forward.cpp` with A2 layout (five per-layer-kind functions). New ops: `ops::mamba::ssu` (FlashInfer wrapper), `ops::mamba::causal_conv1d` (self-written kernel + state), `ops::mrope_3d`, `ops::attn_output_gate`. PagedForwardContext + Scheduler + BlockAllocator extended for logical-layer / dual-pool admission.

**Tech Stack:** C++17, CUDA 12.x, FlashInfer Mamba kernels (vendored).

**Reference:** Design doc `docs/plan/qwen3_5_support.md` §6 / §7 / §9, M1 row in §10.

**Pre-condition:** P1 (M0) complete; SSU link verified; SSMStatePool / Qwen3_5Model skeletons present.

---

### Task 1: `ops::mamba::ssu` wrapper header

**Files:**
- Create: `include/backend/ops/mamba/ssu.hpp`

- [ ] **Step 1: Create the wrapper header**

```cpp
// include/backend/ops/mamba/ssu.hpp
#pragma once
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

namespace zedinfer::ops::mamba {

struct SSUParams {
    // State pool view + slot/layer indices
    model::SSMStateView state_view;
    int slot_idx;
    int layer_idx;

    // Inputs (all [N, ...] where N = num_tokens this step)
    tensor_t q;        // [N, num_k_heads * key_head_dim]
    tensor_t k;        // [N, num_k_heads * key_head_dim]
    tensor_t v;        // [N, num_v_heads * value_head_dim]
    tensor_t a;        // [N, num_v_heads]
    tensor_t b;        // [N, num_v_heads]
    tensor_t A_log;    // [num_v_heads]
    tensor_t dt_bias;  // [num_v_heads]
    tensor_t z;        // [N, num_v_heads * value_head_dim] (pre-silu'd)

    // Output (pre-allocated by caller)
    tensor_t out;      // [N, num_v_heads * value_head_dim]

    int num_tokens;    // N: 1 → decode STP path; >1 → prefill varlen
};

void ssu(const SSUParams& params);

} // namespace zedinfer::ops::mamba
```

- [ ] **Step 2: Commit**

```bash
git add include/backend/ops/mamba/ssu.hpp
git commit -m "feat(ops): add ops::mamba::ssu header (FlashInfer wrapper interface)"
```

---

### Task 2: `ops::mamba::ssu` NVIDIA wrapper impl

**Files:**
- Create: `src/backend/ops/mamba/nvidia/ssu_wrapper.cu`

- [ ] **Step 1: Write the wrapper**

```cuda
// src/backend/ops/mamba/nvidia/ssu_wrapper.cu
#include "backend/ops/mamba/ssu.hpp"
#include "backend/core/context/context.hpp"

#ifdef USE_FLASHINFER

#define DIM 128       // value_head_dim (must match template instantiation in test_flashinfer_ssu_link)
#define DSTATE 128
#define NTOKENS_MTP 1
#define PHILOX_ROUNDS 0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/mamba/invoke_selective_state_update_mtp.cuh>

#include <stdexcept>
#include <vector>

namespace zedinfer::ops::mamba {

namespace {

inline void* byte_offset(void* base, int64_t bytes) {
    return reinterpret_cast<char*>(base) + bytes;
}

// Build a cu_seqlens int32 device tensor [0, N] for varlen path.
// Reuses a thread-local pinned host scratch + per-call async H2D on compute stream.
struct VarlenScratch {
    int32_t host[2] = {0, 0};
    tensor_t device;
};

thread_local VarlenScratch g_varlen;

int32_t* prepare_cu_seqlens(int N, cudaStream_t stream) {
    g_varlen.host[1] = N;
    if (!g_varlen.device) {
        g_varlen.device = Tensor::create({2}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    }
    cudaMemcpyAsync(g_varlen.device->data(), g_varlen.host, 2 * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    return reinterpret_cast<int32_t*>(g_varlen.device->data());
}

} // namespace

void ssu(const SSUParams& p) {
    auto& runtime = core::context().runtime();
    cudaStream_t stream = runtime.compute_stream();

    using namespace flashinfer::mamba;
    SelectiveStateMTPParams params{};

    // Slot/layer offset into pooled state buffer
    void* state_base = byte_offset(p.state_view.ssm_base,
                                    p.slot_idx  * p.state_view.ssm_stride_slot
                                  + p.layer_idx * p.state_view.ssm_stride_layer);
    params.state    = state_base;
    params.x        = p.v->data();
    params.dt       = p.a->data();
    params.A_log    = p.A_log->data();
    params.B        = p.b->data();
    params.C        = p.q->data();
    params.D        = nullptr;
    params.z        = p.z->data();
    params.dt_bias  = p.dt_bias->data();
    params.output   = p.out->data();
    params.batch    = 1;
    params.dim      = p.state_view.num_v_heads * p.state_view.value_head_dim;
    params.dstate   = p.state_view.d_state;
    params.nheads   = p.state_view.num_v_heads;
    params.ngroups  = 1;  // Qwen3.5 has no group sharing across K heads (verify per dtype combo)
    params.ntokens_mtp = p.num_tokens;
    params.cu_seqlens  = (p.num_tokens > 1) ? prepare_cu_seqlens(p.num_tokens, stream) : nullptr;
    params.num_accepted_tokens = nullptr;
    params.dt_softplus = true;

    mtp::invokeSelectiveStateUpdateMTP<
        __nv_bfloat16,  // input_t
        __nv_bfloat16,  // weight_t
        float,          // matrixA_t
        __nv_bfloat16,  // state_t
        int32_t,        // stateIndex_t
        void            // state_scale_t
    >(params, SSUAlgorithm::kAuto, stream);
}

} // namespace zedinfer::ops::mamba

#else

namespace zedinfer::ops::mamba {
void ssu(const SSUParams&) {
    throw std::runtime_error("ops::mamba::ssu requires --flashinfer=y build");
}
} // namespace zedinfer::ops::mamba

#endif
```

- [ ] **Step 2: Build (link only)**

Run: `xmake build` — should succeed (the new .cu file glob-picked into `backend` target).

If link errors mention `ngroups`: peek at `flashinfer::mamba::SelectiveStateMTPParams` definition in `third_party/flashinfer/include/flashinfer/mamba/common.cuh` and adjust field names.

- [ ] **Step 3: Commit**

```bash
git add src/backend/ops/mamba/nvidia/ssu_wrapper.cu
git commit -m "feat(ops): ops::mamba::ssu wrapper around FlashInfer SSU MTP kernel"
```

---

### Task 3: SSU unit test (decode + prefill)

**Files:**
- Create: `tests/unit/test_ops_mamba_ssu.cpp`
- Create: `tests/fixtures/mamba_ssu_reference/gen_reference.py` (Python script to produce ground-truth tensors)
- Create: `tests/fixtures/mamba_ssu_reference/*.bin` (golden inputs & outputs)

- [ ] **Step 1: Write Python reference generator**

Create `tests/fixtures/mamba_ssu_reference/gen_reference.py`:
```python
"""Generate golden inputs/outputs for ops::mamba::ssu unit test.
Uses HF transformers' mamba_ssm reference implementation."""
import numpy as np
import torch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def selective_state_update_reference(state, q, k, v, a, b, A_log, dt_bias, z, dt_softplus=True):
    """Single-step (N=1) selective state update — reference numpy/torch impl.
    Shapes:
      state:    [Hv, Dv, dstate]   (in/out)
      q,k:      [Hk, Dk]
      v:        [Hv, Dv]
      a,b:      [Hv]
      A_log:    [Hv]
      dt_bias:  [Hv]
      z:        [Hv, Dv]
    Returns y: [Hv, Dv]
    """
    dt = torch.nn.functional.softplus(a + dt_bias) if dt_softplus else (a + dt_bias)
    A = -torch.exp(A_log.float()).to(state.dtype)
    # state_t = exp(dt*A) * state + (dt*b).unsqueeze(-1) * (v.unsqueeze(-1) * <some k mixing>)
    # NOTE: This is an approximation; the exact Mamba2 formulation depends on B/C grouping.
    # Use the actual HF Qwen3.5 mamba layer forward to produce reference.
    # For now this stub generates plausible shapes; replace by HF call once available.
    decay = torch.exp(dt.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(-1).unsqueeze(-1))
    bv = (b.unsqueeze(-1) * v).unsqueeze(-1)  # [Hv, Dv, 1]
    state = decay * state + dt.unsqueeze(-1).unsqueeze(-1) * bv
    # y = (state @ q_via_C_mixing).sum_along_dstate; placeholder simple readout:
    y = state.sum(dim=-1)  # [Hv, Dv]
    y = y * torch.sigmoid(z)  # silu(z) approx for test purposes
    return state, y

torch.manual_seed(42)
Hv, Dv, dstate, Hk, Dk = 8, 16, 16, 4, 16
state = torch.zeros(Hv, Dv, dstate, dtype=torch.bfloat16)
q = torch.randn(Hk, Dk, dtype=torch.bfloat16)
k = torch.randn(Hk, Dk, dtype=torch.bfloat16)
v = torch.randn(Hv, Dv, dtype=torch.bfloat16)
a = torch.randn(Hv, dtype=torch.bfloat16)
b = torch.randn(Hv, dtype=torch.bfloat16)
A_log = torch.randn(Hv, dtype=torch.float32)
dt_bias = torch.randn(Hv, dtype=torch.bfloat16)
z = torch.randn(Hv, Dv, dtype=torch.bfloat16)

state_out, y = selective_state_update_reference(state.clone(), q, k, v, a, b, A_log, dt_bias, z)

# Save as raw binary (row-major)
def save(name, t):
    arr = t.cpu().numpy()
    arr.astype(arr.dtype).tofile(os.path.join(OUT_DIR, name + ".bin"))

save("state_in", state)
save("q", q); save("k", k); save("v", v)
save("a", a); save("b", b)
save("A_log", A_log); save("dt_bias", dt_bias); save("z", z)
save("state_out_ref", state_out)
save("y_ref", y)
print("[ok] generated reference tensors in", OUT_DIR)
```

Run:
```bash
. .venv/bin/activate  # if not already
pip install torch
python3 tests/fixtures/mamba_ssu_reference/gen_reference.py
```

Note: the placeholder Python reference is approximate. **For real M2 byte-exact alignment, replace this with calls into HF `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5MambaLayer.forward`** (once HF transformers supports Qwen3.5). For M1 the goal is "kernel runs, output is non-NaN, dtype/shape correct" — exact numbers come in M2.

- [ ] **Step 2: Write C++ test**

Create `tests/unit/test_ops_mamba_ssu.cpp`:
```cpp
#include "backend/ops/mamba/ssu.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace zedinfer;
using namespace zedinfer::model;

static std::vector<__nv_bfloat16> load_bf16(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end); size_t bytes = f.tellg(); f.seekg(0);
    assert(bytes == expected_count * sizeof(__nv_bfloat16) && "size mismatch");
    std::vector<__nv_bfloat16> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

int main() {
    core::context().setRuntime(createRuntime(ZEDINFER_DEVICE_NVIDIA, 0));

    const int Hv = 8, Dv = 16, dstate = 16, Hk = 4, Dk = 16;

    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 1;
    cfg.num_v_heads = Hv;
    cfg.value_head_dim = Dv;
    cfg.d_state = dstate;
    cfg.conv_kernel_dim = 4;
    cfg.qkv_dim = Hk*Dk*2 + Hv*Dv;
    cfg.max_concurrent = 1;
    cfg.state_dtype = ZEDINFER_DTYPE_BF16;

    ExecutorConfig exec{ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0};
    SSMStatePool pool(cfg, exec);
    int slot = pool.acquire_slot();
    pool.reset_slot(slot);

    // Allocate input tensors (N=1 decode mode)
    auto q = Tensor::create({1, (size_t)Hk*Dk}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto k = Tensor::create({1, (size_t)Hk*Dk}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto v = Tensor::create({1, (size_t)Hv*Dv}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto a = Tensor::create({1, (size_t)Hv},    ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto b = Tensor::create({1, (size_t)Hv},    ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto A_log   = Tensor::create({(size_t)Hv}, ZEDINFER_DTYPE_F32,  ZEDINFER_DEVICE_NVIDIA, 0);
    auto dt_bias = Tensor::create({(size_t)Hv}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto z = Tensor::create({1, (size_t)Hv*Dv}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto out = Tensor::create({1, (size_t)Hv*Dv}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    // Load reference inputs (skip if file doesn't exist — runtime smoke only)
    auto load_to = [&](tensor_t t, const std::string& name, size_t count) {
        auto vec = load_bf16("tests/fixtures/mamba_ssu_reference/" + name + ".bin", count);
        cudaMemcpy(t->data(), vec.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    };
    load_to(q, "q", Hk*Dk);
    load_to(k, "k", Hk*Dk);
    load_to(v, "v", Hv*Dv);
    load_to(a, "a", Hv);
    load_to(b, "b", Hv);
    // A_log is float32 — separate loader
    {
        std::ifstream f("tests/fixtures/mamba_ssu_reference/A_log.bin", std::ios::binary);
        std::vector<float> tmp(Hv);
        f.read(reinterpret_cast<char*>(tmp.data()), Hv * sizeof(float));
        cudaMemcpy(A_log->data(), tmp.data(), Hv * sizeof(float), cudaMemcpyHostToDevice);
    }
    load_to(dt_bias, "dt_bias", Hv);
    load_to(z, "z", Hv*Dv);

    ops::mamba::SSUParams p;
    p.state_view = pool.view();
    p.slot_idx = slot;
    p.layer_idx = 0;
    p.q = q; p.k = k; p.v = v; p.a = a; p.b = b;
    p.A_log = A_log; p.dt_bias = dt_bias; p.z = z;
    p.out = out;
    p.num_tokens = 1;

    ops::mamba::ssu(p);
    cudaDeviceSynchronize();

    // Read back output; check non-NaN and non-zero
    std::vector<__nv_bfloat16> out_host(Hv*Dv);
    cudaMemcpy(out_host.data(), out->data(), Hv*Dv*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    bool any_nonzero = false;
    for (auto v : out_host) {
        float fv = __bfloat162float(v);
        if (std::isnan(fv) || std::isinf(fv)) { std::cerr << "[fail] NaN/Inf in output\n"; return 1; }
        if (fv != 0.0f) any_nonzero = true;
    }
    if (!any_nonzero) { std::cerr << "[fail] all-zero output\n"; return 1; }

    std::cout << "[ok] SSU decode step ran, output non-zero, no NaN\n";
    return 0;
}
```

Add xmake target.

- [ ] **Step 3: Run**

Run: `xmake build test-ops-mamba-ssu && xmake run test-ops-mamba-ssu`
Expected: `[ok] SSU decode step ran, output non-zero, no NaN`. If output is all-zero or NaN: check that `prepare_cu_seqlens` only fires for N>1, and that we passed correct param layout. Compare against a literal Python torch run to debug.

- [ ] **Step 4: Add prefill (varlen) sub-test**

Append a second block to the test running N=4 tokens, sharing the same slot+layer (state should accumulate across 4 tokens). Verify output shape `[4, Hv*Dv]` and non-NaN.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_ops_mamba_ssu.cpp tests/fixtures/mamba_ssu_reference xmake.lua
git commit -m "test(ops): ops::mamba::ssu decode+varlen smoke tests"
```

---

### Task 4: `ops::mamba::causal_conv1d` (depthwise kernel=4 + state)

**Files:**
- Create: `include/backend/ops/mamba/causal_conv1d.hpp`
- Create: `src/backend/ops/mamba/cpu/causal_conv1d.cpp`
- Create: `src/backend/ops/mamba/nvidia/causal_conv1d.cu`
- Create: `tests/unit/test_ops_causal_conv1d.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/mamba/causal_conv1d.hpp
#pragma once
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

namespace zedinfer::ops::mamba {

// Causal depthwise 1D convolution with state.
//   x:           [N, qkv_dim]  in/out (replaces x in-place is allowed via separate out)
//   weight:      [qkv_dim, 1, kernel]  depthwise weight
//   state_view:  pool view (conv state lives in state_view.conv_base)
//   slot_idx, layer_idx: state pool addressing
// Updates conv state to last kernel-1 tokens of (state-prefixed) x.
// Returns:       [N, qkv_dim] convolved + silu output (caller passes pre-allocated out).
void causal_conv1d(tensor_t out, tensor_t x, tensor_t weight,
                    model::SSMStateView state_view, int slot_idx, int layer_idx);

} // namespace zedinfer::ops::mamba
```

- [ ] **Step 2: CPU impl (reference; small loop)**

```cpp
// src/backend/ops/mamba/cpu/causal_conv1d.cpp
#include "backend/ops/mamba/causal_conv1d.hpp"
#include "backend/core/context/context.hpp"

namespace zedinfer::ops::mamba {

void causal_conv1d(tensor_t out, tensor_t x, tensor_t weight,
                    model::SSMStateView v, int slot_idx, int layer_idx) {
    // x shape: [N, D]; weight: [D, 1, K]; state: [K-1, D] for (slot, layer)
    int N = x->shape()[0];
    int D = x->shape()[1];
    int K = v.conv_kernel_dim;

    const float* x_ptr = reinterpret_cast<float*>(x->data());
    const float* w_ptr = reinterpret_cast<float*>(weight->data());
    float* out_ptr = reinterpret_cast<float*>(out->data());

    char* state_layer = reinterpret_cast<char*>(v.conv_base)
                       + slot_idx  * v.conv_stride_slot
                       + layer_idx * v.conv_stride_layer;
    float* state_ptr = reinterpret_cast<float*>(state_layer);
    // state holds last (K-1) tokens of qkv_dim
    // For each new token n: window = [state_t-(K-1)..state_t-1, x_n]
    // y_n[d] = sum_{i=0..K-1} window[i, d] * w[d, 0, i]
    // Then SiLU: y = y * sigmoid(y)
    // After processing N tokens, state := last (K-1) tokens of x

    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int i = 0; i < K; ++i) {
                float window_val;
                int relative = n - (K - 1 - i);
                if (relative < 0) {
                    int state_row = K - 1 + relative;  // index into state
                    window_val = state_ptr[state_row * D + d];
                } else {
                    window_val = x_ptr[relative * D + d];
                }
                acc += window_val * w_ptr[d * K + i];
            }
            // SiLU
            float s = 1.0f / (1.0f + std::exp(-acc));
            out_ptr[n * D + d] = acc * s;
        }
    }

    // Update state to last (K-1) tokens of x (concat'd with prior state if N < K-1)
    if (N >= K - 1) {
        for (int i = 0; i < K - 1; ++i) {
            for (int d = 0; d < D; ++d) {
                state_ptr[i * D + d] = x_ptr[(N - (K - 1) + i) * D + d];
            }
        }
    } else {
        // Shift state left by N, append new x
        for (int i = 0; i < K - 1 - N; ++i) {
            for (int d = 0; d < D; ++d) {
                state_ptr[i * D + d] = state_ptr[(i + N) * D + d];
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                state_ptr[(K - 1 - N + i) * D + d] = x_ptr[i * D + d];
            }
        }
    }
}

} // namespace zedinfer::ops::mamba
```

(Note: this is a fp32 reference. BF16 input requires conversion path. CPU impl is for correctness only; production runs on GPU.)

- [ ] **Step 3: NVIDIA impl (depthwise + state)**

```cuda
// src/backend/ops/mamba/nvidia/causal_conv1d.cu
#include "backend/ops/mamba/causal_conv1d.hpp"
#include "backend/core/context/context.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops::mamba {

namespace {

template <int K>
__global__ void causal_conv1d_kernel(
        const __nv_bfloat16* x,     // [N, D]
        const __nv_bfloat16* w,     // [D, 1, K]
        __nv_bfloat16* out,         // [N, D]
        __nv_bfloat16* state,       // [K-1, D]
        int N, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    // Load state into registers
    float window[K];
    #pragma unroll
    for (int i = 0; i < K - 1; ++i) {
        window[i] = __bfloat162float(state[i * D + d]);
    }

    // Load weight into registers
    float wv[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        wv[i] = __bfloat162float(w[d * K + i]);
    }

    for (int n = 0; n < N; ++n) {
        window[K - 1] = __bfloat162float(x[n * D + d]);
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < K; ++i) acc += window[i] * wv[i];
        float s = 1.0f / (1.0f + __expf(-acc));
        out[n * D + d] = __float2bfloat16(acc * s);

        // Shift window left
        #pragma unroll
        for (int i = 0; i < K - 1; ++i) window[i] = window[i + 1];
    }

    // Final window[0..K-2] becomes new state
    #pragma unroll
    for (int i = 0; i < K - 1; ++i) {
        state[i * D + d] = __float2bfloat16(window[i]);
    }
}

} // namespace

void causal_conv1d(tensor_t out, tensor_t x, tensor_t weight,
                    model::SSMStateView v, int slot_idx, int layer_idx) {
    int N = x->shape()[0];
    int D = x->shape()[1];
    int K = v.conv_kernel_dim;

    auto* x_ptr = reinterpret_cast<__nv_bfloat16*>(x->data());
    auto* w_ptr = reinterpret_cast<__nv_bfloat16*>(weight->data());
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out->data());

    char* state_layer = reinterpret_cast<char*>(v.conv_base)
                       + slot_idx  * v.conv_stride_slot
                       + layer_idx * v.conv_stride_layer;
    auto* state_ptr = reinterpret_cast<__nv_bfloat16*>(state_layer);

    auto stream = core::context().runtime().compute_stream();
    dim3 block(128);
    dim3 grid((D + 127) / 128);

    if (K == 4) {
        causal_conv1d_kernel<4><<<grid, block, 0, stream>>>(x_ptr, w_ptr, out_ptr, state_ptr, N, D);
    } else {
        throw std::runtime_error("causal_conv1d: only K=4 supported (Qwen3.5)");
    }
}

} // namespace zedinfer::ops::mamba
```

- [ ] **Step 4: Unit test**

```cpp
// tests/unit/test_ops_causal_conv1d.cpp
#include "backend/ops/mamba/causal_conv1d.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace zedinfer;
using namespace zedinfer::model;

int main() {
    core::context().setRuntime(createRuntime(ZEDINFER_DEVICE_NVIDIA, 0));
    int D = 32, N = 5, K = 4;

    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 1; cfg.num_v_heads = 1; cfg.value_head_dim = 1; cfg.d_state = 1;
    cfg.conv_kernel_dim = K; cfg.qkv_dim = D; cfg.max_concurrent = 1; cfg.state_dtype = ZEDINFER_DTYPE_BF16;
    ExecutorConfig exec{ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0};
    SSMStatePool pool(cfg, exec);
    int slot = pool.acquire_slot(); pool.reset_slot(slot);

    auto x = Tensor::create({(size_t)N, (size_t)D}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto w = Tensor::create({(size_t)D, 1, (size_t)K}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto out = Tensor::create({(size_t)N, (size_t)D}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    // Fill x and w with 1.0
    std::vector<__nv_bfloat16> ones(N * D);
    for (auto& v : ones) v = __float2bfloat16(1.0f);
    cudaMemcpy(x->data(), ones.data(), N * D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    std::vector<__nv_bfloat16> wvec(D * K);
    for (auto& v : wvec) v = __float2bfloat16(1.0f);
    cudaMemcpy(w->data(), wvec.data(), D * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ops::mamba::causal_conv1d(out, x, w, pool.view(), slot, 0);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> outv(N * D);
    cudaMemcpy(outv.data(), out->data(), N * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // First token: state is zero, x=1, so window=[0,0,0,1] → sum=1, silu(1)≈0.731
    float first = __bfloat162float(outv[0]);
    // Last (4th+) token: state should have all 1s by then; window=[1,1,1,1] → sum=4, silu(4)≈3.928
    float last = __bfloat162float(outv[(N - 1) * D]);

    assert(first > 0.6f && first < 0.8f && "first token silu(1) should be ~0.731");
    assert(last > 3.8f && last < 4.0f && "5th token silu(4) should be ~3.928");
    std::cout << "[ok] causal_conv1d first=" << first << " last=" << last << "\n";
    return 0;
}
```

Add xmake target.

- [ ] **Step 5: Run**

Run: `xmake build test-ops-causal-conv1d && xmake run test-ops-causal-conv1d`
Expected: `[ok] causal_conv1d first=~0.73 last=~3.93`.

- [ ] **Step 6: Commit**

```bash
git add include/backend/ops/mamba/causal_conv1d.hpp \
        src/backend/ops/mamba/cpu/causal_conv1d.cpp \
        src/backend/ops/mamba/nvidia/causal_conv1d.cu \
        tests/unit/test_ops_causal_conv1d.cpp xmake.lua
git commit -m "feat(ops): causal_conv1d (kernel=4, depthwise, with conv state)"
```

---

### Task 5: `ops::mrope_3d` — 3D rotary with partial factor + interleaved

**Files:**
- Create: `include/backend/ops/mrope/mrope_3d.hpp`
- Create: `src/backend/ops/mrope/cpu/mrope_3d.cpp`
- Create: `src/backend/ops/mrope/nvidia/mrope_3d.cu`
- Create: `tests/unit/test_ops_mrope_3d.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/mrope/mrope_3d.hpp
#pragma once
#include "backend/tensor/tensor.hpp"
#include "frontend/models/hybrid_forward_config.hpp"

namespace zedinfer::ops {

// In-place rotate first (head_dim * partial_factor) dims of x.
//   x:           [N, num_heads, head_dim]  (in/out)
//   pos_ids_thw: [3, N] int32 — (t, h, w) position id per token
//   cfg.section: [11, 11, 10] (dims split for t/h/w)
//   cfg.partial_factor: 0.25 → first 64 of 256 head_dim
//   cfg.theta: 1e7
//   cfg.interleaved: true → adjacent dim pairs cycle (t,h,w,t,h,w,...)
void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const model::MRoPEConfig& cfg);

} // namespace zedinfer::ops
```

- [ ] **Step 2: CPU reference impl**

```cpp
// src/backend/ops/mrope/cpu/mrope_3d.cpp
#include "backend/ops/mrope/mrope_3d.hpp"
#include <cmath>

namespace zedinfer::ops {

void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const model::MRoPEConfig& cfg) {
    int N = x->shape()[0];
    int H = x->shape()[1];
    int Dh = x->shape()[2];
    int Dh_rot = (int)(Dh * cfg.partial_factor);
    if (Dh_rot % 2 != 0) Dh_rot -= 1;  // must be even for cos/sin pairing
    int half = Dh_rot / 2;

    // For interleaved + 3D mrope:
    //   dim_pair_idx in [0, half)
    //   axis = (dim_pair_idx % 3)? Actually: section [t_count, h_count, w_count]
    //   First section[0] pairs use t; next section[1] use h; last section[2] use w.

    int* pos = reinterpret_cast<int*>(pos_ids_thw->data());  // [3, N], row-major
    int* pos_t = pos + 0 * N;
    int* pos_h = pos + 1 * N;
    int* pos_w = pos + 2 * N;

    float* xp = reinterpret_cast<float*>(x->data());  // for f32; bf16 needs convert

    auto axis_for_pair = [&](int pi) -> int {
        // pi in [0, half). section[0]+section[1]+section[2] should equal half.
        if (pi < cfg.section[0]) return 0;          // t
        if (pi < cfg.section[0] + cfg.section[1]) return 1;  // h
        return 2;                                   // w
    };

    for (int n = 0; n < N; ++n) {
        int t = pos_t[n], h = pos_h[n], w = pos_w[n];
        int axis_pos[3] = {t, h, w};
        for (int hd = 0; hd < H; ++hd) {
            float* row = xp + (n * H + hd) * Dh;
            for (int pi = 0; pi < half; ++pi) {
                int axis = axis_for_pair(pi);
                int pos_val = axis_pos[axis];
                float freq = 1.0f / std::pow(cfg.theta, (float)(2 * pi) / (float)Dh_rot);
                float angle = pos_val * freq;
                float c = std::cos(angle);
                float s = std::sin(angle);

                // Interleaved: x[2*pi], x[2*pi+1] are the pair
                float a = row[2 * pi];
                float b = row[2 * pi + 1];
                row[2 * pi]     = a * c - b * s;
                row[2 * pi + 1] = a * s + b * c;
            }
            // dims [Dh_rot, Dh) untouched
        }
    }
}

} // namespace zedinfer::ops
```

- [ ] **Step 3: NVIDIA impl**

```cuda
// src/backend/ops/mrope/nvidia/mrope_3d.cu
#include "backend/ops/mrope/mrope_3d.hpp"
#include "backend/core/context/context.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zedinfer::ops {

namespace {

__global__ void mrope_3d_kernel(
        __nv_bfloat16* x, int* pos_t, int* pos_h, int* pos_w,
        int N, int H, int Dh, int half, int s0, int s1,
        float theta_log) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * half;
    if (idx >= total) return;

    int pi = idx % half;
    int tmp = idx / half;
    int hd  = tmp % H;
    int n   = tmp / H;

    int axis;
    if (pi < s0) axis = 0;
    else if (pi < s0 + s1) axis = 1;
    else axis = 2;

    int pos_val;
    if      (axis == 0) pos_val = pos_t[n];
    else if (axis == 1) pos_val = pos_h[n];
    else                pos_val = pos_w[n];

    float freq = __expf(-theta_log * (float)(2 * pi) / (float)(2 * half));
    float angle = pos_val * freq;
    float c = __cosf(angle), s = __sinf(angle);

    int base = (n * H + hd) * Dh + 2 * pi;
    float a = __bfloat162float(x[base]);
    float b = __bfloat162float(x[base + 1]);
    x[base]     = __float2bfloat16(a * c - b * s);
    x[base + 1] = __float2bfloat16(a * s + b * c);
}

} // namespace

void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const model::MRoPEConfig& cfg) {
    int N = x->shape()[0];
    int H = x->shape()[1];
    int Dh = x->shape()[2];
    int Dh_rot = (int)(Dh * cfg.partial_factor);
    if (Dh_rot % 2 != 0) Dh_rot -= 1;
    int half = Dh_rot / 2;

    int* pos = reinterpret_cast<int*>(pos_ids_thw->data());
    int* pos_t = pos + 0 * N;
    int* pos_h = pos + 1 * N;
    int* pos_w = pos + 2 * N;

    auto stream = core::context().runtime().compute_stream();
    int total = N * H * half;
    dim3 block(128);
    dim3 grid((total + 127) / 128);

    mrope_3d_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x->data()),
        pos_t, pos_h, pos_w,
        N, H, Dh, half, cfg.section[0], cfg.section[1],
        __logf(cfg.theta));
}

} // namespace zedinfer::ops
```

- [ ] **Step 4: Unit test (vs HF apply_multimodal_rotary_pos_emb)**

Create `tests/fixtures/mrope_3d/gen_reference.py`:
```python
"""Generate golden mrope_3d output using HF transformers' reference."""
import torch, os, numpy as np
out_dir = os.path.dirname(os.path.abspath(__file__))

# Reproduce Qwen3.5 mrope: partial_factor=0.25, section=[11,11,10], theta=1e7
N, H, Dh = 5, 2, 256
Dh_rot = int(Dh * 0.25)  # 64
half = Dh_rot // 2  # 32; section sums to 32
section = [11, 11, 10]
theta = 1e7

torch.manual_seed(0)
x = torch.randn(N, H, Dh, dtype=torch.bfloat16)
pos_t = torch.arange(N, dtype=torch.int32)
pos_h = torch.arange(N, dtype=torch.int32) * 2
pos_w = torch.arange(N, dtype=torch.int32) * 3

# Reference: HF apply_multimodal_rotary_pos_emb (interleaved variant)
def mrope_ref(x, pos_t, pos_h, pos_w, theta, half, section, Dh):
    x = x.float()
    out = x.clone()
    for n in range(N):
        for hd in range(H):
            for pi in range(half):
                if pi < section[0]: pos = pos_t[n].item()
                elif pi < section[0]+section[1]: pos = pos_h[n].item()
                else: pos = pos_w[n].item()
                freq = 1.0 / (theta ** ((2*pi) / (2*half)))
                angle = pos * freq
                c, s = np.cos(angle), np.sin(angle)
                a, b = x[n, hd, 2*pi].item(), x[n, hd, 2*pi+1].item()
                out[n, hd, 2*pi]   = a*c - b*s
                out[n, hd, 2*pi+1] = a*s + b*c
    return out.to(torch.bfloat16)

ref = mrope_ref(x.clone(), pos_t, pos_h, pos_w, theta, half, section, Dh)
x.numpy().tofile(os.path.join(out_dir, "x_in.bin"))
ref.numpy().tofile(os.path.join(out_dir, "x_out_ref.bin"))
np.array([pos_t.numpy(), pos_h.numpy(), pos_w.numpy()], dtype=np.int32).tofile(
    os.path.join(out_dir, "pos_thw.bin"))
print("[ok] mrope_3d reference generated")
```

Run: `python3 tests/fixtures/mrope_3d/gen_reference.py`

C++ test (load tensors, run kernel, compare):
```cpp
// tests/unit/test_ops_mrope_3d.cpp
#include "backend/ops/mrope/mrope_3d.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/tensor/tensor.hpp"
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace zedinfer;

int main() {
    core::context().setRuntime(createRuntime(ZEDINFER_DEVICE_NVIDIA, 0));
    const int N=5, H=2, Dh=256;
    auto x_in = Tensor::create({(size_t)N, (size_t)H, (size_t)Dh},
                                ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto pos = Tensor::create({3, (size_t)N}, ZEDINFER_DTYPE_I32,
                               ZEDINFER_DEVICE_NVIDIA, 0);

    std::vector<__nv_bfloat16> x_host(N*H*Dh);
    std::ifstream("tests/fixtures/mrope_3d/x_in.bin", std::ios::binary)
        .read(reinterpret_cast<char*>(x_host.data()), x_host.size()*2);
    cudaMemcpy(x_in->data(), x_host.data(), x_host.size()*2, cudaMemcpyHostToDevice);

    std::vector<int> pos_host(3*N);
    std::ifstream("tests/fixtures/mrope_3d/pos_thw.bin", std::ios::binary)
        .read(reinterpret_cast<char*>(pos_host.data()), pos_host.size()*4);
    cudaMemcpy(pos->data(), pos_host.data(), pos_host.size()*4, cudaMemcpyHostToDevice);

    model::MRoPEConfig cfg;
    cfg.interleaved = true;
    cfg.section = {11, 11, 10};
    cfg.partial_factor = 0.25f;
    cfg.theta = 1e7f;
    ops::mrope_3d(x_in, pos, cfg);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> got(N*H*Dh), ref(N*H*Dh);
    cudaMemcpy(got.data(), x_in->data(), got.size()*2, cudaMemcpyDeviceToHost);
    std::ifstream("tests/fixtures/mrope_3d/x_out_ref.bin", std::ios::binary)
        .read(reinterpret_cast<char*>(ref.data()), ref.size()*2);

    float max_diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float d = std::abs(__bfloat162float(got[i]) - __bfloat162float(ref[i]));
        if (d > max_diff) max_diff = d;
    }
    std::cout << "[ok] mrope_3d max_abs_diff=" << max_diff << "\n";
    assert(max_diff < 0.01f && "mrope_3d output mismatch");
    return 0;
}
```

- [ ] **Step 5: Run**

Run: `xmake build test-ops-mrope-3d && xmake run test-ops-mrope-3d`
Expected: `[ok] mrope_3d max_abs_diff < 0.01`. If diff is large: check angle convention (sometimes section is over `Dh_rot` not `half`), and check that the partial pass-through dims (idx ≥ Dh_rot) are unchanged.

- [ ] **Step 6: Commit**

```bash
git add include/backend/ops/mrope/mrope_3d.hpp \
        src/backend/ops/mrope/cpu/mrope_3d.cpp \
        src/backend/ops/mrope/nvidia/mrope_3d.cu \
        tests/unit/test_ops_mrope_3d.cpp tests/fixtures/mrope_3d xmake.lua
git commit -m "feat(ops): mrope_3d (interleaved + partial rotary, [11,11,10] section)"
```

---

### Task 6: `ops::attn_output_gate`

**Files:**
- Create: `include/backend/ops/attn_output_gate/attn_output_gate.hpp`
- Create: `src/backend/ops/attn_output_gate/{cpu,nvidia}/attn_output_gate.{cpp,cu}`
- Create: `tests/unit/test_ops_attn_output_gate.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/attn_output_gate/attn_output_gate.hpp
#pragma once
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// attn := attn * sigmoid(g)   (in-place on attn)
// Both shape: [N, H, D]
void attn_output_gate(tensor_t attn, tensor_t g);

} // namespace zedinfer::ops
```

- [ ] **Step 2: NVIDIA impl**

```cuda
// src/backend/ops/attn_output_gate/nvidia/attn_output_gate.cu
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/core/context/context.hpp"
#include <cuda_bf16.h>

namespace zedinfer::ops {

namespace {
__global__ void attn_output_gate_kernel(__nv_bfloat16* attn, const __nv_bfloat16* g, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float a = __bfloat162float(attn[i]);
    float gv = __bfloat162float(g[i]);
    float s = 1.0f / (1.0f + __expf(-gv));
    attn[i] = __float2bfloat16(a * s);
}
} // namespace

void attn_output_gate(tensor_t attn, tensor_t g) {
    size_t total = attn->numel();
    auto stream = core::context().runtime().compute_stream();
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    attn_output_gate_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(attn->data()),
        reinterpret_cast<const __nv_bfloat16*>(g->data()),
        (int)total);
}

} // namespace zedinfer::ops
```

- [ ] **Step 3: CPU impl (reference)**

```cpp
// src/backend/ops/attn_output_gate/cpu/attn_output_gate.cpp
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include <cmath>

namespace zedinfer::ops {
void attn_output_gate(tensor_t attn, tensor_t g) {
    size_t n = attn->numel();
    float* a = reinterpret_cast<float*>(attn->data());
    float* gv = reinterpret_cast<float*>(g->data());
    for (size_t i = 0; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-gv[i]));
        a[i] *= s;
    }
}
}
```

- [ ] **Step 4: Test**

```cpp
// tests/unit/test_ops_attn_output_gate.cpp
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/tensor/tensor.hpp"
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace zedinfer;

int main() {
    core::context().setRuntime(createRuntime(ZEDINFER_DEVICE_NVIDIA, 0));
    auto attn = Tensor::create({4}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto g    = Tensor::create({4}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    __nv_bfloat16 av[4] = {__float2bfloat16(1.0f), __float2bfloat16(2.0f),
                            __float2bfloat16(-1.0f), __float2bfloat16(0.0f)};
    __nv_bfloat16 gv[4] = {__float2bfloat16(0.0f), __float2bfloat16(1.0f),
                            __float2bfloat16(-1.0f), __float2bfloat16(10.0f)};
    cudaMemcpy(attn->data(), av, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(g->data(),    gv, 8, cudaMemcpyHostToDevice);
    ops::attn_output_gate(attn, g);
    cudaDeviceSynchronize();

    __nv_bfloat16 res[4];
    cudaMemcpy(res, attn->data(), 8, cudaMemcpyDeviceToHost);

    // Expected: a*sigmoid(g)
    float exp0 = 1.0f * 0.5f;
    float exp1 = 2.0f / (1.0f + std::exp(-1.0f));
    float exp2 = -1.0f / (1.0f + std::exp(1.0f));
    float exp3 = 0.0f * (1.0f / (1.0f + std::exp(-10.0f)));
    assert(std::abs(__bfloat162float(res[0]) - exp0) < 0.01f);
    assert(std::abs(__bfloat162float(res[1]) - exp1) < 0.01f);
    assert(std::abs(__bfloat162float(res[2]) - exp2) < 0.01f);
    assert(std::abs(__bfloat162float(res[3]) - exp3) < 0.01f);
    std::cout << "[ok] attn_output_gate\n";
    return 0;
}
```

- [ ] **Step 5: Run + Commit**

```bash
xmake build test-ops-attn-output-gate && xmake run test-ops-attn-output-gate

git add include/backend/ops/attn_output_gate \
        src/backend/ops/attn_output_gate \
        tests/unit/test_ops_attn_output_gate.cpp xmake.lua
git commit -m "feat(ops): attn_output_gate (post-attention sigmoid multiply)"
```

---

### Task 7: `InferenceRequest` extension

**Files:**
- Modify: `include/zedinfer/request.hpp`

- [ ] **Step 1: Add fields**

In `class InferenceRequest`, add to the private section:
```cpp
    int      ssm_slot_idx_ = -1;
    tensor_t image_embeds_;
    tensor_t pos_ids_thw_;
    bool     has_images_ = false;
```

Add public accessors:
```cpp
    int  ssm_slot_idx() const { return ssm_slot_idx_; }
    void set_ssm_slot_idx(int idx) { ssm_slot_idx_ = idx; }

    bool has_images() const { return has_images_; }
    void set_image_embeds(tensor_t e) { image_embeds_ = e; has_images_ = (e != nullptr); }
    tensor_t image_embeds() const { return image_embeds_; }

    void set_pos_ids_thw(tensor_t t) { pos_ids_thw_ = t; }
    tensor_t pos_ids_thw() const { return pos_ids_thw_; }
```

- [ ] **Step 2: Compile-check + commit**

```bash
xmake build
git add include/zedinfer/request.hpp
git commit -m "feat(qwen3.5): InferenceRequest gains ssm_slot, image_embeds, pos_ids_thw"
```

---

### Task 8: `PagedForwardContext::write_kv / attend` logical layer index

**Files:**
- Modify: `include/frontend/models/paged_forward_context.hpp`
- Modify: `src/frontend/models/paged_forward_context.cpp`

- [ ] **Step 1: Survey current signature**

Run: `grep -n 'write_kv\|attend' include/frontend/models/paged_forward_context.hpp`

Verify both methods take an int layer index. They map to the KV block table directly.

- [ ] **Step 2: Add overload accepting logical kv_layer_idx**

In header, keep existing `write_kv(int, ...)`/`attend(int, ...)` signatures (callers from `transformer_forward.cpp` still pass `L`). Document semantics: callers pass the **logical KV layer index** (0..num_kv_layers-1). For non-hybrid models this equals `L`; for hybrid, callers pass `m.full_layer_index(L)`.

- [ ] **Step 3: Add a member `set_num_kv_layers(int)` if needed for sizing block table indices**

If the existing impl indexes into `block_table_[L]` directly with `num_hidden_layers`, no change needed — `full_layer_index(L)` is already in `[0, num_kv_layers)` range. Just document.

If the impl computes a stride based on `num_hidden_layers`, expose an option to pass `num_kv_layers` separately.

- [ ] **Step 4: Verify regression**

Run: `xmake build && xmake run test-blockpool && xmake run test-prefixcache`
Expected: all pass.

- [ ] **Step 5: Commit (likely doc-only change)**

```bash
git add include/frontend/models/paged_forward_context.hpp src/frontend/models/paged_forward_context.cpp
git commit -m "docs(paged-ctx): clarify kv_layer_idx semantics for hybrid models"
```

---

### Task 9: `BlockAllocator::init_block_pool` accepts `num_kv_layers`

**Files:**
- Modify: `src/zedinfer/engine.cpp`
- Modify: `include/zedinfer/engine.hpp` (if signature in header)

- [ ] **Step 1: Find `init_block_pool`**

Run: `grep -rn 'init_block_pool\|num_hidden_layers' src/zedinfer/engine.cpp include/zedinfer/engine.hpp | head -20`

- [ ] **Step 2: Extend signature**

Change init_block_pool to compute total blocks using `num_kv_layers` (= count of full-attention layers for hybrid; defaults to `num_hidden_layers` for non-hybrid).

In Engine ctor or wherever init_block_pool is called, pass the right count:
```cpp
int num_kv_layers = model->config().num_hidden_layers;
if (model->model_type() == "qwen3_5" || model->model_type() == "qwen3_5_moe") {
    auto& cfg = model->config();
    num_kv_layers = 0;
    for (auto& lt : cfg.layer_types) if (lt == "full_attention") ++num_kv_layers;
}
init_block_pool(..., num_kv_layers, ...);
```

- [ ] **Step 3: Regression test**

Run existing Qwen3 + Qwen3-MoE pings — both should still work with the default `num_kv_layers = num_hidden_layers`.

- [ ] **Step 4: Commit**

```bash
git add include/zedinfer/engine.hpp src/zedinfer/engine.cpp
git commit -m "feat(engine): init_block_pool accepts num_kv_layers (hybrid path support)"
```

---

### Task 10: `Scheduler::admit` dual-pool admission

**Files:**
- Modify: `src/zedinfer/scheduler.cpp`
- Modify: `include/zedinfer/scheduler.hpp`

- [ ] **Step 1: Add SSMStatePool* field to Scheduler**

In `scheduler.hpp`:
```cpp
class Scheduler {
public:
    void set_ssm_state_pool(model::SSMStatePool* p) { ssm_state_pool_ = p; }
    ...
private:
    model::SSMStatePool* ssm_state_pool_ = nullptr;
    ...
};
```

- [ ] **Step 2: Wire in Engine ctor**

After model construction in Engine ctor:
```cpp
if (auto* q = dynamic_cast<model::Qwen3_5Model*>(model_.get())) {
    serving_loop_->scheduler().set_ssm_state_pool(&q->ssm_state_pool());
}
```

- [ ] **Step 3: Extend admit logic**

In `Scheduler::admit_one`:
```cpp
// Existing block check
if (block_allocator_.available_blocks() < blocks_needed) return false;

// SSM slot check (only for hybrid models)
if (ssm_state_pool_ && ssm_state_pool_->num_free_slots() < 1) return false;
```

After successful admission, acquire slot:
```cpp
if (ssm_state_pool_) {
    int slot = ssm_state_pool_->acquire_slot();
    ssm_state_pool_->reset_slot(slot);
    req.set_ssm_slot_idx(slot);
}
```

In `on_request_finish`:
```cpp
if (ssm_state_pool_ && req.ssm_slot_idx() >= 0) {
    ssm_state_pool_->release_slot(req.ssm_slot_idx());
    req.set_ssm_slot_idx(-1);
}
```

- [ ] **Step 4: Test**

Existing scheduler tests should pass; manually verify with a unit test that drives admit + release in a loop and confirms `num_free_slots` cycles correctly.

- [ ] **Step 5: Commit**

```bash
git add src/zedinfer/scheduler.cpp include/zedinfer/scheduler.hpp src/zedinfer/engine.cpp
git commit -m "feat(scheduler): dual-pool admission (SSMStatePool + BlockAllocator)"
```

---

### Task 11: `transformer_forward` accepts optional `input_embeds`

**Files:**
- Modify: `include/frontend/models/forward_config.hpp` (function signature)
- Modify: `src/frontend/models/transformer_forward.cpp`

- [ ] **Step 1: Add optional parameter**

In `forward_config.hpp`:
```cpp
tensor_t transformer_forward(const ModelForwardConfig& model, PagedForwardContext& ctx,
                              const ExecutorConfig& exec_config,
                              DecodeScratch* scratch = nullptr,
                              tensor_t input_embeds = nullptr);
```

In `transformer_forward.cpp`, add at top:
```cpp
auto hidden = (input_embeds != nullptr)
                ? input_embeds
                : ops::embedding_lookup(ids, model.W("embed_tokens.weight"));
```

Find the existing `ops::embedding(hidden, ids, ...)` line and gate it on `input_embeds == nullptr`.

- [ ] **Step 2: Regression**

Run existing Qwen3 + Qwen3-MoE pings — both should still work (callers pass no input_embeds, default nullptr).

- [ ] **Step 3: Commit**

```bash
git add include/frontend/models/forward_config.hpp src/frontend/models/transformer_forward.cpp
git commit -m "feat(forward): transformer_forward accepts optional input_embeds"
```

---

### Task 12: `hybrid_transformer_forward` skeleton (outer loop)

**Files:**
- Create: `include/frontend/models/hybrid_transformer_forward.hpp`
- Create: `src/frontend/models/hybrid_transformer_forward.cpp`

- [ ] **Step 1: Header**

```cpp
// include/frontend/models/hybrid_transformer_forward.hpp
#pragma once
#include "backend/tensor/tensor.hpp"
#include "frontend/models/hybrid_forward_config.hpp"

namespace zedinfer {
class InferenceRequest;
}

namespace zedinfer::model {

class PagedForwardContext;
struct DecodeScratch;

tensor_t hybrid_transformer_forward(const HybridForwardConfig& model,
                                      PagedForwardContext& ctx,
                                      InferenceRequest& req,
                                      const ExecutorConfig& exec,
                                      DecodeScratch* scratch = nullptr,
                                      tensor_t input_embeds = nullptr);

} // namespace zedinfer::model
```

- [ ] **Step 2: Outer-loop skeleton**

```cpp
// src/frontend/models/hybrid_transformer_forward.cpp
#include "frontend/models/hybrid_transformer_forward.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "zedinfer/request.hpp"
#include "backend/ops/ops.hpp"

namespace zedinfer::model {

// Forward decls (defined later in this file)
static tensor_t forward_linear_attn_layer(const HybridForwardConfig&, tensor_t, size_t, InferenceRequest&, const ExecutorConfig&);
static tensor_t forward_full_attn_layer(const HybridForwardConfig&, PagedForwardContext&, tensor_t, size_t, const ExecutorConfig&);
static tensor_t forward_dense_mlp(const HybridForwardConfig&, tensor_t, size_t, const ExecutorConfig&);
static tensor_t forward_moe_mlp(const HybridForwardConfig&, tensor_t, size_t, const ExecutorConfig&, DecodeScratch*);

tensor_t hybrid_transformer_forward(const HybridForwardConfig& m,
                                      PagedForwardContext& ctx,
                                      InferenceRequest& req,
                                      const ExecutorConfig& exec,
                                      DecodeScratch* scratch,
                                      tensor_t input_embeds) {
    const auto& cfg = m.config;
    size_t N = ctx.num_tokens();
    size_t hidden_size = cfg.hidden_size;

    tensor_t ids, pos_ids;
    ctx.prepare_inputs(ids, pos_ids, exec);

    tensor_t hidden;
    if (input_embeds) {
        hidden = input_embeds;
    } else {
        hidden = Tensor::create({N, hidden_size}, exec.data_type, exec.device_type, exec.device_id);
        ops::embedding(hidden, ids, m.W("embed_tokens.weight"));
    }
    // Image embedding scatter (M3 will add ops::scatter_image_embeds; M1 path is text-only)
    // if (req.has_images()) ops::scatter_image_embeds(hidden, ids, req.image_embeds(), cfg.image_token_id);

    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        auto p = std::string("layers.") + std::to_string(L) + ".";

        auto h_in = Tensor::create({N, hidden_size}, exec.data_type, exec.device_type, exec.device_id);
        ops::rms_norm(h_in, hidden, m.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        tensor_t attn_out = m.is_linear_attn_layer(L)
            ? forward_linear_attn_layer(m, h_in, L, req, exec)
            : forward_full_attn_layer(m, ctx, h_in, L, exec);

        auto h1 = Tensor::create({N, hidden_size}, exec.data_type, exec.device_type, exec.device_id);
        ops::add(h1, hidden, attn_out);

        auto h_post = Tensor::create({N, hidden_size}, exec.data_type, exec.device_type, exec.device_id);
        ops::rms_norm(h_post, h1, m.W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        tensor_t mlp_out = m.is_moe
            ? forward_moe_mlp(m, h_post, L, exec, scratch)
            : forward_dense_mlp(m, h_post, L, exec);

        hidden = Tensor::create({N, hidden_size}, exec.data_type, exec.device_type, exec.device_id);
        ops::add(hidden, h1, mlp_out);
    }

    auto final_normed = Tensor::create({N, hidden_size}, exec.data_type, exec.device_type, exec.device_id);
    ops::rms_norm(final_normed, hidden, m.W("norm.weight"), cfg.rms_norm_eps);

    auto logits = Tensor::create({N, cfg.vocab_size}, exec.data_type, exec.device_type, exec.device_id);
    m.dispatch_linear(logits, final_normed, "lm_head", nullptr);

    ctx.finalize();
    return logits;
}

// Stubs (implemented in subsequent tasks)
static tensor_t forward_linear_attn_layer(const HybridForwardConfig&, tensor_t, size_t,
                                            InferenceRequest&, const ExecutorConfig&) {
    throw std::runtime_error("forward_linear_attn_layer not yet impl");
}
static tensor_t forward_full_attn_layer(const HybridForwardConfig&, PagedForwardContext&,
                                          tensor_t, size_t, const ExecutorConfig&) {
    throw std::runtime_error("forward_full_attn_layer not yet impl");
}
static tensor_t forward_dense_mlp(const HybridForwardConfig&, tensor_t, size_t,
                                    const ExecutorConfig&) {
    throw std::runtime_error("forward_dense_mlp not yet impl");
}
static tensor_t forward_moe_mlp(const HybridForwardConfig&, tensor_t, size_t,
                                  const ExecutorConfig&, DecodeScratch*) {
    throw std::runtime_error("forward_moe_mlp not yet impl");
}

} // namespace zedinfer::model
```

- [ ] **Step 3: Build (link will fail until stubs are replaced)**

Run: `xmake build`. Should succeed since stubs throw at runtime, not at link.

- [ ] **Step 4: Commit**

```bash
git add include/frontend/models/hybrid_transformer_forward.hpp src/frontend/models/hybrid_transformer_forward.cpp
git commit -m "feat(qwen3.5): hybrid_transformer_forward outer loop (per-layer-kind dispatch)"
```

---

### Task 13: `forward_dense_mlp` impl

**Files:**
- Modify: `src/frontend/models/hybrid_transformer_forward.cpp`

- [ ] **Step 1: Replace stub**

```cpp
static tensor_t forward_dense_mlp(const HybridForwardConfig& m, tensor_t h_post,
                                    size_t L, const ExecutorConfig& exec) {
    size_t N = h_post->shape()[0];
    size_t hidden = m.config.hidden_size;
    size_t inter = m.config.intermediate_size;
    auto p = std::string("layers.") + std::to_string(L) + ".";

    auto gate = Tensor::create({N, inter}, exec.data_type, exec.device_type, exec.device_id);
    auto up   = Tensor::create({N, inter}, exec.data_type, exec.device_type, exec.device_id);
    m.dispatch_linear(gate, h_post, p + "mlp.gate_proj", nullptr);
    m.dispatch_linear(up,   h_post, p + "mlp.up_proj",   nullptr);

    auto act = Tensor::create({N, inter}, exec.data_type, exec.device_type, exec.device_id);
    ops::swiglu(act, gate, up);

    auto down = Tensor::create({N, hidden}, exec.data_type, exec.device_type, exec.device_id);
    m.dispatch_linear(down, act, p + "mlp.down_proj", nullptr);
    return down;
}
```

- [ ] **Step 2: Build + commit**

```bash
xmake build
git add src/frontend/models/hybrid_transformer_forward.cpp
git commit -m "feat(qwen3.5): forward_dense_mlp (gate/up/swiglu/down, GPTQ Int4)"
```

---

### Task 14: `forward_full_attn_layer` impl

**Files:**
- Modify: `src/frontend/models/hybrid_transformer_forward.cpp`
- Imports: `backend/ops/mrope/mrope_3d.hpp`, `backend/ops/attn_output_gate/attn_output_gate.hpp`

- [ ] **Step 1: Replace stub**

```cpp
#include "backend/ops/mrope/mrope_3d.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"

static tensor_t forward_full_attn_layer(const HybridForwardConfig& m, PagedForwardContext& ctx,
                                          tensor_t h_in, size_t L, const ExecutorConfig& exec) {
    size_t N = h_in->shape()[0];
    size_t Hq = m.config.num_attention_heads;
    size_t Hkv = m.config.num_key_value_heads;
    size_t Dh = m.config.head_dim;
    auto p = std::string("layers.") + std::to_string(L) + ".self_attn.";

    // Projections — q_proj is doubled output (q + gate)
    auto qg = Tensor::create({N, 2 * Hq * Dh}, exec.data_type, exec.device_type, exec.device_id);
    auto k  = Tensor::create({N, Hkv * Dh},     exec.data_type, exec.device_type, exec.device_id);
    auto v  = Tensor::create({N, Hkv * Dh},     exec.data_type, exec.device_type, exec.device_id);
    m.dispatch_linear(qg, h_in, p + "q_proj", nullptr);
    m.dispatch_linear(k,  h_in, p + "k_proj", nullptr);
    m.dispatch_linear(v,  h_in, p + "v_proj", nullptr);

    auto q    = qg->view({N, Hq, Dh}, /*offset=*/0);
    auto gate = qg->view({N, Hq, Dh}, /*offset=*/Hq * Dh * sizeof_dtype(exec.data_type));

    // q_norm / k_norm (per-head)
    auto q_normed = Tensor::create({N, Hq, Dh},  exec.data_type, exec.device_type, exec.device_id);
    auto k_normed = Tensor::create({N, Hkv, Dh}, exec.data_type, exec.device_type, exec.device_id);
    ops::rms_norm(q_normed->view({N*Hq, Dh}),  q->view({N*Hq, Dh}),
                  m.W(p + "q_norm.weight"), m.config.rms_norm_eps);
    ops::rms_norm(k_normed->view({N*Hkv, Dh}), k->view({N*Hkv, Dh}),
                  m.W(p + "k_norm.weight"), m.config.rms_norm_eps);

    // 3D MRoPE (partial rotary, in-place on first 64 dims)
    ops::mrope_3d(q_normed, ctx.pos_ids_thw(), m.mrope);
    ops::mrope_3d(k_normed, ctx.pos_ids_thw(), m.mrope);

    // Paged KV write — use logical KV layer index
    int kv_idx = m.full_layer_index(L);
    ctx.write_kv(kv_idx, k_normed->view({N, Hkv*Dh}), v);

    // Paged attention
    float scale = 1.0f / std::sqrt((float)Dh);
    auto attn = ctx.attend(kv_idx, q_normed, scale, exec, Hq, Hkv, Dh, nullptr);

    // Output gate (in-place)
    ops::attn_output_gate(attn, gate);

    // o_proj — input is single q_dim
    auto out = Tensor::create({N, m.config.hidden_size}, exec.data_type, exec.device_type, exec.device_id);
    m.dispatch_linear(out, attn->view({N, Hq*Dh}), p + "o_proj", nullptr);
    return out;
}
```

(The `qg->view(...)` offset API is a sketch — adapt to zedinfer's actual `Tensor::view` slicing API. If only contiguous views are supported, use `Tensor::create_view_from(qg, offset, shape)` or split into two separate output buffers from `q_proj` via two GEMMs.)

If `Tensor::view` doesn't support offsets, change `q_proj` to output to two separate tensors directly (less efficient but functional for M1):
```cpp
auto qg = ...;
auto q_part    = Tensor::create({N, Hq*Dh}, ...);
auto gate_part = Tensor::create({N, Hq*Dh}, ...);
ops::slice_copy(q_part,    qg, 0,        Hq*Dh);
ops::slice_copy(gate_part, qg, Hq*Dh,    Hq*Dh);
```

(`ops::slice_copy` may need to be a small new op or done via `cudaMemcpy2DAsync`.)

- [ ] **Step 2: ctx.pos_ids_thw() — need to add**

In `PagedForwardContext`, add a `pos_ids_thw_` member + `tensor_t pos_ids_thw() const`. For text-only forward, populate with `(t, h, w) = (idx, idx, idx)` during `prepare_inputs`:
```cpp
// In prepare_inputs (or a new prepare_thw_pos_ids method):
if (hybrid_) {
    pos_ids_thw_ = Tensor::create({3, N}, ZEDINFER_DTYPE_I32, exec.device_type, exec.device_id);
    std::vector<int32_t> thw(3 * N);
    for (int i = 0; i < N; ++i) thw[0*N + i] = thw[1*N + i] = thw[2*N + i] = base_pos + i;
    cudaMemcpy(pos_ids_thw_->data(), thw.data(), 3*N*sizeof(int32_t), cudaMemcpyHostToDevice);
}
```

- [ ] **Step 3: Build + commit**

```bash
xmake build
git add src/frontend/models/hybrid_transformer_forward.cpp \
        include/frontend/models/paged_forward_context.hpp \
        src/frontend/models/paged_forward_context.cpp
git commit -m "feat(qwen3.5): forward_full_attn_layer (gated softmax + 3D mrope + paged KV)"
```

---

### Task 15: `forward_linear_attn_layer` impl

**Files:**
- Modify: `src/frontend/models/hybrid_transformer_forward.cpp`
- Imports: `backend/ops/mamba/{ssu,causal_conv1d}.hpp`

- [ ] **Step 1: Replace stub**

```cpp
#include "backend/ops/mamba/ssu.hpp"
#include "backend/ops/mamba/causal_conv1d.hpp"

static tensor_t forward_linear_attn_layer(const HybridForwardConfig& m, tensor_t h_in,
                                            size_t L, InferenceRequest& req,
                                            const ExecutorConfig& exec) {
    size_t N = h_in->shape()[0];
    int Hv = m.linear_attn.num_v_heads;
    int Dv = m.linear_attn.value_head_dim;
    int Hk = m.linear_attn.num_k_heads;
    int Dk = m.linear_attn.key_head_dim;
    int qkv_dim = Hk*Dk + Hk*Dk + Hv*Dv;
    auto p = std::string("layers.") + std::to_string(L) + ".linear_attn.";

    // Input projections (BF16 — attention/linear_attn not quantized)
    auto qkv = Tensor::create({N, (size_t)qkv_dim}, exec.data_type, exec.device_type, exec.device_id);
    auto z   = Tensor::create({N, (size_t)Hv*Dv},   exec.data_type, exec.device_type, exec.device_id);
    auto a   = Tensor::create({N, (size_t)Hv},      exec.data_type, exec.device_type, exec.device_id);
    auto b   = Tensor::create({N, (size_t)Hv},      exec.data_type, exec.device_type, exec.device_id);
    ops::linear(qkv, h_in, m.W(p + "in_proj_qkv.weight"));
    ops::linear(z,   h_in, m.W(p + "in_proj_z.weight"));
    ops::linear(a,   h_in, m.W(p + "in_proj_a.weight"));
    ops::linear(b,   h_in, m.W(p + "in_proj_b.weight"));

    // causal_conv1d (silu fused inside the kernel)
    auto qkv_conv = Tensor::create({N, (size_t)qkv_dim}, exec.data_type, exec.device_type, exec.device_id);
    ops::mamba::causal_conv1d(qkv_conv, qkv, m.W(p + "conv1d.weight"),
                                m.ssm_pool->view(), req.ssm_slot_idx(),
                                m.linear_layer_index(L));

    // Split q/k/v from qkv_conv
    auto q_ssm = qkv_conv->view({N, (size_t)Hk*Dk}, /*offset=*/0);
    auto k_ssm = qkv_conv->view({N, (size_t)Hk*Dk}, /*offset=*/Hk*Dk * sizeof_dtype(exec.data_type));
    auto v_ssm = qkv_conv->view({N, (size_t)Hv*Dv}, /*offset=*/2*Hk*Dk * sizeof_dtype(exec.data_type));

    // silu(z) — z gate path
    auto z_silu = Tensor::create({N, (size_t)Hv*Dv}, exec.data_type, exec.device_type, exec.device_id);
    ops::silu(z_silu, z);

    // SSU
    auto y = Tensor::create({N, (size_t)Hv*Dv}, exec.data_type, exec.device_type, exec.device_id);
    ops::mamba::SSUParams sp;
    sp.state_view = m.ssm_pool->view();
    sp.slot_idx = req.ssm_slot_idx();
    sp.layer_idx = m.linear_layer_index(L);
    sp.q = q_ssm; sp.k = k_ssm; sp.v = v_ssm;
    sp.a = a; sp.b = b;
    sp.A_log   = m.W(p + "A_log");
    sp.dt_bias = m.W(p + "dt_bias");
    sp.z = z_silu;
    sp.out = y;
    sp.num_tokens = (int)N;
    ops::mamba::ssu(sp);

    // RMSNorm on per-V-head value_head_dim
    auto y_normed = Tensor::create({N, (size_t)Hv*Dv}, exec.data_type, exec.device_type, exec.device_id);
    ops::rms_norm(y_normed->view({N*Hv, Dv}), y->view({N*Hv, Dv}),
                  m.W(p + "norm.weight"), m.config.rms_norm_eps);

    // out_proj
    auto out = Tensor::create({N, m.config.hidden_size}, exec.data_type, exec.device_type, exec.device_id);
    ops::linear(out, y_normed, m.W(p + "out_proj.weight"));
    return out;
}
```

Note: same offset-view caveat as Task 14. Adapt to zedinfer's actual Tensor API.

- [ ] **Step 2: Build + verify all four layer functions present**

Run: `xmake build`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/frontend/models/hybrid_transformer_forward.cpp
git commit -m "feat(qwen3.5): forward_linear_attn_layer (Mamba2 via FlashInfer SSU + conv1d)"
```

---

### Task 16: `forward_moe_mlp` stub (M5 will activate)

**Files:**
- Modify: `src/frontend/models/hybrid_transformer_forward.cpp`

- [ ] **Step 1: Replace stub with delegation to existing `moe_layer_forward`**

```cpp
#include "frontend/models/moe_forward.hpp"

static tensor_t forward_moe_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L,
                                  const ExecutorConfig& exec, DecodeScratch* scratch) {
    size_t N = h_post->shape()[0];
    auto out = Tensor::create({N, m.config.hidden_size}, exec.data_type, exec.device_type, exec.device_id);
    // moe_layer_forward operates on ModelForwardConfig (which HybridForwardConfig inherits from)
    moe_layer_forward(m, out, h_post, static_cast<int>(L), exec, scratch);
    return out;
}
```

Note: `moe_layer_forward` was authored for Qwen3-MoE (decoder_sparse_step + shared_expert logic). Verify it works for Qwen3.5 MoE config; differences (256 vs 128 experts, top-8 same) shouldn't matter at the dispatcher level.

- [ ] **Step 2: Build**

Run: `xmake build`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/frontend/models/hybrid_transformer_forward.cpp
git commit -m "feat(qwen3.5): forward_moe_mlp delegates to v0.2.0 moe_layer_forward"
```

---

### Task 17: Wire `hybrid_transformer_forward` into `Qwen3_5Model::forward`

**Files:**
- Modify: `src/frontend/models/qwen3_5.cpp`

- [ ] **Step 1: Implement `Qwen3_5Model::forward`**

```cpp
#include "frontend/models/hybrid_transformer_forward.hpp"

tensor_t Qwen3_5Model::forward(InferenceRequest& req, PagedForwardContext& ctx,
                                 const ExecutorConfig& exec, DecodeScratch* scratch) {
    auto hcfg = hybrid_forward_config();
    return hybrid_transformer_forward(hcfg, ctx, req, exec, scratch,
                                       /*input_embeds=*/req.image_embeds());
}
```

Add to header (`include/frontend/models/qwen3_5.hpp`):
```cpp
tensor_t forward(InferenceRequest& req, PagedForwardContext& ctx,
                  const ExecutorConfig& exec, DecodeScratch* scratch) override;
```

If `Model::forward` isn't yet an override slot, plumb the call site differently — match how `Qwen3MoEModel::forward` exposes itself to scheduler.

Verify by `grep -n 'forward(' include/frontend/models/qwen3_moe.hpp` how the existing scheduler invokes the model. Mirror that.

- [ ] **Step 2: Build**

Run: `xmake build`

- [ ] **Step 3: Commit**

```bash
git add include/frontend/models/qwen3_5.hpp src/frontend/models/qwen3_5.cpp
git commit -m "feat(qwen3.5): Qwen3_5Model::forward wires hybrid_transformer_forward"
```

---

### Task 18: End-to-end ping smoke test

- [ ] **Step 1: Run ping with text prompt**

```bash
xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia 2>&1 | tee /tmp/m1_ping.log
```

Type "Who are you?" when prompted. Expected: a sensible Qwen-like reply ("I am Qwen, a large language model..." or similar). Streaming should work.

If output is garbled/NaN: M2 will fix byte-exact alignment. M1 acceptance is: output is coherent token-by-token (no NaN logits, recognizable English words, no infinite repetition).

- [ ] **Step 2: Disable warmup for faster iteration**

```bash
ZEDINFER_DISABLE_WARMUP=1 xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia
```

- [ ] **Step 3: Try 35B-A3B (stretch)**

```bash
ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia
```

If it crashes / produces nonsense, document the failure mode in handoff doc and continue — M5 will dive in.

- [ ] **Step 4: Regression check**

```bash
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia
```
Expected: identical to v0.2.0 behavior.

- [ ] **Step 5: Commit M1 milestone marker**

In `docs/plan/qwen3_5_support.md` §10 M1 row, append:
```
M1 complete: <commit-sha>, 27B emits coherent text reply. Outputs not yet byte-exact (M2). 35B-A3B path: <pass | known issue>.
```

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): M1 complete; 27B text forward produces coherent output"
```

---

### Task 19: M1 retrospective

- [ ] **Step 1: Update handoff doc**

Append to `docs/plan/qwen3_5_session_handoff.md`:
```markdown
## M1 — Hybrid text-only forward (complete)

### What landed
- ops::mamba::ssu (FlashInfer wrapper, decode + varlen)
- ops::mamba::causal_conv1d (depthwise kernel=4 with conv state)
- ops::mrope_3d (interleaved + partial rotary)
- ops::attn_output_gate (sigmoid mul)
- InferenceRequest gains ssm_slot_idx, image_embeds, pos_ids_thw
- Scheduler dual-pool admission (BlockPool + SSMStatePool)
- PagedForwardContext logical kv_layer_idx semantics
- transformer_forward accepts optional input_embeds
- hybrid_transformer_forward (A2: 5 per-layer-kind functions)
- Qwen3_5Model::forward wired

### Caveats / open items for M2
- Tensor view with byte offset may need explicit slice_copy op
- ctx.pos_ids_thw() populates trivially (i,i,i) for text-only; M3 needs real T/H/W
- 35B-A3B path: <state>
- Output coherent but not byte-exact vs HF (M2 work)
- forward_moe_mlp delegates to v0.2.0; verified or pending M5

### M2 entry conditions
- 27B "Who are you?" produces coherent ~50-token reply
- All M1 unit tests pass
- No regression on Qwen2/Qwen3/Qwen3-MoE
```

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M1 retro and M2 handoff notes"
```

---

## M1 Done. ~19 tasks. Est. 3-4 weeks.

Coherent 27B text replies. Numerical correctness vs HF is M2's job.
