# GatedDeltaNet Kernel for Ampere/Ada Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wrong Mamba2 SSU primitive in Qwen3.5's `forward_linear_attn_layer` with a custom GatedDeltaNet CUDA kernel that runs on Ampere (sm_86, RTX 3090 / A6000) and Ada (sm_89, RTX 4090). Target outcome: `ping Qwen3.5-35B-A3B-GPTQ-Int4` produces coherent English instead of all-`!`.

**Architecture:**
- Two kernel paths under one op (`ops::mamba::gdn`): decode (N=1, in-place state update) and prefill (N>1, sequential token loop). Dispatch by `num_tokens` inside the wrapper.
- One CTA per V-head per request slot. Threads in the CTA cooperate over (Dv, Dk) tiles via warp-cooperative reduction. No TMA, no WGMMA — plain `cp.async` and scalar/vector math suitable for sm_86.
- State buffer layout (`[slots, layers, Hv, Dv, Dk]` fp32) reuses the existing `SSMStatePool` since in Qwen3.5 `dstate == Dk == 128`; only the interpretation of state changes.
- Numerical reference is a PyTorch implementation of the recurrence (in a `tests/scripts/` offline script); fixtures are pre-generated `.bin` files compared bit-by-bit (with a bf16 tolerance) in gtest C++ unit tests. No Python at serving time.

**Tech Stack:** CUDA C++ (kernel) · xmake (build) · gtest (unit tests) · PyTorch in an offline venv (fixture generation only).

**Prerequisites carried in from P3-debug (already on `feat/qwen3.5`):**
- `9b1f56b` — SSU operand mapping correction (kept for reference; will be removed once GDN kernel lands)
- `1066b94` — P3 critical finding written into `docs/plan/qwen3_5_session_handoff.md`
- `f0f1d59` — Passthrough experiment recorded; confirms NaN cascade hypothesis

**Out of scope (deferred):**
- Performance tuning beyond "correctness first" — speed work belongs to M6 / P7.
- Chunked algorithm (Yang et al. 2025 §3.3 sub-matrices); recurrent prefill is correct and fast enough for ping.
- `27B-GPTQ-Int4` end-to-end smoke — blocked by separate host pinned-memory `ulimit -l` issue, unrelated to this work.
- Multi-batch (the scheduler already binds N to a single request slot per call; bs>1 inference is M6).
- INT8/INT4 quantization of the GDN op weights — `in_proj_*.weight` are bf16 by Qwen3.5 quant policy, so this is not needed.

**Filename:** `docs/plan/qwen3_5_p3a_gdn_kernel.md`. This plan supersedes `qwen3_5_p3_m2_token_alignment.md`'s blocker; that plan becomes the M2 follow-up after this kernel lands.

---

## File Structure

**New files:**

| Path | Purpose |
|---|---|
| `tests/scripts/gen_gdn_fixtures.py` | PyTorch reference + binary fixture generator. Run once, output checked into `tests/data/gdn_fixtures/`. |
| `tests/data/gdn_fixtures/decode_n1.bin` | Decode fixture: 1 token, Hv=8, Hk=4, Dv=128, Dk=128. |
| `tests/data/gdn_fixtures/prefill_n4.bin` | Prefill fixture: 4 tokens, same head dims. |
| `tests/data/gdn_fixtures/prefill_n12.bin` | Prefill fixture: 12 tokens, same head dims. |
| `include/backend/ops/mamba/gdn.hpp` | Op interface header: `GDNParams` struct + `gdn(params)` entry point. |
| `src/backend/ops/mamba/nvidia/gdn_kernel.cuh` | Shared device math: `compute_gate_decay`, `compute_beta`, one-token state-step inline functions. |
| `src/backend/ops/mamba/nvidia/gdn_decode.cu` | Decode kernel (N=1). |
| `src/backend/ops/mamba/nvidia/gdn_prefill.cu` | Prefill kernel (sequential token loop, varlen `cu_seqlens=[0,N]`). |
| `src/backend/ops/mamba/nvidia/gdn_wrapper.cu` | Dispatch wrapper: `ops::mamba::gdn` entry, picks decode vs prefill, manages cu_seqlens scratch. |
| `tests/ops/test_gdn.cu` | gtest target: 3 cases (decode_n1, prefill_n4, prefill_n12). |
| `xmake/tests.lua` | Add `test-gdn` target. |

**Modified files:**

| Path | Modification |
|---|---|
| `src/frontend/models/hybrid_transformer_forward.cpp` | Replace `ops::mamba::ssu(sp)` with `ops::mamba::gdn(gp)` in `forward_linear_attn_layer`; add silu(z)*y gating between norm and out_proj. |
| `docs/plan/qwen3_5_session_handoff.md` | Append M2 retrospective after kernel lands. |
| `docs/roadmap.md` | Mark M2 complete, move "token-byte-exact alignment" to M2-follow-up. |

**Files removed:**

None. `include/backend/ops/mamba/ssu.hpp` and `src/backend/ops/mamba/nvidia/ssu_wrapper.cu` stay — they are correct Mamba2 SSU code that future models (real Mamba2-based) may want. The `test-mamba-ssu` test target stays.

---

## Task 1: Document the exact GDN recurrence math

**Files:**
- Modify: `docs/plan/qwen3_5_p3a_gdn_kernel.md` (this file — the "GDN Math Reference" section below).

The math must be precise before any kernel coding. fla-org's `recurrent_gated_delta_rule` is the canonical reference; HF transformers v5.5.4 calls it through `torch_recurrent_gated_delta_rule`.

- [x] **Step 1: Pull fla-org reference source**

Run:
```bash
mkdir -p /tmp/fla-ref
cd /tmp/fla-ref
git clone --depth 1 https://github.com/fla-org/flash-linear-attention fla
find fla -name "*.py" -path "*gated_delta*" -not -path "*test*"
```
Expected: at least `fla/fla/ops/gated_delta_rule/naive.py` and `fla/fla/ops/gated_delta_rule/chunk.py`.

- [x] **Step 2: Extract the recurrent reference**

Open `fla/fla/ops/gated_delta_rule/naive.py` and locate the function named `naive_recurrent_gated_delta_rule` (or close). Copy its body verbatim into a fenced code block in this plan's "GDN Math Reference" section below, with a one-line comment per line explaining what it does in our variable names (`b → beta`, `a + dt_bias → softplus arg`, `A_log → log-decay base`).

- [x] **Step 3: Cross-check against HF**

Run:
```bash
cd /tmp/fla-ref
git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/transformers transformers-ref
cd transformers-ref
git sparse-checkout set src/transformers/models/qwen3_5_moe
ls src/transformers/models/qwen3_5_moe/
grep -n "GatedDeltaNet\|gated_delta_rule" src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py | head -30
```
Expected: matches for `Qwen3_5MoeGatedDeltaNet` class and calls into `torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule`. Read the `forward` method of `Qwen3_5MoeGatedDeltaNet` and confirm:
  - `q_proj` produces an output that is *split* into (q, attn-output-gate-z)
  - `b_proj` / `a_proj` / `in_proj_qkv` produce raw `b`, `a` (not yet sigmoid/softplus'd)
  - silu(z) is applied to `o_norm` *after* the rule, not inside

- [x] **Step 4: Fill in the math reference section below with verbatim quoted code**

Replace the placeholder block in "GDN Math Reference" section below with the actual fla-org / HF code, attributed by file + line.

- [x] **Step 5: Commit**

```bash
git add docs/plan/qwen3_5_p3a_gdn_kernel.md
git commit -m "docs(qwen3.5): document exact GDN recurrence from fla-org + HF reference"
```

---

### GDN Math Reference

#### fla-org canonical recurrent reference

Source: `fla/fla/ops/gated_delta_rule/naive.py`, function `naive_recurrent_gated_delta_rule`,
commit `5aea42b7740f9968f6418c6c60b78ea785ce6140` in `fla-org/flash-linear-attention`.

```python
# Verbatim copy (no added annotations) — fla/fla/ops/gated_delta_rule/naive.py lines 13-64, commit 5aea42b
def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h
```

#### Variable mapping (fla → zedinfer)

- `q, k` (fla `[B, T, H, K]`) → our `q, k` (per-token `[Dk]`)
- `v` (fla `[B, T, H, V]`) → our `v` (per-token `[Dv]`)
- `beta` (fla `[B, T, H]`) → our `beta = sigmoid(b)` from `in_proj_b`
- `g` (fla `[B, T, H]`) → our log-decay; HF Qwen3.5 caller computes `g = -exp(A_log) * softplus(a + dt_bias)` before calling
- `h` (fla state `[B, H, K, V]`) → our `S` (state `[Dv, Dk]` — same math, transposed layout)

**Key observations from the verbatim code:**

1. **`g` vs `A_log`/`dt_bias`:** The naive function receives `g` already computed (log-decay
   in log-space). In the fla `GatedDeltaNet` layer (`gated_deltanet.py` line 266–296), `g` is
   passed as `self.a_proj(hidden_states)` (raw `a`), then the kernel applies
   `-exp(A_log) * softplus(a + dt_bias)` internally (`USE_GATE_IN_KERNEL=True`). In HF
   Qwen3.5's `Qwen3_5MoeGatedDeltaNet.forward` (line 514), `g` is pre-computed by the *caller*:
   ```python
   g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
   ```
   The passed `g` is therefore already the final log-decay (a negative number). The recurrence
   then uses `exp(g)` as the multiplicative gate, i.e. `decay = exp(g)`, which equals
   `exp(-exp(A_log) * softplus(a + dt_bias))`. This confirms the plan's placeholder formula was
   correct.

2. **Decay applies to the FULL prior state before the new update:** Line
   `h = h.clone() * g[:, :, i].exp()[..., None, None]` comes *before* the beta term is added.
   The update is:
   ```
   S_t = exp(g_t) * S_{t-1}  +  outer(beta_t * (v_t - exp(g_t)*S_{t-1} @ k_t), k_t)
       = decay * S_{t-1}     +  beta_t * outer(v_t - decay * S_{t-1} @ k_t, k_t)
   ```
   (Argument order: `outer(delta_v, k)` matches our `S [Dv, Dk]` row-major layout where each row
   is one V-head value slot and each column is one K-head key slot. fla's `h [K, V]` uses the
   transposed convention; the math is identical.)
   This matches Yang et al. 2025 §3.1. The plan's placeholder had the correct form.

3. **State layout:** fla uses `h [K, V]` (key-dim × value-dim). Our kernel uses `S [Dv, Dk]`
   (value-dim × key-dim), i.e. the transpose. The math is identical; the kernel reads/writes
   transposed relative to fla's convention.

4. **beta:** Applied inside the loop as `b_v * b_beta[..., None]` — it scales the *error
   (delta)* vector, not the outer product coefficient separately. This is equivalent to the
   plan's `outer(delta_v, k)` formulation since `beta * outer(delta_v, k) = outer(beta*delta_v, k)`.

#### HF Qwen3.5 call site

Source: `src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`,
commit `0b25f8c49c37530ce9f8742d7a8c19ed8d254d7d` in `huggingface/transformers`.

Relevant lines from `Qwen3_5MoeGatedDeltaNet.forward` (lines 456–554):

```python
# (1) q/k/v come from a single fused projection + causal conv1d:
mixed_qkv = self.in_proj_qkv(hidden_states)    # [key_dim*2 + value_dim] split below
# NOTE: there is NO q_proj that produces a (q, z) split.
# The output gate z comes from a SEPARATE projection:
z = self.in_proj_z(hidden_states)              # [value_dim] — the output gate

# (2) b and a come from their own projections (raw, no sigmoid/softplus yet):
b = self.in_proj_b(hidden_states)              # [num_v_heads]  — raw beta input
a = self.in_proj_a(hidden_states)              # [num_v_heads]  — raw gate input

# ... conv1d applied to mixed_qkv ...
query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)

# (3) Sigmoid for beta, full g formula applied by caller before passing to kernel:
beta = b.sigmoid()                             # beta = sigmoid(b)   in (0,1)
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
                                               # g = -exp(A_log) * softplus(a + dt_bias)   (log-decay, <=0)

# (4) Recurrence kernel called with pre-computed g (NOT raw a):
core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
    query, key, value, g=g, beta=beta, ...)

# (5) Output gate applied INSIDE norm (NOT as a separate silu-then-mul):
core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
z             = z.reshape(-1, self.head_v_dim)
core_attn_out = self.norm(core_attn_out, z)    # norm is RMSNormGated: rms_norm(x) * silu(z)
```

The `Qwen3_5MoeRMSNormGated.forward` (lines 183–198) performs:
```python
hidden_states = hidden_states * torch.rsqrt(variance + eps)   # RMS normalize
hidden_states = self.weight * hidden_states                    # learned scale
hidden_states = hidden_states * F.silu(gate)                   # silu(z) gate applied here
```

**Implication for our kernel:** The silu(z) output gate is applied *inside* the norm operation in
HF — not as a separate step after the norm. Our plan (Task 9) has it as a separate step
(`ops::silu_mul` after `ops::rms_norm`), which produces the same result as long as the fused
norm and the silu are applied together. The fused `FusedRMSNormGated` from fla modules is the
fast path; if we split it into `rms_norm` then `silu_mul`, the math is equivalent.

#### Confirmed recurrence formula (kernel-ready)

For one token step, per V-head, with our `[Dv, Dk]` state layout:

```
# Inputs (all for this one token, this one V-head):
#   q[Dk], k[Dk]  — post-conv, l2-normed (scale = 1/sqrt(Dk) applied to q)
#   v[Dv]         — post-conv
#   b (scalar)    — raw beta input (from in_proj_b)
#   a (scalar)    — raw gate input (from in_proj_a)
#   A_log (scalar, persistent weight, fp32)
#   dt_bias (scalar, persistent weight, bf16)
#   S [Dv, Dk]    — fp32 state, in-place updated
#
# Per-head scalars (computed in-kernel):
beta  = sigmoid(b)                                     # in (0, 1)
g     = -exp(A_log) * softplus(a + dt_bias)            # log-decay, <= 0
decay = exp(g)                                         # in (0, 1]
#
# State update (confirmed: decay multiplies the FULL prior S before beta term):
Sk      = S @ k                                        # [Dv];  key projection of state
delta_v = v - Sk                                       # [Dv];  reconstruction error
S       = decay * S + beta * outer(delta_v, k)         # [Dv, Dk];  state update
#  (equivalent to fla's: h = exp(g)*h + outer(k, beta*(v - exp(g)*h @ k))
#   with h transposed to our [Dv,Dk] convention)
y       = S @ q                                        # [Dv];  readout (pre-norm)
#
# Output gate (applied AFTER kernel, in RMSNormGated):
y_out = rms_norm(y) * silu(z)                          # z from in_proj_z, separate projection
```

---

## Task 2: PyTorch reference + binary fixture generator

**Files:**
- Create: `tests/scripts/gen_gdn_fixtures.py`
- Create: `tests/data/gdn_fixtures/decode_n1.bin`
- Create: `tests/data/gdn_fixtures/prefill_n4.bin`
- Create: `tests/data/gdn_fixtures/prefill_n12.bin`

- [x] **Step 1: Set up a torch venv for offline use**

Run:
```bash
python3 -m venv /tmp/zedinfer-gdn-venv
source /tmp/zedinfer-gdn-venv/bin/activate
pip install --quiet torch numpy
python3 -c "import torch; print(torch.__version__)"
```
Expected: torch version printed, no error. **This venv is OFFLINE-only for fixture generation. Per `CLAUDE.md` rule #5, Python is forbidden at serving time. The generated `.bin` fixtures are what's used by the C++ test.**

- [x] **Step 2: Write `gen_gdn_fixtures.py`**

Create `tests/scripts/gen_gdn_fixtures.py`:

```python
"""GDN fixture generator. Offline only — produces .bin files used by test_gdn.cu.

Per CLAUDE.md rule #5, this script is NOT part of the serving path. Run it once
to generate tests/data/gdn_fixtures/*.bin, then commit those binaries.
"""
import struct
import torch
from pathlib import Path

torch.manual_seed(0xCAFE)

# Match the smaller of Qwen3.5-27B / 35B-A3B head dims for fixtures.
# Hv >= 8 exercises multi-warp paths; Hk=4 exercises grouping.
Hv, Hk = 8, 4
Dv, Dk = 128, 128

OUT = Path(__file__).resolve().parent.parent / "data" / "gdn_fixtures"
OUT.mkdir(parents=True, exist_ok=True)

def naive_recurrent_gdn(q, k, v, b, a, A_log, dt_bias, S0):
    """Reference implementation: one batch, multiple tokens, multiple V-heads.

    q, k:    [N, Hk, Dk]  bf16
    v:       [N, Hv, Dv]  bf16
    b, a:    [N, Hv]      bf16
    A_log:   [Hv]         f32 (persistent)
    dt_bias: [Hv]         bf16 (persistent)
    S0:      [Hv, Dv, Dk] f32 (initial state)

    Returns y [N, Hv, Dv] bf16, S_T [Hv, Dv, Dk] f32.
    """
    N, _, _ = v.shape
    S = S0.clone().float()
    ys = []
    for t in range(N):
        beta  = torch.sigmoid(b[t].float())                                  # [Hv]
        decay = torch.exp(-torch.exp(A_log) *
                          torch.nn.functional.softplus(a[t].float()
                                                        + dt_bias.float())) # [Hv]
        # Map K-head -> V-head broadcasting
        rep = Hv // Hk
        k_v = k[t].float().repeat_interleave(rep, dim=0)  # [Hv, Dk]
        q_v = q[t].float().repeat_interleave(rep, dim=0)  # [Hv, Dk]
        # Per-V-head update
        # S: [Hv, Dv, Dk]   ; S @ k: [Hv, Dv]
        Sk    = torch.einsum('vdk,vk->vd', S, k_v)        # [Hv, Dv]
        dv    = v[t].float() - Sk                          # [Hv, Dv]
        S     = decay.view(Hv, 1, 1) * S \
              + beta.view(Hv, 1, 1)  * torch.einsum('vd,vk->vdk', dv, k_v)
        y     = torch.einsum('vdk,vk->vd', S, q_v)         # [Hv, Dv]
        ys.append(y.to(torch.bfloat16))
    return torch.stack(ys, dim=0), S

def gen(N: int, name: str):
    q   = torch.randn(N, Hk, Dk, dtype=torch.bfloat16) * 0.1
    k   = torch.randn(N, Hk, Dk, dtype=torch.bfloat16) * 0.1
    v   = torch.randn(N, Hv, Dv, dtype=torch.bfloat16) * 0.5
    b   = torch.randn(N, Hv, dtype=torch.bfloat16) * 1.0
    a   = torch.randn(N, Hv, dtype=torch.bfloat16) * 0.1 - 2.0  # softplus arg ~ -2 → small decay
    A_log   = torch.randn(Hv, dtype=torch.float32) - 1.0
    dt_bias = torch.randn(Hv, dtype=torch.bfloat16) * 0.1
    S0      = torch.zeros(Hv, Dv, Dk, dtype=torch.float32)
    y, S_T  = naive_recurrent_gdn(q, k, v, b, a, A_log, dt_bias, S0)

    # Header: 6 i32 (N, Hv, Hk, Dv, Dk, _reserved=0)
    # Tensors (bf16 little-endian, then f32 little-endian for the f32s):
    #   q, k, v, b, a       (bf16)
    #   A_log               (f32)
    #   dt_bias             (bf16)
    #   S0                  (f32)
    #   y_expected          (bf16)
    #   S_T_expected        (f32)
    p = OUT / name
    with p.open("wb") as f:
        f.write(struct.pack("<6i", N, Hv, Hk, Dv, Dk, 0))
        for t in [q, k, v, b, a]:           f.write(t.contiguous().view(torch.int16).numpy().tobytes())
        f.write(A_log.contiguous().numpy().tobytes())
        f.write(dt_bias.contiguous().view(torch.int16).numpy().tobytes())
        f.write(S0.contiguous().numpy().tobytes())
        f.write(y.contiguous().view(torch.int16).numpy().tobytes())
        f.write(S_T.contiguous().numpy().tobytes())
    print(f"wrote {p}  N={N}  size={p.stat().st_size}")

if __name__ == "__main__":
    gen(1,  "decode_n1.bin")
    gen(4,  "prefill_n4.bin")
    gen(12, "prefill_n12.bin")
```

- [x] **Step 3: Run the generator**

Run:
```bash
source /tmp/zedinfer-gdn-venv/bin/activate
python3 tests/scripts/gen_gdn_fixtures.py
ls -la tests/data/gdn_fixtures/
```
Expected: three `.bin` files printed, sizes roughly: decode_n1 ~340KB, prefill_n4 ~360KB, prefill_n12 ~400KB (state dominates; tokens are small).

- [x] **Step 4: Commit the script + the fixtures**

```bash
git add tests/scripts/gen_gdn_fixtures.py tests/data/gdn_fixtures/
git commit -m "test(ops): GDN fixture generator + 3 reference binaries (decode_n1, prefill_n4, prefill_n12)"
```

---

## Task 3: GDN op header — `GDNParams` + `ops::mamba::gdn` declaration

**Files:**
- Create: `include/backend/ops/mamba/gdn.hpp`

This mirrors the `SSUParams` pattern from `ssu.hpp` so the integration call site in `hybrid_transformer_forward.cpp` only changes the param-struct construction.

- [x] **Step 1: Write the header**

Create `include/backend/ops/mamba/gdn.hpp`:

```cpp
#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

namespace zedinfer::ops::mamba {

// Parameters for the Qwen3.5 Gated Delta Rule kernel.
//
// Per-token, per-V-head computation:
//   beta  = sigmoid(b)
//   decay = exp( -exp(A_log) * softplus(a + dt_bias) )
//   delta = v - S @ k
//   S     = decay * S + beta * outer(delta, k)
//   y     = S @ q
//
// Shapes (N = num_tokens this call):
//   q, k:    [N, num_k_heads * key_head_dim]  bf16 (post-conv, post-norm)
//   v:       [N, num_v_heads * value_head_dim] bf16 (post-conv)
//   b, a:    [N, num_v_heads]                 bf16 (raw, sigmoid/softplus applied in-kernel)
//   A_log:   [num_v_heads]                    f32  persistent weight
//   dt_bias: [num_v_heads]                    bf16 persistent weight
//   out:     [N, num_v_heads * value_head_dim] bf16 (pre-norm; caller applies rms_norm + silu(z) outside)
//
// State (in SSMStatePool):
//   slot[layer_idx]: [num_v_heads, value_head_dim, key_head_dim] f32
//   Reused from the existing pool — Qwen3.5 d_state == key_head_dim, so
//   the storage size is unchanged from M1. The interpretation flips from
//   "diagonal SSM state" to "delta-rule matrix state".
struct GDNParams {
    // State pool view + addressing
    model::SSMStateView state_view;
    int slot_idx = -1;
    int layer_idx = -1;

    // Inputs
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t b;       // beta input (sigmoid applied in-kernel)
    tensor_t a;       // gate input (softplus + decay applied in-kernel)

    // Persistent weights
    tensor_t A_log;
    tensor_t dt_bias;

    // Pre-allocated output [N, num_v_heads * value_head_dim] bf16.
    tensor_t out;

    // Number of tokens this call:
    //   1   → decode kernel (single-step, in-place state update).
    //   >1  → prefill kernel (sequential token loop; caller passes cu_seqlens=[0,N] semantically).
    int num_tokens = 0;
};

// Dispatch one GDN step against the pool slot.
//
// Side effects:
//   - state_view's ssm buffer at (slot_idx, layer_idx) is updated in place.
//   - out is written with the per-V-head readout for each input token.
//
// Caller contract:
//   - reset_slot(slot_idx) must have been called once at request start so
//     the state is zeroed before the first call for this request.
//   - Compute stream is the runtime's compute stream.
void gdn(const GDNParams& params);

} // namespace zedinfer::ops::mamba
```

- [x] **Step 2: Verify it includes cleanly**

Run:
```bash
xmake build -j1 2>&1 | tail -3
```
Expected: build ok (header is unused so far, only adds to the include surface; the include path resolves via existing `ssm_state_pool.hpp`).

- [x] **Step 3: Commit**

```bash
git add include/backend/ops/mamba/gdn.hpp
git commit -m "feat(ops): add ops::mamba::gdn header (GDNParams + entry declaration)"
```

---

## Task 4: GDN shared device math (`gdn_kernel.cuh`)

**Files:**
- Create: `src/backend/ops/mamba/nvidia/gdn_kernel.cuh`

This holds device-side inline functions used by both decode and prefill kernels: per-head scalar gate/beta computation, and the per-V-head one-token state-step. Keeping this in a `.cuh` avoids duplicating math between decode and prefill, and gives unit-testable building blocks if we later add a tile test.

- [x] **Step 1: Write the .cuh**

Create `src/backend/ops/mamba/nvidia/gdn_kernel.cuh`:

```cpp
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
```

- [x] **Step 2: Verify it parses with nvcc**

Run:
```bash
xmake build -j1 2>&1 | tail -3
```
Expected: build ok (header is not yet included by any .cu).

- [x] **Step 3: Commit**

```bash
git add src/backend/ops/mamba/nvidia/gdn_kernel.cuh
git commit -m "feat(ops): GDN shared device math (prepare_scalars / dot_row / update_row / readout_row)"
```

---

## Task 5: Decode kernel (N=1)

**Files:**
- Create: `src/backend/ops/mamba/nvidia/gdn_decode.cu`

Design: one CTA per V-head; 32 threads (one warp); each thread handles `Dk / 32 = 4` K-columns. The full Dv=128 (or 256) is iterated as an outer loop within the CTA. State row reads/writes hit GMEM directly (no SMEM staging — each row is only touched once per token).

- [ ] **Step 1: Write the kernel and host launcher**

Create `src/backend/ops/mamba/nvidia/gdn_decode.cu`:

```cpp
#include "backend/ops/mamba/gdn.hpp"
#include "src/backend/ops/mamba/nvidia/gdn_kernel.cuh"

#include "backend/core/context/context.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace zedinfer::ops::mamba {

namespace {

using device::__nv_bfloat16;

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
```

- [ ] **Step 2: Build (kernel still has no caller, just needs to compile)**

Run:
```bash
xmake build -j1 2>&1 | tail -5
```
Expected: build ok. If linker complains about `gdn_decode_launch` being unused, that's fine (will be called from Task 7's wrapper).

- [ ] **Step 3: Commit**

```bash
git add src/backend/ops/mamba/nvidia/gdn_decode.cu
git commit -m "feat(ops): GDN decode kernel (one CTA per V-head, warp-cooperative Dk reduction)"
```

---

## Task 6: Prefill kernel (sequential token loop)

**Files:**
- Create: `src/backend/ops/mamba/nvidia/gdn_prefill.cu`

Design: same kernel shape as decode (one CTA per V-head, 32 threads) but the outermost iteration is over tokens. The per-token math is identical; just loops `for t in [0..N)`. This is the **naive recurrent** prefill — correct but linear in N. A future task can replace with a chunked version, but ping-time prefill is N≤128 tokens, so this is fine.

- [ ] **Step 1: Write the kernel and host launcher**

Create `src/backend/ops/mamba/nvidia/gdn_prefill.cu`:

```cpp
#include "backend/ops/mamba/gdn.hpp"
#include "src/backend/ops/mamba/nvidia/gdn_kernel.cuh"

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
```

- [ ] **Step 2: Build**

Run:
```bash
xmake build -j1 2>&1 | tail -3
```
Expected: build ok.

- [ ] **Step 3: Commit**

```bash
git add src/backend/ops/mamba/nvidia/gdn_prefill.cu
git commit -m "feat(ops): GDN prefill kernel (naive sequential token loop, one CTA per V-head)"
```

---

## Task 7: GDN dispatch wrapper (`gdn_wrapper.cu`)

**Files:**
- Create: `src/backend/ops/mamba/nvidia/gdn_wrapper.cu`

This is the public `ops::mamba::gdn` entry: validates the params, picks decode vs prefill, and invokes the right launcher. Mirrors the structure of `src/backend/ops/mamba/nvidia/ssu_wrapper.cu` (already in the repo).

- [ ] **Step 1: Write the wrapper**

Create `src/backend/ops/mamba/nvidia/gdn_wrapper.cu`:

```cpp
#include "backend/ops/mamba/gdn.hpp"

#include <stdexcept>

namespace zedinfer::ops::mamba {

void gdn_decode_launch(const GDNParams& p);
void gdn_prefill_launch(const GDNParams& p);

void gdn(const GDNParams& p) {
    if (p.num_tokens <= 0) {
        throw std::runtime_error("[ops::mamba::gdn] num_tokens must be >= 1");
    }
    if (!p.state_view.ssm_base) {
        throw std::runtime_error("[ops::mamba::gdn] state_view.ssm_base is null");
    }
    if (p.num_tokens == 1) {
        gdn_decode_launch(p);
    } else {
        gdn_prefill_launch(p);
    }
}

} // namespace zedinfer::ops::mamba
```

- [ ] **Step 2: Build**

Run:
```bash
xmake build -j1 2>&1 | tail -3
```
Expected: build ok; the `gdn` symbol is now visible to callers.

- [ ] **Step 3: Commit**

```bash
git add src/backend/ops/mamba/nvidia/gdn_wrapper.cu
git commit -m "feat(ops): GDN dispatch wrapper (decode vs prefill by num_tokens)"
```

---

## Task 8: gtest unit test against PyTorch fixtures

**Files:**
- Create: `tests/ops/test_gdn.cu`
- Modify: `xmake/tests.lua` (add `test-gdn` target)

This loads each `.bin` fixture, copies inputs to device, calls `ops::mamba::gdn`, and compares `out`/`S_T` against the expected outputs from the fixture. Tolerance: bf16 means we accept |a - b| < max(1e-2, 1e-2 * |b|). State is fp32; tolerance 1e-4.

- [ ] **Step 1: Write the test**

Create `tests/ops/test_gdn.cu`:

```cpp
#include "backend/ops/mamba/gdn.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "backend/core/context/context.hpp"

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace zedinfer;

namespace {

struct Fixture {
    int N, Hv, Hk, Dv, Dk;
    std::vector<uint16_t> q, k, v, b, a;
    std::vector<float>    A_log;
    std::vector<uint16_t> dt_bias;
    std::vector<float>    S0;
    std::vector<uint16_t> y_expected;
    std::vector<float>    S_T_expected;
};

template<typename T>
void read_block(std::ifstream& f, std::vector<T>& v, size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
}

Fixture load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("fixture not found: " + path);
    Fixture fx;
    int32_t hdr[6]{};
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    fx.N=hdr[0]; fx.Hv=hdr[1]; fx.Hk=hdr[2]; fx.Dv=hdr[3]; fx.Dk=hdr[4];
    read_block(f, fx.q,  (size_t)fx.N * fx.Hk * fx.Dk);
    read_block(f, fx.k,  (size_t)fx.N * fx.Hk * fx.Dk);
    read_block(f, fx.v,  (size_t)fx.N * fx.Hv * fx.Dv);
    read_block(f, fx.b,  (size_t)fx.N * fx.Hv);
    read_block(f, fx.a,  (size_t)fx.N * fx.Hv);
    read_block(f, fx.A_log,   (size_t)fx.Hv);
    read_block(f, fx.dt_bias, (size_t)fx.Hv);
    read_block(f, fx.S0,            (size_t)fx.Hv * fx.Dv * fx.Dk);
    read_block(f, fx.y_expected,    (size_t)fx.N * fx.Hv * fx.Dv);
    read_block(f, fx.S_T_expected,  (size_t)fx.Hv * fx.Dv * fx.Dk);
    return fx;
}

float bf16_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f; std::memcpy(&f, &u, sizeof(f));
    return f;
}

void run(const std::string& fixture_path) {
    auto fx = load(fixture_path);
    auto& ctx = core::context();
    auto dev = ZEDINFER_DEVICE_NVIDIA;
    auto api = ctx.runtime().api();

    // Wrap inputs into device tensors via Tensor::create + memcpy H2D.
    auto to_dev_bf16 = [&](const std::vector<uint16_t>& host, std::initializer_list<size_t> shape) {
        auto t = Tensor::create(shape, ZEDINFER_DTYPE_BF16, dev, 0);
        api->memcpy_sync(t->data(), host.data(), host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
        return t;
    };
    auto to_dev_f32 = [&](const std::vector<float>& host, std::initializer_list<size_t> shape) {
        auto t = Tensor::create(shape, ZEDINFER_DTYPE_F32, dev, 0);
        api->memcpy_sync(t->data(), host.data(), host.size() * sizeof(float), ZEDINFER_MEMCPY_H2D);
        return t;
    };

    auto q  = to_dev_bf16(fx.q,  {(size_t)fx.N, (size_t)fx.Hk * fx.Dk});
    auto k  = to_dev_bf16(fx.k,  {(size_t)fx.N, (size_t)fx.Hk * fx.Dk});
    auto v  = to_dev_bf16(fx.v,  {(size_t)fx.N, (size_t)fx.Hv * fx.Dv});
    auto b  = to_dev_bf16(fx.b,  {(size_t)fx.N, (size_t)fx.Hv});
    auto a  = to_dev_bf16(fx.a,  {(size_t)fx.N, (size_t)fx.Hv});
    auto Al = to_dev_f32 (fx.A_log,   {(size_t)fx.Hv});
    auto dtb= to_dev_bf16(fx.dt_bias, {(size_t)fx.Hv});

    // Build a single-slot single-layer SSMStatePool view inline.
    auto state = to_dev_f32(fx.S0, {(size_t)fx.Hv, (size_t)fx.Dv, (size_t)fx.Dk});
    auto out   = Tensor::create({(size_t)fx.N, (size_t)fx.Hv * fx.Dv}, ZEDINFER_DTYPE_BF16, dev, 0);

    ops::mamba::GDNParams p;
    p.state_view.ssm_base       = state->data();
    p.state_view.ssm_stride_slot  = 0;
    p.state_view.ssm_stride_layer = 0;
    p.state_view.num_v_heads    = fx.Hv;
    p.state_view.value_head_dim = fx.Dv;
    p.state_view.d_state        = fx.Dk;
    p.slot_idx = 0; p.layer_idx = 0;
    p.q = q; p.k = k; p.v = v; p.b = b; p.a = a;
    p.A_log = Al; p.dt_bias = dtb; p.out = out;
    p.num_tokens = fx.N;
    ops::mamba::gdn(p);
    api->stream_sync();

    // Compare out
    std::vector<uint16_t> out_host((size_t)fx.N * fx.Hv * fx.Dv);
    api->memcpy_sync(out_host.data(), out->data(), out_host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    int bad = 0;
    for (size_t i = 0; i < out_host.size(); ++i) {
        float got = bf16_to_f32(out_host[i]);
        float exp = bf16_to_f32(fx.y_expected[i]);
        float tol = std::max(1e-2f, 1e-2f * std::abs(exp));
        if (std::abs(got - exp) > tol) ++bad;
    }
    EXPECT_LT(bad, (int)(out_host.size() / 200))  // < 0.5% outliers tolerable for bf16
        << "fixture=" << fixture_path << " mismatched=" << bad << "/" << out_host.size();

    // Compare S_T
    std::vector<float> S_host((size_t)fx.Hv * fx.Dv * fx.Dk);
    api->memcpy_sync(S_host.data(), state->data(), S_host.size() * sizeof(float), ZEDINFER_MEMCPY_D2H);
    int bad_s = 0;
    for (size_t i = 0; i < S_host.size(); ++i) {
        float tol = std::max(1e-4f, 1e-4f * std::abs(fx.S_T_expected[i]));
        if (std::abs(S_host[i] - fx.S_T_expected[i]) > tol) ++bad_s;
    }
    EXPECT_LT(bad_s, (int)(S_host.size() / 1000))
        << "fixture=" << fixture_path << " state mismatched=" << bad_s << "/" << S_host.size();
}

} // namespace

TEST(GDN, DecodeN1)      { run("tests/data/gdn_fixtures/decode_n1.bin"); }
TEST(GDN, PrefillN4)     { run("tests/data/gdn_fixtures/prefill_n4.bin"); }
TEST(GDN, PrefillN12)    { run("tests/data/gdn_fixtures/prefill_n12.bin"); }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

- [ ] **Step 2: Add test target to xmake/tests.lua**

Open `xmake/tests.lua` and append (matching the style of existing `test-mamba-ssu`):

```lua
target("test-gdn")
    set_kind("binary")
    add_files("tests/ops/test_gdn.cu")
    add_deps("zedinfer")
    add_packages("gtest")
    set_default(false)
    add_cugencodes("native")
```

Run:
```bash
xmake clean && xmake build -j1 test-gdn 2>&1 | tail -5
```
Expected: build ok, `test-gdn` binary produced.

- [ ] **Step 3: Run tests**

Run:
```bash
xmake run test-gdn
```
Expected: 3 tests, 3 PASS. If a fixture fails: dump per-element diffs by lowering tolerance temporarily, identify whether the bug is in the host launcher (param wiring) or the kernel math (probably the `decay` formula or the `Sk` reduction).

- [ ] **Step 4: Commit**

```bash
git add tests/ops/test_gdn.cu xmake/tests.lua
git commit -m "test(ops): GDN fixture-driven unit tests (decode_n1, prefill_n4, prefill_n12)"
```

---

## Task 9: Wire GDN into `forward_linear_attn_layer`

**Files:**
- Modify: `src/frontend/models/hybrid_transformer_forward.cpp`

Replace the SSU call with the GDN call. Add the silu(z) gating step that GDN expects outside the kernel.

- [ ] **Step 1: Locate the current SSU block**

Read `src/frontend/models/hybrid_transformer_forward.cpp` around lines 235–252 — the `ops::mamba::SSUParams sp; … ops::mamba::ssu(sp);` block in `forward_linear_attn_layer`.

- [ ] **Step 2: Replace SSU with GDN**

Replace the entire `ops::mamba::SSUParams sp; … ops::mamba::ssu(sp);` block with:

```cpp
    // 4. GDN: in-place update of SSMStatePool slot's state buffer + write out y.
    //    Per P3 retrospective: Qwen3.5's linear-attn is GatedDeltaNet, not
    //    Mamba2 SSM. The kernel applies sigmoid(b), softplus(a+dt_bias), and
    //    the delta-rule recurrence S = decay * S + beta * outer(v - S k, k).
    //    The silu(z) output gate is applied separately below (Step 7), not
    //    inside the kernel.
    auto y = make({N, Hv_Dv});
    ops::mamba::GDNParams gp;
    gp.state_view = m.ssm_pool->view();
    gp.slot_idx   = req.ssm_slot_idx();
    gp.layer_idx  = m.linear_layer_index(L);
    gp.q = q_ssm; gp.k = k_ssm; gp.v = v_ssm;
    gp.b = b;     gp.a = a;
    gp.A_log   = m.W(p + "A_log");
    gp.dt_bias = m.W(p + "dt_bias");
    gp.out = y;
    gp.num_tokens = static_cast<int>(N);
    ops::mamba::gdn(gp);
```

- [ ] **Step 3: Update the include**

Find the `#include "backend/ops/mamba/ssu.hpp"` line near the top of `hybrid_transformer_forward.cpp` and add **next to it** (do not remove the SSU include — `copy_strided_rows` still lives there):

```cpp
#include "backend/ops/mamba/gdn.hpp"
```

- [ ] **Step 4: Add the silu(z) output gate (between rms_norm and out_proj)**

Locate the block that does `ops::rms_norm(y_normed, …)` followed by `ops::linear(out, y_normed, …)`. **After `rms_norm` and before `linear`**, insert:

```cpp
    // GDN output gating: out = silu(z) * y_normed.  Performed externally because
    // the kernel only emits the delta-rule readout; the Qwen3.5 module applies
    // silu(z) * o_norm before out_proj.
    {
        auto y_gated = make({N, Hv_Dv});
        ops::silu_mul(y_gated, z, y_normed);
        y_normed = y_gated;
    }
```

If `ops::silu_mul` doesn't exist yet (likely — check `include/backend/ops/ops.hpp`), implement it as two calls instead:

```cpp
    // GDN output gating: out = silu(z) * y_normed.
    auto z_silu = make({N, Hv_Dv});
    ops::silu(z_silu, z);                  // existing op
    auto y_gated = make({N, Hv_Dv});
    ops::mul(y_gated, z_silu, y_normed);   // existing elementwise mul
    y_normed = y_gated;
```

If neither `ops::silu` nor `ops::mul` exists, fall back to one-off in-place math via existing ops or add a tiny kernel. (`grep -rn "ops::silu\|ops::mul " include/ src/` to confirm; if missing, the smallest fix is a `silu_mul` op in `src/backend/ops/silu_mul/nvidia/silu_mul.cu` — analogous to `attn_output_gate.cu`. Spec the op: `y[i] = (1 / (1 + exp(-z[i]))) * z[i] * x[i]` for the silu-and-multiply fused form.)

- [ ] **Step 5: Build**

Run:
```bash
xmake build -j1 2>&1 | tail -5
```
Expected: build ok.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/models/hybrid_transformer_forward.cpp \
        $(ls -1 src/backend/ops/silu_mul 2>/dev/null) \
        $(ls -1 include/backend/ops/silu_mul 2>/dev/null)
git commit -m "feat(qwen3.5): wire GDN kernel into forward_linear_attn_layer + silu(z) output gate"
```

---

## Task 10: End-to-end ping on Qwen3.5-35B-A3B

**Files:**
- (No source changes — runtime smoke + handoff doc update.)
- Modify: `docs/plan/qwen3_5_session_handoff.md` (M2 retrospective).

- [ ] **Step 1: Run the ping**

Run:
```bash
ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --nvidia --gpu-memory-utilization 0.5 2>&1 | tail -40
```
Expected: the `=== [Inference] Generated ===` block contains coherent English text (e.g. "I am an AI assistant created by Alibaba Cloud..." or similar — not `!!!`, not `mhey mhey mhey`).

- [ ] **Step 2: Run the Qwen3-30B-A3B regression**

Run:
```bash
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia --gpu-memory-utilization 0.5 2>&1 | tail -10
```
Expected: coherent English (no regression on the v0.2.0 model — this path doesn't touch the GDN kernel, but we want to confirm the build hasn't broken it).

- [ ] **Step 3: Run the new GDN test target**

Run:
```bash
xmake run test-gdn
```
Expected: 3/3 PASS.

- [ ] **Step 4: Update handoff doc**

Append to `docs/plan/qwen3_5_session_handoff.md`:

```markdown
## M2 — GatedDeltaNet kernel for Ampere/Ada (complete)

**Completed: <YYYY-MM-DD>.** Branch `feat/qwen3.5`.

### What landed

| Commit | Title |
|---|---|
| <hash> | docs(qwen3.5): document exact GDN recurrence from fla-org + HF reference |
| <hash> | test(ops): GDN fixture generator + 3 reference binaries |
| <hash> | feat(ops): add ops::mamba::gdn header |
| <hash> | feat(ops): GDN shared device math |
| <hash> | feat(ops): GDN decode kernel |
| <hash> | feat(ops): GDN prefill kernel |
| <hash> | feat(ops): GDN dispatch wrapper |
| <hash> | test(ops): GDN fixture-driven unit tests |
| <hash> | feat(qwen3.5): wire GDN kernel into forward_linear_attn_layer + silu(z) output gate |

### Components introduced
- `ops::mamba::gdn` — new op with decode (N=1) and prefill (sequential) kernels for sm_86/sm_89.
- `GDNParams` struct mirrors `SSUParams`; only the operand semantics differ.
- Shared device math in `gdn_kernel.cuh`: per-head scalar prep, warp-cooperative dot/update/readout.
- gtest target `test-gdn` against PyTorch fixtures (decode_n1 / prefill_n4 / prefill_n12) with bf16 tolerance.

### Verification

| Check | Result |
|---|---|
| `test-gdn` (3 fixtures: N=1, 4, 12) | PASS |
| 35B-A3B ping produces coherent English (verbatim sample) | PASS — sample: <paste 1-2 lines> |
| Qwen3-30B-A3B v0.2.0 regression | PASS |
| Full build clean | PASS |

### M2-follow-up entry conditions (now unblocked)
- Token byte-exact alignment (`qwen3_5_p3_m2_token_alignment.md`) can proceed.
- 27B dense ping (still blocked on host pinned-memory ulimit).
- Performance work (P7) for the kernel — current kernel is single-warp per V-head; SMEM/cluster optimization can come later.
```

- [ ] **Step 5: Commit handoff update**

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M2 retrospective — GDN kernel lands; 35B-A3B ping coherent"
```

---

## Task 11: Roadmap + cleanup

**Files:**
- Modify: `docs/roadmap.md` (mark M2 done).
- Modify: `docs/plan/qwen3_5_p3_m2_token_alignment.md` (add a header note pointing back here).

- [ ] **Step 1: Mark M2 done in roadmap**

Open `docs/roadmap.md`, find the Qwen3.5 section, mark M2 complete. Add: "M2 was rescoped during P3 debug — original token-alignment work is now M2-follow-up; the rescope (GDN kernel) is the actual M2 completion."

- [ ] **Step 2: Annotate the deferred plan**

At the top of `docs/plan/qwen3_5_p3_m2_token_alignment.md`, insert above any existing content:

```markdown
> **Deferred — see `qwen3_5_p3a_gdn_kernel.md`.** During P3 debug we discovered Qwen3.5's linear-attn is GatedDeltaNet, not Mamba2 SSM. A custom Ampere/Ada GDN kernel (P3a) had to land before token-by-token alignment work was meaningful. This plan resumes after P3a's verification step (Task 10 in P3a) confirms coherent generation.
```

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.md docs/plan/qwen3_5_p3_m2_token_alignment.md
git commit -m "docs(roadmap): M2 complete via GDN kernel (P3a); token alignment becomes follow-up"
```

---

## Self-Review Checklist (for plan author, NOT for the executor)

- **Spec coverage:**
  - ✅ Replace SSU primitive — Tasks 5–7 + 9.
  - ✅ Run on Ampere/Ada — kernel uses no Hopper-only features; sm_86 baseline.
  - ✅ Coherent generation — Task 10 ping check.
  - ✅ Test harness — Task 8.
  - ✅ Doc-first (CLAUDE.md rule #2) — this plan is the doc; existing P3 design doc updated in Task 11.

- **Placeholder scan:** GDN Math Reference section explicitly marked TBD to be filled in Task 1, Step 4 — but Task 1 is itself the placeholder-resolution work, so this is by design, not a plan defect. The fallback formula is concrete enough to write the kernel; Task 1 confirms or amends it before kernel coding begins.

- **Type consistency:**
  - `GDNParams` fields match across Tasks 3 (declared), 5 (decode launcher), 6 (prefill launcher), 7 (dispatch), 8 (test wiring), 9 (forward integration). All use `q, k, v, b, a, A_log, dt_bias, out`.
  - State shape `[Hv, Dv, Dk]` is consistent everywhere.
  - Kernel naming: `gdn_decode_kernel` / `gdn_decode_launch` / `gdn_prefill_kernel` / `gdn_prefill_launch` / `gdn` (entry). No drift.

- **Hardware:**
  - sm_86 baseline (A6000 dev box, 3090 target) — kernel uses only mma-free scalar/vector math + `__shfl_xor_sync` (warp shuffle, sm_30+).
  - sm_89 (4090) — sm_86 PTX runs forward-compatible via JIT; no specialization needed for v1.
  - 27B dense — same code path, blocked only by environmental ulimit; will validate on a larger-pinned-memory host when available.
