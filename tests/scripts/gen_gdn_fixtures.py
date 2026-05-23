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
        # Per-V-head update, matches HF torch_recurrent_gated_delta_rule and
        # fla naive_recurrent_gated_delta_rule: state is decayed BEFORE the
        # delta is computed, so delta = v - decay * (S_old @ k).
        # S: [Hv, Dv, Dk]
        S     = decay.view(Hv, 1, 1) * S                   # decay first
        Sk    = torch.einsum('vdk,vk->vd', S, k_v)         # [Hv, Dv] on decayed S
        dv    = v[t].float() - Sk                          # [Hv, Dv]
        S     = S + beta.view(Hv, 1, 1) * torch.einsum('vd,vk->vdk', dv, k_v)
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
