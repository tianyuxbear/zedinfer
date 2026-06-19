"""Generate golden inputs/outputs for ops::mrope_3d unit test.

This reference is byte-aligned with HuggingFace transformers'
`Qwen3_5MoeTextRotaryEmbedding.forward` (modeling_qwen3_5_moe.py:91-180) and
`apply_rotary_pos_emb` (lines 566-585). The key invariants:

  - cos/sin are built from `cat((freqs, freqs), dim=-1)` so they duplicate along
    the rotated axis: cos[..., i] == cos[..., i+half] for i in [0, half).
  - The rotation pairs (i, i+half) — NOT (2*i, 2*i+1). HF uses `rotate_half`,
    which is the Llama / NeoX convention, not the GPT-J interleaved-pair one.
  - `apply_interleaved_mrope` decides the t/h/w axis per index of freqs (the
    first half-dim slots): T owns slots 0,3,6,...; H owns 1,4,7,... up to
    section[1]*3; W owns 2,5,8,... up to section[2]*3.

Shapes / config (matches Qwen3.5 softmax-attention layer):
    head_dim         = 256
    partial_factor   = 0.25      -> Dh_rot = 64
    section          = [11, 11, 10] over half = 32 freq slots
    theta            = 1e7
    interleaved      = True      (HF apply_interleaved_mrope axis pattern)
"""

import os
import numpy as np
import torch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

N, H, Dh = 5, 2, 256
partial_factor = 0.25
theta = 1e7
section = [11, 11, 10]
Dh_rot = int(Dh * partial_factor)            # 64
half = Dh_rot // 2                            # 32
assert sum(section) == half, "section must sum to half"

torch.manual_seed(0)
x = torch.randn(N, H, Dh, dtype=torch.bfloat16)
pos_t = torch.arange(N, dtype=torch.int32)
pos_h = torch.arange(N, dtype=torch.int32) * 2
pos_w = torch.arange(N, dtype=torch.int32) * 3

# -- Build cos/sin exactly as Qwen3_5MoeTextRotaryEmbedding.forward does. --

# inv_freq[i] = 1 / theta^(2*i/Dh_rot) for i in [0, half).
inv_freq = 1.0 / (theta ** (torch.arange(0, Dh_rot, 2, dtype=torch.float32) / Dh_rot))

# pos stacked to (3, bs=1, N).
pos_thw = torch.stack([pos_t.long(), pos_h.long(), pos_w.long()], dim=0).unsqueeze(1)

# freqs of shape (3, bs=1, N, half).
inv_freq_exp = inv_freq[None, None, :, None].expand(3, 1, half, 1)   # (3, 1, half, 1)
pos_exp = pos_thw[:, :, None, :].float()                              # (3, 1, 1, N)
freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)                      # (3, 1, N, half)

# apply_interleaved_mrope: start with T; H overwrites slots [1, 4, 7, ...] up to
# section[1]*3 - 1; W overwrites [2, 5, 8, ...] up to section[2]*3 - 1.
freqs_t = freqs[0].clone()   # (1, N, half)
for dim, offset in enumerate((1, 2), start=1):
    length = section[dim] * 3
    idx = slice(offset, length, 3)
    freqs_t[..., idx] = freqs[dim, ..., idx]

# cos/sin duplicated to width Dh_rot.
emb = torch.cat((freqs_t, freqs_t), dim=-1)   # (1, N, Dh_rot)
cos_full = emb.cos()                          # (1, N, Dh_rot)
sin_full = emb.sin()

# -- Apply rotation: q_embed = q * cos + rotate_half(q) * sin (on the rotated dims). --

x_fp32 = x.clone().float()
q_rot = x_fp32[..., :Dh_rot]                  # (N, H, Dh_rot)


def rotate_half(t):
    t1 = t[..., : t.shape[-1] // 2]
    t2 = t[..., t.shape[-1] // 2 :]
    return torch.cat((-t2, t1), dim=-1)


# Broadcast cos/sin (1, N, Dh_rot) against q (N, H, Dh_rot): unsqueeze H axis.
cos_b = cos_full.squeeze(0).unsqueeze(1)      # (N, 1, Dh_rot)
sin_b = sin_full.squeeze(0).unsqueeze(1)      # (N, 1, Dh_rot)
q_out = q_rot * cos_b + rotate_half(q_rot) * sin_b

x_out = x_fp32.clone()
x_out[..., :Dh_rot] = q_out
x_out_bf16 = x_out.to(torch.bfloat16)

# Sanity: dims [Dh_rot, Dh) must be untouched at bf16 precision.
assert torch.equal(x_out_bf16[:, :, Dh_rot:], x[:, :, Dh_rot:]), \
    "pass-through dims diverged in reference"


def bf16_bytes(t):
    """Serialize a bf16 tensor as raw 2-byte little-endian elements."""
    return t.contiguous().view(torch.int16).numpy().astype(np.int16).tobytes()


with open(os.path.join(OUT_DIR, "x_in.bin"), "wb") as f:
    f.write(bf16_bytes(x))
with open(os.path.join(OUT_DIR, "x_out_ref.bin"), "wb") as f:
    f.write(bf16_bytes(x_out_bf16))
pos_thw_np = np.stack([pos_t.numpy(), pos_h.numpy(), pos_w.numpy()], axis=0).astype(np.int32)
with open(os.path.join(OUT_DIR, "pos_thw.bin"), "wb") as f:
    f.write(pos_thw_np.tobytes())

print("[ok] mrope_3d reference (HF rotate_half) generated:", tuple(x_out.shape), "->", OUT_DIR)
