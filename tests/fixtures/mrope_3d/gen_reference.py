"""Generate golden inputs/outputs for ops::mrope_3d unit test.

We implement the Qwen3.5 3D mrope formula directly in pure Python/Torch so
the test fixture is hermetic and does not depend on whether HuggingFace
transformers has shipped a Qwen3.5 implementation yet.

Shapes / config (matches Qwen3.5 softmax-attention layer):
    head_dim         = 256
    partial_factor   = 0.25      -> Dh_rot = 64
    section          = [11, 11, 10] over half = 32 dim-pairs
    theta            = 1e7
    interleaved      = True      (adjacent dim pairs)
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

# Work in fp32 for the reference math, then truncate back to bf16 at the end
# so the golden output matches what a bf16 kernel will store.
x_out = x.clone().float()
for n in range(N):
    for hd in range(H):
        for pi in range(half):
            if pi < section[0]:
                pos = pos_t[n].item()
            elif pi < section[0] + section[1]:
                pos = pos_h[n].item()
            else:
                pos = pos_w[n].item()
            freq = theta ** (-(2 * pi) / (2 * half))
            angle = pos * freq
            c, s = np.cos(angle), np.sin(angle)
            a = x_out[n, hd, 2 * pi].item()
            b = x_out[n, hd, 2 * pi + 1].item()
            x_out[n, hd, 2 * pi]     = a * c - b * s
            x_out[n, hd, 2 * pi + 1] = a * s + b * c

x_out_bf16 = x_out.to(torch.bfloat16)

# Sanity: dims [Dh_rot, Dh) must be untouched at bf16 precision.
assert torch.equal(x_out_bf16[:, :, Dh_rot:], x[:, :, Dh_rot:]), \
    "pass-through dims diverged in reference"


def bf16_bytes(t):
    """Serialize a bf16 tensor as raw 2-byte little-endian elements."""
    # bf16 -> int16 view -> bytes; preserves the exact bit pattern.
    return t.contiguous().view(torch.int16).numpy().astype(np.int16).tobytes()


with open(os.path.join(OUT_DIR, "x_in.bin"), "wb") as f:
    f.write(bf16_bytes(x))
with open(os.path.join(OUT_DIR, "x_out_ref.bin"), "wb") as f:
    f.write(bf16_bytes(x_out_bf16))
pos_thw = np.stack([pos_t.numpy(), pos_h.numpy(), pos_w.numpy()], axis=0).astype(np.int32)
with open(os.path.join(OUT_DIR, "pos_thw.bin"), "wb") as f:
    f.write(pos_thw.tobytes())

print("[ok] mrope_3d reference generated:", tuple(x_out.shape), "->", OUT_DIR)
