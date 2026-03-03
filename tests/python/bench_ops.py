#!/usr/bin/env python3
"""
Performance benchmarks comparing zedinfer operators against PyTorch.

Usage:
    python bench_ops.py                      # benchmark all ops on CPU
    python bench_ops.py --device cuda         # benchmark on CUDA
    python bench_ops.py --op add swiglu       # benchmark specific ops
    python bench_ops.py --warmup 10 --iters 100

Output: table of (operator, size, zedinfer_ms, pytorch_ms, speedup).
"""

import argparse
import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import zedinfer_ops  # noqa: E402


# ============================================================================
# Timing utilities
# ============================================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, device="cpu"):
        self.device = device
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            zedinfer_ops.device_synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device == "cuda":
            torch.cuda.synchronize()
            zedinfer_ops.device_synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def bench(fn, warmup, iters, device="cpu"):
    """Run fn() for warmup iterations, then measure iters iterations."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        with Timer(device) as t:
            fn()
        times.append(t.elapsed_ms)

    return np.median(times)


# ============================================================================
# Benchmark definitions
# Each benchmark returns (zedinfer_ms, pytorch_ms) for a given size config.
# ============================================================================

def bench_add(size, device, warmup, iters):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(size).astype(np.float32)
    b = rng.standard_normal(size).astype(np.float32)

    torch_dev = device
    a_t = torch.from_numpy(a).to(torch_dev)
    b_t = torch.from_numpy(b).to(torch_dev)

    zed_ms = bench(lambda: zedinfer_ops.add(a, b, device=device), warmup, iters, device)
    pt_ms = bench(lambda: torch.add(a_t, b_t), warmup, iters, device)
    return zed_ms, pt_ms


def bench_swiglu(size, device, warmup, iters):
    rng = np.random.default_rng(0)
    gate = rng.standard_normal(size).astype(np.float32)
    up = rng.standard_normal(size).astype(np.float32)

    torch_dev = device
    gate_t = torch.from_numpy(gate).to(torch_dev)
    up_t = torch.from_numpy(up).to(torch_dev)

    zed_ms = bench(lambda: zedinfer_ops.swiglu(gate, up, device=device), warmup, iters, device)
    pt_ms = bench(lambda: up_t * torch.nn.functional.silu(gate_t), warmup, iters, device)
    return zed_ms, pt_ms


def bench_rms_norm(size, device, warmup, iters):
    seq_len, hidden = size
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((seq_len, hidden)).astype(np.float32)
    weight = rng.standard_normal(hidden).astype(np.float32)
    eps = 1e-6

    torch_dev = device
    inp_t = torch.from_numpy(inp).to(torch_dev)
    weight_t = torch.from_numpy(weight).to(torch_dev)

    def pt_rms_norm():
        rms = torch.sqrt(torch.mean(inp_t ** 2, dim=-1, keepdim=True) + eps)
        return weight_t * inp_t / rms

    zed_ms = bench(lambda: zedinfer_ops.rms_norm(inp, weight, eps=eps, device=device),
                   warmup, iters, device)
    pt_ms = bench(pt_rms_norm, warmup, iters, device)
    return zed_ms, pt_ms


def bench_linear(size, device, warmup, iters):
    M, N, K = size
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((M, K)).astype(np.float32)
    weight = rng.standard_normal((N, K)).astype(np.float32)

    torch_dev = device
    inp_t = torch.from_numpy(inp).to(torch_dev)
    weight_t = torch.from_numpy(weight).to(torch_dev)

    zed_ms = bench(lambda: zedinfer_ops.linear(inp, weight, device=device),
                   warmup, iters, device)
    pt_ms = bench(lambda: torch.nn.functional.linear(inp_t, weight_t),
                  warmup, iters, device)
    return zed_ms, pt_ms


def bench_rope(size, device, warmup, iters):
    seq_len, num_heads, head_dim = size
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
    pos_ids = np.arange(seq_len, dtype=np.int64)
    theta = 10000.0

    zed_ms = bench(lambda: zedinfer_ops.rope(inp, pos_ids, theta=theta, device=device),
                   warmup, iters, device)

    # PyTorch reference (manual RoPE)
    inp_t = torch.from_numpy(inp).to(device)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32).to(device) / head_dim))
    pos_t = torch.from_numpy(pos_ids).float().to(device)

    def pt_rope():
        angles = pos_t.unsqueeze(1) * freqs.unsqueeze(0)
        cos_v = torch.cos(angles).unsqueeze(1)
        sin_v = torch.sin(angles).unsqueeze(1)
        x1 = inp_t[..., :head_dim // 2]
        x2 = inp_t[..., head_dim // 2:]
        return torch.cat([x1 * cos_v - x2 * sin_v, x2 * cos_v + x1 * sin_v], dim=-1)

    pt_ms = bench(pt_rope, warmup, iters, device)
    return zed_ms, pt_ms


def bench_embedding(size, device, warmup, iters):
    seq_len, vocab_size, hidden = size
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((vocab_size, hidden)).astype(np.float32)
    index = rng.integers(0, vocab_size, size=seq_len).astype(np.int32)

    torch_dev = device
    weight_t = torch.from_numpy(weight).to(torch_dev)
    index_t = torch.from_numpy(index.astype(np.int64)).to(torch_dev)

    zed_ms = bench(lambda: zedinfer_ops.embedding(index, weight, device=device),
                   warmup, iters, device)
    pt_ms = bench(lambda: torch.nn.functional.embedding(index_t, weight_t),
                  warmup, iters, device)
    return zed_ms, pt_ms


def bench_self_attention(size, device, warmup, iters):
    seq_len, total_len, num_heads, num_kv_heads, head_dim = size
    rng = np.random.default_rng(0)
    q = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((total_len, num_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((total_len, num_kv_heads, head_dim)).astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)

    zed_ms = bench(lambda: zedinfer_ops.self_attention(q, k, v, scale=scale, device=device),
                   warmup, iters, device)

    # PyTorch reference
    torch_dev = device
    q_t = torch.from_numpy(q).to(torch_dev)
    k_t = torch.from_numpy(k).to(torch_dev)
    v_t = torch.from_numpy(v).to(torch_dev)
    group_size = num_heads // num_kv_heads
    prefix_len = total_len - seq_len

    def pt_self_attention():
        qt = q_t.permute(1, 0, 2)
        kt = k_t.permute(1, 0, 2).repeat_interleave(group_size, dim=0)
        vt = v_t.permute(1, 0, 2).repeat_interleave(group_size, dim=0)
        scores = torch.bmm(qt, kt.transpose(1, 2)) * scale
        mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=torch_dev),
                          diagonal=prefix_len + 1)
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, vt).permute(1, 0, 2)

    pt_ms = bench(pt_self_attention, warmup, iters, device)
    return zed_ms, pt_ms


# ============================================================================
# Benchmark registry
# ============================================================================

BENCHMARKS = {
    "add": {
        "fn": bench_add,
        "sizes": {
            "small":  (1024,),
            "medium": (256 * 1024,),
            "large":  (4 * 1024 * 1024,),
        },
    },
    "swiglu": {
        "fn": bench_swiglu,
        "sizes": {
            "small":  (1024,),
            "medium": (256 * 1024,),
            "large":  (4 * 1024 * 1024,),
        },
    },
    "rms_norm": {
        "fn": bench_rms_norm,
        "sizes": {
            "small":  (1, 256),
            "medium": (16, 2048),
            "large":  (64, 4096),
        },
    },
    "linear": {
        "fn": bench_linear,
        "sizes": {
            "small":  (1, 128, 64),
            "medium": (16, 1024, 512),
            "large":  (64, 4096, 2048),
        },
    },
    "rope": {
        "fn": bench_rope,
        "sizes": {
            "small":  (1, 8, 64),
            "medium": (16, 32, 128),
            "large":  (64, 32, 128),
        },
    },
    "embedding": {
        "fn": bench_embedding,
        "sizes": {
            "small":  (4, 1000, 128),
            "medium": (32, 10000, 512),
            "large":  (128, 50000, 1024),
        },
    },
    "self_attention": {
        "fn": bench_self_attention,
        "sizes": {
            # (seq_len, total_len, num_heads, num_kv_heads, head_dim)
            "small":  (1, 16, 8, 8, 64),
            "medium": (4, 64, 32, 8, 128),
            "large":  (16, 256, 32, 8, 128),
        },
    },
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark zedinfer ops vs PyTorch")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to benchmark on")
    parser.add_argument("--op", nargs="*", default=None,
                        help="Specific operators to benchmark (default: all)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of benchmark iterations")
    parser.add_argument("--size", default=None, choices=["small", "medium", "large"],
                        help="Only run a specific size (default: all)")
    args = parser.parse_args()

    if args.device == "cuda":
        if not zedinfer_ops.has_cuda():
            print("ERROR: zedinfer was not compiled with CUDA support.")
            print("Rebuild with: xmake f -m release --nv-gpu=y --pytest=y && xmake build zedinfer_ops")
            sys.exit(1)
        if not torch.cuda.is_available():
            print("ERROR: PyTorch CUDA is not available.")
            sys.exit(1)

    ops = args.op if args.op else list(BENCHMARKS.keys())

    # Print header
    print(f"\nDevice: {args.device} | Warmup: {args.warmup} | Iterations: {args.iters}")
    print("=" * 85)
    print(f"{'Operator':<20} {'Size':<10} {'ZedInfer (ms)':>14} {'PyTorch (ms)':>14} {'Speedup':>10}")
    print("-" * 85)

    for op_name in ops:
        if op_name not in BENCHMARKS:
            print(f"WARNING: Unknown operator '{op_name}', skipping.")
            continue

        bm = BENCHMARKS[op_name]
        sizes = bm["sizes"]
        if args.size:
            sizes = {args.size: sizes[args.size]}

        for size_name, size_config in sizes.items():
            try:
                zed_ms, pt_ms = bm["fn"](size_config, args.device, args.warmup, args.iters)
                speedup = pt_ms / zed_ms if zed_ms > 0 else float("inf")
                marker = "<--" if speedup > 1.0 else ""
                print(f"{op_name:<20} {size_name:<10} {zed_ms:>13.3f} {pt_ms:>13.3f} {speedup:>9.2f}x {marker}")
            except Exception as e:
                print(f"{op_name:<20} {size_name:<10} {'ERROR':>14} {str(e)}")

    print("=" * 85)
    print("Speedup > 1.0x means zedinfer is faster than PyTorch.")


if __name__ == "__main__":
    main()
