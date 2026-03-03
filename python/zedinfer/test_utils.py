"""
Shared test utilities for zedinfer operator testing.

Provides:
    random_tensor(shape, dtype_name, device_name) -> (torch.Tensor, zedinfer.Tensor)
    check_equal(zedinfer_t, torch_t, atol, rtol)  -> bool
    benchmark(torch_fn, zedinfer_fn, device_name)  -> prints timing

All data exchange uses Tensor.from_torch() / Tensor.to_torch() — no numpy
dependency, so bfloat16 works natively.
"""

import time

import torch

import zedinfer

# dtype name -> torch dtype
DTYPE_MAP = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
}

# zedinfer device name -> torch device name
TORCH_DEVICE = {
    "cpu": "cpu",
    "nvidia": "cuda",
}


def random_tensor(shape, dtype_name="f32", device_name="cpu", scale=None):
    """Create (torch_tensor, zedinfer_tensor) with identical random data.

    Uses from_torch() internally — supports all dtypes including bf16.
    """
    torch_dtype = DTYPE_MAP[dtype_name]
    torch_device = TORCH_DEVICE[device_name]

    if torch_dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        t = torch.randint(-100, 100, shape, dtype=torch_dtype, device=torch_device)
    else:
        t = torch.randn(shape, dtype=torch.float32).to(torch_dtype).to(torch_device)

    if scale is not None:
        t *= scale

    t_ = zedinfer.Tensor.from_torch(t)
    return t, t_


def zero_tensor(shape, dtype_name="f32", device_name="cpu"):
    """Create (torch_tensor, zedinfer_tensor) pair filled with zeros."""
    torch_dtype = DTYPE_MAP[dtype_name]
    torch_device = TORCH_DEVICE[device_name]
    t = torch.zeros(shape, dtype=torch_dtype, device=torch_device)
    t_ = zedinfer.Tensor.from_torch(t)
    return t, t_


def random_int_tensor(shape, device_name="cpu", high=100):
    """Create (torch_tensor, zedinfer_tensor) with identical random int32 data."""
    torch_device = TORCH_DEVICE[device_name]
    t = torch.randint(0, high, shape, dtype=torch.int32, device=torch_device)
    t_ = zedinfer.Tensor.from_torch(t)
    return t, t_


def arrange_tensor(start, end, device_name="cpu"):
    """Create (torch_tensor, zedinfer_tensor) with identical arange data."""
    torch_device = TORCH_DEVICE[device_name]
    t = torch.arange(start, end, device=torch_device)
    t_ = zedinfer.Tensor.from_torch(t)
    return t, t_


def check_equal(zedinfer_t, torch_t, atol=1e-5, rtol=1e-5, strict=False):
    """Compare a zedinfer tensor against a torch tensor. Returns True if close.

    Uses to_torch() internally — supports all dtypes including bf16.
    If strict=True, requires exact equality (no tolerance).
    """
    result = zedinfer_t.to_torch().cpu()
    expected = torch_t.cpu()
    if strict:
        return torch.equal(result, expected)
    return torch.allclose(result.float(), expected.float(), atol=atol, rtol=rtol)


def benchmark(torch_fn, zedinfer_fn, device_name, warmup=10, iters=100):
    """Benchmark torch vs zedinfer, print timing comparison."""
    for _ in range(warmup):
        torch_fn()
        zedinfer_fn()

    if device_name == "nvidia":
        torch.cuda.synchronize()
        zedinfer.device_synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch_fn()
    if device_name == "nvidia":
        torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) / iters * 1000

    start = time.perf_counter()
    for _ in range(iters):
        zedinfer_fn()
    if device_name == "nvidia":
        zedinfer.device_synchronize()
    zed_ms = (time.perf_counter() - start) / iters * 1000

    speedup = torch_ms / zed_ms if zed_ms > 0 else float("inf")
    print(
        f"    torch: {torch_ms:.4f} ms | zedinfer: {zed_ms:.4f} ms | speedup: {speedup:.2f}x"
    )
