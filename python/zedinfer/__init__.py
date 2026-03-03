"""
ZedInfer: Python bindings for the ZedInfer inference engine.

Build:
    # Standard (Default: CPU)
    xmake f -m release --pytest=y && xmake build zedinfer_ops

    # NVIDIA GPU
    xmake f -m release --nv-gpu=y --pytest=y && xmake build zedinfer_ops

Install:
    uv pip install -e python/

Generate Stubs:
    pybind11-stubgen zedinfer.zedinfer_ops -o python/

Usage:
    from zedinfer import Tensor, Ops, has_cuda, device_synchronize
"""

from .zedinfer_ops import (
    Tensor,
    Ops,
    has_cuda,
    device_synchronize,
)

__all__ = [
    "Tensor",
    "Ops",
    "has_cuda",
    "device_synchronize",
]
