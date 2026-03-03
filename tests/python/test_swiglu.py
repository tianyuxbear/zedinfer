"""Correctness tests for the swiglu operator: out = up * silu(gate)."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


class TestSwiGLU:
    """SwiGLU activation: out = up * silu(gate), where silu(x) = x * sigmoid(x)."""

    @pytest.mark.parametrize("shape", [(128,), (32, 64), (4, 16, 32)])
    def test_f32(self, device, rng, shape):
        gate = rng.standard_normal(shape).astype(np.float32)
        up = rng.standard_normal(shape).astype(np.float32)

        result = zedinfer_ops.swiglu(gate, up, device=device)

        gate_t = torch.from_numpy(gate)
        up_t = torch.from_numpy(up)
        expected = (up_t * torch.nn.functional.silu(gate_t)).numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("shape", [(128,), (32, 64)])
    def test_f16(self, device, rng, shape):
        gate = rng.standard_normal(shape).astype(np.float16)
        up = rng.standard_normal(shape).astype(np.float16)

        result = zedinfer_ops.swiglu(gate, up, device=device)

        gate_t = torch.from_numpy(gate.astype(np.float32))
        up_t = torch.from_numpy(up.astype(np.float32))
        expected = (up_t * torch.nn.functional.silu(gate_t)).numpy().astype(np.float16)

        atol, rtol = get_tolerance(np.float16)
        np.testing.assert_allclose(
            result.astype(np.float32), expected.astype(np.float32),
            atol=atol, rtol=rtol,
        )

    def test_against_pytorch(self, device, rng):
        """Compare with PyTorch SiLU gating."""
        gate_np = rng.standard_normal((64, 256)).astype(np.float32)
        up_np = rng.standard_normal((64, 256)).astype(np.float32)

        zed_result = zedinfer_ops.swiglu(gate_np, up_np, device=device)

        torch_device = "cuda" if device == "cuda" else "cpu"
        gate_t = torch.from_numpy(gate_np).to(torch_device)
        up_t = torch.from_numpy(up_np).to(torch_device)
        torch_result = (up_t * torch.nn.functional.silu(gate_t)).cpu().numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(zed_result, torch_result, atol=atol, rtol=rtol)

    def test_zero_gate(self, device):
        """silu(0) = 0, so swiglu should output zeros when gate is zero."""
        gate = np.zeros((32,), dtype=np.float32)
        up = np.ones((32,), dtype=np.float32)
        result = zedinfer_ops.swiglu(gate, up, device=device)
        np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-7)

    def test_large_tensor(self, device, rng):
        gate = rng.standard_normal((1024, 1024)).astype(np.float32)
        up = rng.standard_normal((1024, 1024)).astype(np.float32)

        result = zedinfer_ops.swiglu(gate, up, device=device)

        gate_t = torch.from_numpy(gate)
        up_t = torch.from_numpy(up)
        expected = (up_t * torch.nn.functional.silu(gate_t)).numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)
