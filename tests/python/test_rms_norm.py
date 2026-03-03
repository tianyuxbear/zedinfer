"""Correctness tests for the rms_norm operator."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


def pytorch_rms_norm(x, weight, eps):
    """Reference RMS norm implementation using PyTorch."""
    x_t = torch.from_numpy(x)
    w_t = torch.from_numpy(weight)
    rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
    return (w_t * x_t / rms).numpy()


class TestRmsNorm:
    """RMS normalization: out = weight * input / rms(input)."""

    @pytest.mark.parametrize(
        "seq_len,hidden_size",
        [(1, 64), (4, 128), (16, 256)],
    )
    def test_f32(self, device, rng, seq_len, hidden_size):
        inp = rng.standard_normal((seq_len, hidden_size)).astype(np.float32)
        weight = rng.standard_normal(hidden_size).astype(np.float32)
        eps = 1e-6

        result = zedinfer_ops.rms_norm(inp, weight, eps=eps, device=device)
        expected = pytorch_rms_norm(inp, weight, eps)

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    def test_against_pytorch(self, device, rng):
        """Compare zedinfer rms_norm with a PyTorch implementation."""
        seq_len, hidden_size = 8, 512
        eps = 1e-5
        inp_np = rng.standard_normal((seq_len, hidden_size)).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        zed_result = zedinfer_ops.rms_norm(inp_np, weight_np, eps=eps, device=device)

        torch_device = "cuda" if device == "cuda" else "cpu"
        inp_t = torch.from_numpy(inp_np).to(torch_device)
        weight_t = torch.from_numpy(weight_np).to(torch_device)
        rms = torch.sqrt(torch.mean(inp_t ** 2, dim=-1, keepdim=True) + eps)
        torch_result = (weight_t * inp_t / rms).cpu().numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(zed_result, torch_result, atol=atol, rtol=rtol)

    def test_unit_weight(self, device, rng):
        """With weight=1, rms_norm should just normalize by RMS."""
        seq_len, hidden_size = 2, 64
        inp = rng.standard_normal((seq_len, hidden_size)).astype(np.float32)
        weight = np.ones(hidden_size, dtype=np.float32)
        eps = 1e-6

        result = zedinfer_ops.rms_norm(inp, weight, eps=eps, device=device)

        rms = np.sqrt(np.mean(inp ** 2, axis=-1, keepdims=True) + eps)
        expected = inp / rms

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)
