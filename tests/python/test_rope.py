"""Correctness tests for the RoPE (Rotary Position Embedding) operator."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


def pytorch_rope(x_np, pos_ids_np, theta):
    """
    Reference RoPE implementation.

    x: [seq_len, num_heads, head_dim]  (float32)
    pos_ids: [seq_len]                 (int64)
    theta: float

    For each position i with pos_id p:
      inv_theta[k] = 1 / theta^(2k / head_dim)
      angle[k] = p * inv_theta[k]
      out[i, h, k]             = x[i,h,k] * cos(angle[k]) - x[i,h,k+half] * sin(angle[k])
      out[i, h, k+half]        = x[i,h,k+half] * cos(angle[k]) + x[i,h,k] * sin(angle[k])
    """
    x = torch.from_numpy(x_np).float()
    seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2

    # Compute inv_theta: 1 / theta^(2k/head_dim)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    # Angles: [seq_len, half_dim]
    pos = torch.from_numpy(pos_ids_np).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)

    cos_vals = torch.cos(angles).unsqueeze(1)  # [seq_len, 1, half_dim]
    sin_vals = torch.sin(angles).unsqueeze(1)  # [seq_len, 1, half_dim]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x2 * cos_vals + x1 * sin_vals

    return torch.cat([out1, out2], dim=-1).numpy()


class TestRoPE:
    """Rotary Position Embedding operator."""

    @pytest.mark.parametrize(
        "seq_len,num_heads,head_dim",
        [(1, 4, 64), (4, 8, 128), (16, 2, 32)],
    )
    def test_f32(self, device, rng, seq_len, num_heads, head_dim):
        inp = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
        pos_ids = np.arange(seq_len, dtype=np.int64)
        theta = 10000.0

        result = zedinfer_ops.rope(inp, pos_ids, theta=theta, device=device)
        expected = pytorch_rope(inp, pos_ids, theta)

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    def test_non_sequential_positions(self, device, rng):
        """Test with non-sequential position IDs (e.g., continuation of a sequence)."""
        seq_len, num_heads, head_dim = 4, 4, 64
        inp = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
        pos_ids = np.array([10, 11, 12, 13], dtype=np.int64)
        theta = 10000.0

        result = zedinfer_ops.rope(inp, pos_ids, theta=theta, device=device)
        expected = pytorch_rope(inp, pos_ids, theta)

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    def test_different_theta(self, device, rng):
        """Test with different theta values."""
        seq_len, num_heads, head_dim = 4, 4, 64
        inp = rng.standard_normal((seq_len, num_heads, head_dim)).astype(np.float32)
        pos_ids = np.arange(seq_len, dtype=np.int64)

        for theta in [500.0, 10000.0, 100000.0]:
            result = zedinfer_ops.rope(inp, pos_ids, theta=theta, device=device)
            expected = pytorch_rope(inp, pos_ids, theta)

            atol, rtol = get_tolerance(np.float32)
            np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol,
                                       err_msg=f"Failed with theta={theta}")
