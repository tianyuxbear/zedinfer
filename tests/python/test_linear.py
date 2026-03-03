"""Correctness tests for the linear operator: out = input @ weight^T + bias."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


class TestLinear:
    """Linear transform: out[M,N] = in[M,K] @ weight[N,K]^T + bias[N]."""

    @pytest.mark.parametrize(
        "M,N,K",
        [(1, 64, 32), (4, 128, 64), (16, 256, 128)],
    )
    def test_f32_no_bias(self, device, rng, M, N, K):
        inp = rng.standard_normal((M, K)).astype(np.float32)
        weight = rng.standard_normal((N, K)).astype(np.float32)

        result = zedinfer_ops.linear(inp, weight, bias=None, device=device)
        expected = inp @ weight.T

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "M,N,K",
        [(1, 64, 32), (4, 128, 64)],
    )
    def test_f32_with_bias(self, device, rng, M, N, K):
        inp = rng.standard_normal((M, K)).astype(np.float32)
        weight = rng.standard_normal((N, K)).astype(np.float32)
        bias = rng.standard_normal(N).astype(np.float32)

        result = zedinfer_ops.linear(inp, weight, bias=bias, device=device)
        expected = inp @ weight.T + bias

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    def test_against_pytorch(self, device, rng):
        """Compare zedinfer linear with torch.nn.functional.linear."""
        M, N, K = 8, 128, 64
        inp_np = rng.standard_normal((M, K)).astype(np.float32)
        weight_np = rng.standard_normal((N, K)).astype(np.float32)
        bias_np = rng.standard_normal(N).astype(np.float32)

        zed_result = zedinfer_ops.linear(inp_np, weight_np, bias=bias_np, device=device)

        torch_device = "cuda" if device == "cuda" else "cpu"
        inp_t = torch.from_numpy(inp_np).to(torch_device)
        weight_t = torch.from_numpy(weight_np).to(torch_device)
        bias_t = torch.from_numpy(bias_np).to(torch_device)
        torch_result = torch.nn.functional.linear(inp_t, weight_t, bias_t).cpu().numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(zed_result, torch_result, atol=atol, rtol=rtol)

    def test_vecmul_path(self, device, rng):
        """M=1 triggers the vecmul (GEMV) fast path."""
        K, N = 256, 128
        inp = rng.standard_normal((1, K)).astype(np.float32)
        weight = rng.standard_normal((N, K)).astype(np.float32)

        result = zedinfer_ops.linear(inp, weight, bias=None, device=device)
        expected = inp @ weight.T

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)
