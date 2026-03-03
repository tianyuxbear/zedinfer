"""Correctness tests for the embedding operator."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


class TestEmbedding:
    """Embedding lookup: out[i] = weight[index[i]]."""

    @pytest.mark.parametrize(
        "seq_len,vocab_size,hidden_size",
        [(4, 100, 64), (16, 1000, 128), (1, 50, 32)],
    )
    def test_f32(self, device, rng, seq_len, vocab_size, hidden_size):
        weight = rng.standard_normal((vocab_size, hidden_size)).astype(np.float32)
        index = rng.integers(0, vocab_size, size=seq_len).astype(np.int32)

        result = zedinfer_ops.embedding(index, weight, device=device)
        expected = weight[index]

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    def test_against_pytorch(self, device, rng):
        """Compare zedinfer embedding with torch.nn.functional.embedding."""
        vocab_size, hidden_size, seq_len = 500, 128, 32
        weight_np = rng.standard_normal((vocab_size, hidden_size)).astype(np.float32)
        index_np = rng.integers(0, vocab_size, size=seq_len).astype(np.int32)

        zed_result = zedinfer_ops.embedding(index_np, weight_np, device=device)

        torch_device = "cuda" if device == "cuda" else "cpu"
        weight_t = torch.from_numpy(weight_np).to(torch_device)
        index_t = torch.from_numpy(index_np.astype(np.int64)).to(torch_device)
        torch_result = torch.nn.functional.embedding(index_t, weight_t).cpu().numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(zed_result, torch_result, atol=atol, rtol=rtol)

    def test_single_token(self, device, rng):
        weight = rng.standard_normal((10, 8)).astype(np.float32)
        index = np.array([3], dtype=np.int32)
        result = zedinfer_ops.embedding(index, weight, device=device)
        np.testing.assert_allclose(result[0], weight[3], atol=1e-6)
