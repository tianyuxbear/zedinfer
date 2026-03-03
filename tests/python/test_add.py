"""Correctness tests for the add operator: c = a + b."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


class TestAdd:
    """Element-wise addition: c = a + b."""

    @pytest.mark.parametrize("shape", [(128,), (32, 64), (4, 16, 32)])
    def test_f32(self, device, rng, shape):
        a = rng.standard_normal(shape).astype(np.float32)
        b = rng.standard_normal(shape).astype(np.float32)

        result = zedinfer_ops.add(a, b, device=device)
        expected = a + b

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("shape", [(128,), (32, 64)])
    def test_f16(self, device, rng, shape):
        a = rng.standard_normal(shape).astype(np.float16)
        b = rng.standard_normal(shape).astype(np.float16)

        result = zedinfer_ops.add(a, b, device=device)
        expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)

        atol, rtol = get_tolerance(np.float16)
        np.testing.assert_allclose(
            result.astype(np.float32), expected.astype(np.float32),
            atol=atol, rtol=rtol,
        )

    def test_against_pytorch(self, device, rng):
        """Compare zedinfer add with torch.add."""
        a_np = rng.standard_normal((64, 128)).astype(np.float32)
        b_np = rng.standard_normal((64, 128)).astype(np.float32)

        zed_result = zedinfer_ops.add(a_np, b_np, device=device)

        torch_device = "cuda" if device == "cuda" else "cpu"
        a_t = torch.from_numpy(a_np).to(torch_device)
        b_t = torch.from_numpy(b_np).to(torch_device)
        torch_result = (a_t + b_t).cpu().numpy()

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(zed_result, torch_result, atol=atol, rtol=rtol)

    def test_large_tensor(self, device, rng):
        """Test with a large tensor to stress-test the implementation."""
        a = rng.standard_normal((1024, 1024)).astype(np.float32)
        b = rng.standard_normal((1024, 1024)).astype(np.float32)

        result = zedinfer_ops.add(a, b, device=device)
        expected = a + b

        atol, rtol = get_tolerance(np.float32)
        np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol)
