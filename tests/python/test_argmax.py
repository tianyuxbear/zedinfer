"""Correctness tests for the argmax operator."""

import numpy as np
import pytest
import torch

import zedinfer_ops
from conftest import get_tolerance


class TestArgmax:
    """Global argmax: returns (index, value) of the maximum element."""

    @pytest.mark.parametrize("size", [128, 1024, 4096])
    def test_f32(self, device, rng, size):
        vals = rng.standard_normal(size).astype(np.float32)

        idx_arr, val_arr = zedinfer_ops.argmax(vals, device=device)
        idx = int(idx_arr[0])
        val = float(val_arr[0])

        expected_idx = int(np.argmax(vals))
        expected_val = float(vals[expected_idx])

        assert idx == expected_idx, f"Index mismatch: {idx} vs {expected_idx}"
        assert abs(val - expected_val) < 1e-5, f"Value mismatch: {val} vs {expected_val}"

    def test_against_pytorch(self, device, rng):
        """Compare zedinfer argmax with torch.argmax."""
        vals_np = rng.standard_normal(2048).astype(np.float32)

        idx_arr, val_arr = zedinfer_ops.argmax(vals_np, device=device)
        zed_idx = int(idx_arr[0])

        torch_device = "cuda" if device == "cuda" else "cpu"
        vals_t = torch.from_numpy(vals_np).to(torch_device)
        torch_idx = int(torch.argmax(vals_t).cpu().item())

        assert zed_idx == torch_idx

    def test_single_element(self, device):
        vals = np.array([42.0], dtype=np.float32)
        idx_arr, val_arr = zedinfer_ops.argmax(vals, device=device)
        assert int(idx_arr[0]) == 0
        assert abs(float(val_arr[0]) - 42.0) < 1e-5

    def test_duplicate_max(self, device):
        """When multiple elements have the same max, the first index should be returned."""
        vals = np.array([1.0, 5.0, 3.0, 5.0, 2.0], dtype=np.float32)
        idx_arr, _ = zedinfer_ops.argmax(vals, device=device)
        assert int(idx_arr[0]) == 1  # first occurrence
