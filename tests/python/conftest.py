"""
Shared pytest fixtures and configuration for zedinfer operator tests.

Usage:
    1. Build the pybind11 module:
       xmake f -m release --pytest=y [--nv-gpu=y]
       xmake build zedinfer_ops

    2. Run tests:
       cd tests/python
       pytest -v                       # all correctness tests
       pytest test_add.py -v           # single operator
       python bench_ops.py             # performance benchmarks
"""

import sys
import os
import pytest
import numpy as np

# Ensure the built .so can be imported from tests/python/
sys.path.insert(0, os.path.dirname(__file__))

import zedinfer_ops  # noqa: E402


# ---------------------------------------------------------------------------
# Device fixtures
# ---------------------------------------------------------------------------

def get_devices():
    """Return list of available devices for parameterized tests."""
    devices = ["cpu"]
    if zedinfer_ops.has_cuda():
        devices.append("cuda")
    return devices


@pytest.fixture(params=get_devices())
def device(request):
    """Parameterized fixture that yields each available device."""
    return request.param


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------

# Default tolerances per dtype for correctness checks
TOLERANCES = {
    np.float32: {"atol": 1e-5, "rtol": 1e-5},
    np.float16: {"atol": 1e-2, "rtol": 1e-2},
}


def get_tolerance(dtype):
    """Return (atol, rtol) for a given numpy dtype."""
    tol = TOLERANCES.get(dtype, {"atol": 1e-5, "rtol": 1e-5})
    return tol["atol"], tol["rtol"]


# ---------------------------------------------------------------------------
# Random data generation helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)
