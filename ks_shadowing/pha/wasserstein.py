"""Hera library bindings for Wasserstein distance computation."""

from ctypes import CDLL, POINTER, c_double, c_int
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_lib: CDLL | None = None


def _get_lib() -> CDLL:
    """Return the cached library singleton."""
    global _lib  # noqa: PLW0603
    if _lib is None:
        so_path = Path(__file__).parent / "libhera2py.so"
        _lib = CDLL(str(so_path))

        _lib.wasserstein_dist_c.argtypes = [
            POINTER(c_double),  # dgm_a
            c_int,  # n_a
            POINTER(c_double),  # dgm_b
            c_int,  # n_b
            c_double,  # delta
        ]
        _lib.wasserstein_dist_c.restype = c_double

    return _lib


def wasserstein_distance(
    dgm_a: NDArray[np.float64],
    dgm_b: NDArray[np.float64],
    delta: float = 0.01,
) -> float:
    """Compute Wasserstein-2 distance between two persistence diagrams.

    Uses Hera library for (1+delta)-approximate computation.

    Args:
        dgm_a: First diagram, shape (n_a, 2) with (birth, death) pairs.
        dgm_b: Second diagram, shape (n_b, 2) with (birth, death) pairs.
        delta: Relative error tolerance (default 0.01 = 1%).

    Returns:
        The Wasserstein-2 distance.
    """
    lib = _get_lib()

    n_a = dgm_a.shape[0] if dgm_a.ndim == 2 else 0  # noqa: PLR2004
    n_b = dgm_b.shape[0] if dgm_b.ndim == 2 else 0  # noqa: PLR2004

    # Handle empty diagrams
    if n_a == 0 and n_b == 0:
        return 0.0

    # Only copy if not already contiguous float64
    if n_a > 0:
        if dgm_a.dtype != np.float64 or not dgm_a.flags["C_CONTIGUOUS"]:
            dgm_a = np.ascontiguousarray(dgm_a, dtype=np.float64)
        a_ptr = dgm_a.ctypes.data_as(POINTER(c_double))
    else:
        a_ptr = None

    if n_b > 0:
        if dgm_b.dtype != np.float64 or not dgm_b.flags["C_CONTIGUOUS"]:
            dgm_b = np.ascontiguousarray(dgm_b, dtype=np.float64)
        b_ptr = dgm_b.ctypes.data_as(POINTER(c_double))
    else:
        b_ptr = None

    return lib.wasserstein_dist_c(a_ptr, n_a, b_ptr, n_b, delta)
