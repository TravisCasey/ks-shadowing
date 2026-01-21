"""Hera library bindings for Wasserstein distance computation."""

from ctypes import CDLL, POINTER, c_double, c_int, c_int64
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

        _lib.wasserstein_matrix_c.argtypes = [
            POINTER(c_double),  # traj_points
            POINTER(c_int64),  # traj_offsets
            c_int64,  # num_traj
            POINTER(c_double),  # rpo_points
            POINTER(c_int64),  # rpo_offsets
            c_int64,  # num_rpo
            c_double,  # delta
            POINTER(c_double),  # out
        ]
        _lib.wasserstein_matrix_c.restype = None

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


def _flatten_diagrams(
    diagrams: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Flatten diagrams into (points, offsets) arrays for the batch C API."""
    if not diagrams:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64)

    # Filter to valid 2D arrays and compute lengths
    valid = [dgm for dgm in diagrams if dgm.ndim == 2]  # noqa: PLR2004
    lengths = np.array([dgm.shape[0] for dgm in valid], dtype=np.int64)

    # Build offsets via cumsum
    offsets = np.zeros(len(diagrams) + 1, dtype=np.int64)
    valid_index = 0
    for i, dgm in enumerate(diagrams):
        if dgm.ndim == 2:  # noqa: PLR2004
            offsets[i + 1] = offsets[i] + lengths[valid_index]
            valid_index += 1
        else:
            offsets[i + 1] = offsets[i]

    total = int(offsets[-1])
    if total == 0:
        return np.zeros((0, 2), dtype=np.float64), offsets

    # Concatenate all valid diagrams
    nonempty = [dgm for dgm in valid if dgm.shape[0] > 0]
    if nonempty:
        points = np.vstack(nonempty).astype(np.float64, copy=False)
    else:
        points = np.zeros((0, 2), dtype=np.float64)

    return np.ascontiguousarray(points), offsets


def wasserstein_matrix(
    traj_diagrams: list[NDArray[np.float64]],
    rpo_diagrams: list[NDArray[np.float64]],
    delta: float = 0.01,
) -> NDArray[np.float64]:
    """Compute Wasserstein distance matrix using batch API.

    Computes `W[i, j] = W_2(traj_diagrams[i], rpo_diagrams[j])` for all pairs
    in a single C call, eliminating per-pair ctypes overhead.

    Args:
        traj_diagrams: List of I persistence diagrams for trajectory.
        rpo_diagrams: List of J persistence diagrams for RPO.
        delta: Relative error tolerance (default 0.01 = 1%).

    Returns:
        Matrix of shape `(I, J)` with Wasserstein distances.
    """
    lib = _get_lib()

    num_traj = len(traj_diagrams)
    num_rpo = len(rpo_diagrams)

    # Handle edge cases
    if num_traj == 0 or num_rpo == 0:
        return np.empty((num_traj, num_rpo), dtype=np.float64)

    traj_points, traj_offsets = _flatten_diagrams(traj_diagrams)
    rpo_points, rpo_offsets = _flatten_diagrams(rpo_diagrams)

    out = np.empty((num_traj, num_rpo), dtype=np.float64)

    # Get pointers (handle empty points arrays)
    traj_pts_ptr = traj_points.ctypes.data_as(POINTER(c_double)) if traj_points.size > 0 else None
    rpo_pts_ptr = rpo_points.ctypes.data_as(POINTER(c_double)) if rpo_points.size > 0 else None

    lib.wasserstein_matrix_c(
        traj_pts_ptr,
        traj_offsets.ctypes.data_as(POINTER(c_int64)),
        c_int64(num_traj),
        rpo_pts_ptr,
        rpo_offsets.ctypes.data_as(POINTER(c_int64)),
        c_int64(num_rpo),
        c_double(delta),
        out.ctypes.data_as(POINTER(c_double)),
    )

    return out
