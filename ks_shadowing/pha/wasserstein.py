"""`Hera <https://github.com/anigmetov/hera>`_ library bindings for Wasserstein
distance computation.
"""

from ctypes import CDLL, POINTER, c_double, c_int64
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


def _flatten_diagrams(
    diagrams: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Flatten diagrams into (points, offsets) arrays for the batch C API."""
    if not diagrams:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64)

    assert all(dgm.ndim == 2 for dgm in diagrams), "All diagrams must be 2D arrays"  # noqa: PLR2004

    lengths = np.array([dgm.shape[0] for dgm in diagrams], dtype=np.int64)
    offsets = np.zeros(len(diagrams) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    total = int(offsets[-1])
    if total == 0:
        return np.zeros((0, 2), dtype=np.float64), offsets

    nonempty = [dgm for dgm in diagrams if dgm.shape[0] > 0]
    points = np.vstack(nonempty).astype(np.float64, copy=False)

    return np.ascontiguousarray(points), offsets


def _wasserstein_matrix(
    traj_diagrams: list[NDArray[np.float64]],
    rpo_diagrams: list[NDArray[np.float64]],
    delta: float = 0.01,
) -> NDArray[np.float64]:
    r"""Compute Wasserstein distance matrix using batch C API.

    Computes :math:`W[i, j] = W_2(\text{traj}[i], \text{rpo}[j])` for all pairs
    in a single C call, eliminating per-pair ctypes overhead.

    Parameters
    ----------
    traj_diagrams : list[NDArray[np.float64]]
        List of ``I`` persistence diagrams for the trajectory.
    rpo_diagrams : list[NDArray[np.float64]]
        List of ``J`` persistence diagrams for the RPO.
    delta : float, optional
        Relative error tolerance. Default is 0.01 (1%).

    Returns
    -------
    NDArray[np.float64], shape (I, J)
        Wasserstein distance matrix.
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


def _wasserstein_column(
    traj_points: NDArray[np.float64],
    traj_offsets: NDArray[np.int64],
    num_traj: int,
    rpo_diagram: NDArray[np.float64],
    delta: float = 0.01,
) -> NDArray[np.float64]:
    r"""Compute Wasserstein distances from pre-flattened trajectory diagrams to one RPO diagram.

    Parameters
    ----------
    traj_points : NDArray[np.float64], shape (total_points, 2)
        Concatenated persistence points from all trajectory diagrams.
    traj_offsets : NDArray[np.int64], shape (num_traj + 1,)
        Cumulative offsets into ``traj_points`` for each diagram.
    num_traj : int
        Number of trajectory diagrams.
    rpo_diagram : NDArray[np.float64], shape (n_points, 2)
        Single RPO persistence diagram.
    delta : float, optional
        Relative error tolerance. Default is 0.01 (1%).

    Returns
    -------
    NDArray[np.float64], shape (num_traj,)
        Wasserstein distance from each trajectory diagram to the RPO diagram.
    """
    if num_traj == 0:
        return np.empty(0, dtype=np.float64)

    lib = _get_lib()

    rpo_points, rpo_offsets = _flatten_diagrams([rpo_diagram])
    out = np.empty((num_traj, 1), dtype=np.float64)

    traj_ptr = traj_points.ctypes.data_as(POINTER(c_double)) if traj_points.size > 0 else None
    rpo_ptr = rpo_points.ctypes.data_as(POINTER(c_double)) if rpo_points.size > 0 else None

    lib.wasserstein_matrix_c(
        traj_ptr,
        traj_offsets.ctypes.data_as(POINTER(c_int64)),
        c_int64(num_traj),
        rpo_ptr,
        rpo_offsets.ctypes.data_as(POINTER(c_int64)),
        c_int64(1),
        c_double(delta),
        out.ctypes.data_as(POINTER(c_double)),
    )

    return out[:, 0]
