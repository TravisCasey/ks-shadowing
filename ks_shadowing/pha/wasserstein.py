"""`Hera <https://github.com/anigmetov/hera>`_ library bindings for batched
Wasserstein distance computation.
"""

from ctypes import CDLL, POINTER, c_double, c_int, c_int64
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.pha.persistence import KSPersistenceTrajectory


@cache
def _get_lib() -> CDLL:
    """Load and return the cached Wasserstein shared object."""
    so_path = Path(__file__).parent / "libhera2py.so"
    if not so_path.exists():
        raise RuntimeError(f"Could not find libhera2py.so at {so_path}.")
    lib = CDLL(str(so_path))

    lib.wasserstein_column_c.argtypes = [
        POINTER(c_double),  # dgms_a
        POINTER(c_int64),  #  offsets_a
        c_int64,  #           num_a
        POINTER(c_double),  # dgm_b
        c_int64,  #           length_b
        c_double,  #          delta
        POINTER(c_double),  # out
    ]
    lib.wasserstein_column_c.restype = c_int

    return lib


def _wasserstein_column(  # noqa: PLR0913, PLR0917
    diagrams_a: NDArray[np.float64],
    offsets_a: NDArray[np.int64],
    essential_births_a: NDArray[np.float64],
    diagram_b: NDArray[np.float64],
    essential_births_b: NDArray[np.float64],
    delta: float = 0.01,
) -> NDArray[np.float64]:
    r"""Compute Wasserstein distances from pre-flattened trajectory diagrams to
    one RPO diagram.

    See ``KSPersistenceTrajectory._flatten`` to format a trajectory of
    persistence diagrams into the expected ``diagrams_a``, ``offsets_a`` form.

    If any of the three input arrays are not contiguous and row-major, a copy
    is made to pass to the batched C API.

    Parameters
    ----------
    diagrams_a : NDArray[np.float64], shape (offsets_a[-1], 2)
        Concatenated finite pairs (row-major) from a set of diagrams.
    offsets_a : NDArray[np.int64], shape (num_diagrams + 1,)
        Cumulative offsets into ``diagrams_a`` for each diagram. The trailing
        sentinel ``offsets_a[-1]`` equals ``diagrams_a.shape[0]``.
    essential_births_a : NDArray[np.float64], shape (num_diagrams, 2)
        Essential births of each diagram in ``diagrams_a``.
    diagram_b : NDArray[np.float64], shape (num_pairs_b, 2)
        Single comparison diagram of finite pairs (row-major).
    essential_births_b : NDArray[np.float64], shape (2,)
        Essential births of ``diagram_b``.
    delta : float, optional
        Relative error tolerance for the finite part. Default is 0.01 (1%).

    Returns
    -------
    NDArray[np.float64], shape (num_diagrams,)
        Wasserstein distance from each diagram in ``diagrams_a`` to
        ``diagram_b``.
    """
    num_diagrams_a = offsets_a.size - 1
    if num_diagrams_a <= 0:
        return np.empty(0, dtype=np.float64)

    lib = _get_lib()

    diagrams_a = np.ascontiguousarray(diagrams_a.astype(np.float64, copy=False))
    offsets_a = np.ascontiguousarray(offsets_a.astype(np.int64, copy=False))
    diagram_b_length = diagram_b.shape[0]
    diagram_b = np.ascontiguousarray(diagram_b.astype(np.float64, copy=False))

    out = np.empty(num_diagrams_a, dtype=np.float64)

    diagrams_a_ptr = diagrams_a.ctypes.data_as(POINTER(c_double)) if diagrams_a.size > 0 else None
    diagram_b_ptr = diagram_b.ctypes.data_as(POINTER(c_double)) if diagram_b.size > 0 else None

    ret = lib.wasserstein_column_c(
        diagrams_a_ptr,
        offsets_a.ctypes.data_as(POINTER(c_int64)),
        c_int64(num_diagrams_a),
        diagram_b_ptr,
        c_int64(diagram_b_length),
        c_double(delta),
        out.ctypes.data_as(POINTER(c_double)),
    )
    if ret != 0:
        raise RuntimeError("wasserstein_column_c failed")

    essential_sq = np.sum((essential_births_a - essential_births_b) ** 2, axis=1)
    return np.sqrt(out**2 + essential_sq)


def wasserstein_matrix(
    diagrams_a: KSPersistenceTrajectory,
    diagrams_b: KSPersistenceTrajectory,
    delta: float = 0.01,
) -> NDArray[np.float64]:
    """Compute Wasserstein distances between every pair of persistence diagrams.

    Entry ``(i, j)`` is the distance between diagram ``i`` of ``diagrams_a`` and
    diagram ``j`` of ``diagrams_b``. The result holds
    ``len(diagrams_a) * len(diagrams_b)`` float64 entries. Each entry is the
    distance between full diagrams: finite pairs plus the essential classes,
    see ``_wasserstein_column``.

    Detection never materializes this matrix; it streams one column at a time
    and reduces as it goes, so
    :func:`~ks_shadowing.pha.detection.compute_min_distances` is the right entry
    point for threshold selection. This function is for analyses that need the
    distances themselves.

    Parameters
    ----------
    diagrams_a : :class:`~ks_shadowing.pha.persistence.KSPersistenceTrajectory`
        Diagrams indexing the rows of the result.
    diagrams_b : :class:`~ks_shadowing.pha.persistence.KSPersistenceTrajectory`
        Diagrams indexing the columns of the result.
    delta : float, optional
        Relative error tolerance. Default is 0.01 (1%).

    Returns
    -------
    NDArray[np.float64], shape (len(diagrams_a), len(diagrams_b))
        Wasserstein distance between each pair of diagrams.
    """
    flat_diagrams, offsets, essential_births = diagrams_a._flatten()

    matrix = np.empty((len(diagrams_a), len(diagrams_b)), dtype=np.float64)
    for index, diagram in enumerate(diagrams_b):
        matrix[:, index] = _wasserstein_column(
            flat_diagrams,
            offsets,
            essential_births,
            diagram,
            diagrams_b.essential_births[index],
            delta,
        )

    return matrix
