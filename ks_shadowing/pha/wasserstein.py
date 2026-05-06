"""`Hera <https://github.com/anigmetov/hera>`_ library bindings for batched
Wasserstein distance computation.
"""

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

        _lib.wasserstein_column_c.argtypes = [
            POINTER(c_double),  # dgms_a
            POINTER(c_int64),  #  offsets_a
            c_int64,  #           num_a
            POINTER(c_double),  # dgm_b
            c_int64,  #           length_b
            c_double,  #          delta
            POINTER(c_double),  # out
        ]
        _lib.wasserstein_column_c.restype = c_int

    return _lib


def _wasserstein_column(
    diagrams_a: NDArray[np.float64],
    offsets_a: NDArray[np.int64],
    diagram_b: NDArray[np.float64],
    delta: float = 0.01,
) -> NDArray[np.float64]:
    """Compute Wasserstein distances from pre-flattened trajectory diagrams to
    one RPO diagram.

    See :meth:`~ks_shadowing.pha.persistence._KSPersistenceTrajectory._flatten`
    to format a trajectory of persistence diagrams into the expected
    ``diagrams_a``, ``offsets_a`` form.

    If any of the three input arrays are not contiguous and row-major, a copy
    is made to pass to the batched C API.

    Parameters
    ----------
    diagrams_a : NDArray[np.float64], shape (offsets_a[-1], 2)
        Concatenated persistence pairs (row-major) from a set of diagrams.
    offsets_a : NDArray[np.int64], shape (num_diagrams + 1,)
        Cumulative offsets into ``diagrams_a`` for each diagram. The trailing
        sentinel ``offsets_a[-1]`` equals ``diagrams_a.shape[0]``.
    diagram_b : NDArray[np.float64]
        Single comparison diagram of persistence pairs (row-major).
    delta : float, optional
        Relative error tolerance. Default is 0.01 (1%).

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
    rpo_ptr = diagram_b.ctypes.data_as(POINTER(c_double)) if diagram_b.size > 0 else None

    ret = lib.wasserstein_column_c(
        diagrams_a_ptr,
        offsets_a.ctypes.data_as(POINTER(c_int64)),
        c_int64(num_diagrams_a),
        rpo_ptr,
        c_int64(diagram_b_length),
        c_double(delta),
        out.ctypes.data_as(POINTER(c_double)),
    )
    if ret != 0:
        raise RuntimeError("wasserstein_column_c failed")

    return out
