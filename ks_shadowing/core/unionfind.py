"""Batch union-find via C++ for connected component labeling."""

from ctypes import CDLL, POINTER, c_int32, c_int64
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_lib: CDLL | None = None


def _get_lib() -> CDLL:
    """Return the cached union-find library singleton."""
    global _lib  # noqa: PLW0603
    if _lib is None:
        so_path = Path(__file__).parent / "libunionfind2py.so"
        _lib = CDLL(str(so_path))

        _lib.connected_components_c.argtypes = [
            c_int32,  # n
            POINTER(c_int32),  # edges_a
            POINTER(c_int32),  # edges_b
            c_int64,  # num_edges
            POINTER(c_int32),  # out
        ]
        _lib.connected_components_c.restype = None

    return _lib


def _find_components(
    n: int,
    edges_a: NDArray[np.int32],
    edges_b: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Compute connected component labels using batch union-find in C++.

    Takes edge pairs and returns a label array where each element is assigned
    its component root. Uses path compression and union by rank internally.

    Parameters
    ----------
    n : int
        Number of elements (indexed 0 to ``n - 1``).
    edges_a : NDArray[np.int32], shape (num_edges,)
        First element of each edge pair.
    edges_b : NDArray[np.int32], shape (num_edges,)
        Second element of each edge pair.

    Returns
    -------
    NDArray[np.int32], shape (n,)
        Component root for each element.
    """
    lib = _get_lib()

    edges_a = np.ascontiguousarray(edges_a, dtype=np.int32)
    edges_b = np.ascontiguousarray(edges_b, dtype=np.int32)
    num_edges = len(edges_a)
    out = np.empty(n, dtype=np.int32)

    lib.connected_components_c(
        c_int32(n),
        edges_a.ctypes.data_as(POINTER(c_int32)) if num_edges > 0 else None,
        edges_b.ctypes.data_as(POINTER(c_int32)) if num_edges > 0 else None,
        c_int64(num_edges),
        out.ctypes.data_as(POINTER(c_int32)),
    )

    return out
