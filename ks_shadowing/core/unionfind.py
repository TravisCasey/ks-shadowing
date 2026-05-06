"""Batch union-find via C++ for connected component labeling."""

from ctypes import CDLL, POINTER, c_int, c_int32
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@cache
def _get_lib() -> CDLL:
    """Load and return the cached union-find shared object."""
    so_path = Path(__file__).parent / "libunionfind2py.so"
    lib = CDLL(str(so_path))

    lib.connected_components_c.argtypes = [
        c_int32,  # n
        POINTER(c_int32),  # edges_a
        POINTER(c_int32),  # edges_b
        c_int32,  # num_edges
        POINTER(c_int32),  # out
    ]
    lib.connected_components_c.restype = c_int

    return lib


def _find_components(
    num_elements: int,
    edges_a: NDArray[np.int32],
    edges_b: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Compute connected component labels using batch union-find in C++.

    Takes edge pairs and returns a label array where each element is assigned
    its component root. Uses path compression and union by rank internally.

    Parameters
    ----------
    num_elements : int
        Number of elements (indexed 0 to ``num_elements - 1``).
    edges_a : NDArray[np.int32], shape (num_edges,)
        First element of each edge pair.
    edges_b : NDArray[np.int32], shape (num_edges,)
        Second element of each edge pair.

    Returns
    -------
    NDArray[np.int32], shape (num_elements,)
        Component root for each element.
    """
    lib = _get_lib()

    edges_a = np.ascontiguousarray(edges_a, dtype=np.int32)
    edges_b = np.ascontiguousarray(edges_b, dtype=np.int32)
    num_edges = len(edges_a)
    out = np.empty(num_elements, dtype=np.int32)

    ret = lib.connected_components_c(
        c_int32(num_elements),
        edges_a.ctypes.data_as(POINTER(c_int32)) if num_edges > 0 else None,
        edges_b.ctypes.data_as(POINTER(c_int32)) if num_edges > 0 else None,
        c_int32(num_edges),
        out.ctypes.data_as(POINTER(c_int32)),
    )
    if ret != 0:
        raise RuntimeError("connected_components_c failed")

    return out
