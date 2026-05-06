"""Tests for the C++ batch union-find wrapper."""

import numpy as np

from ks_shadowing.core.unionfind import _find_components


def test_no_edges_all_distinct() -> None:
    """``_find_components`` with no edges assigns each element a unique
    component label."""
    labels = _find_components(5, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
    assert len(np.unique(labels)) == 5


def test_transitive_merge_and_disjoint() -> None:
    """Edges ``0-1`` and ``1-2`` collapse elements 0, 1, 2 into one
    component; isolated edges ``0-1`` and ``2-3`` produce two components."""
    labels = _find_components(5, np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32))
    assert labels[0] == labels[1] == labels[2]

    labels = _find_components(4, np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32))
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
