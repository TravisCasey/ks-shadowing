"""Tests for utility functions."""

import numpy as np

from ks_shadowing.core.unionfind import _find_components


class TestFindComponents:
    """Tests for the C++ batch union-find."""

    def test_no_edges(self):
        """Each element is its own component when there are no edges."""
        labels = _find_components(5, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        # Each element should have a unique label (itself)
        assert len(np.unique(labels)) == 5

    def test_single_edge(self):
        """A single edge merges two elements."""
        labels = _find_components(5, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32))
        assert labels[0] == labels[1]
        assert len(np.unique(labels)) == 4

    def test_transitive_merge(self):
        """Chained edges result in a single component."""
        labels = _find_components(
            5,
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
        )
        assert labels[0] == labels[1] == labels[2]

    def test_disjoint_components(self):
        """Separate edge sets remain separate components."""
        labels = _find_components(
            4,
            np.array([0, 2], dtype=np.int32),
            np.array([1, 3], dtype=np.int32),
        )
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
