"""Tests for Wasserstein distance computation."""

import numpy as np
import pytest

from ks_shadowing.pha.persistence import _compute_persistence_diagram
from ks_shadowing.pha.wasserstein import _wasserstein_matrix


class TestWassersteinMatrix:
    def test_self_distance_zero(self):
        """Diagonal of W(diagrams, diagrams) is zero."""
        rng = np.random.default_rng(42)
        diagrams = [_compute_persistence_diagram(rng.standard_normal(32)) for _ in range(4)]

        result = _wasserstein_matrix(diagrams, diagrams)

        assert result.shape == (4, 4)
        for i in range(4):
            assert result[i, i] == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        """W(A, B) == W(B, A)^T."""
        rng = np.random.default_rng(42)
        diagrams_a = [_compute_persistence_diagram(rng.standard_normal(32)) for _ in range(3)]
        diagrams_b = [_compute_persistence_diagram(rng.standard_normal(32)) for _ in range(4)]

        ab = _wasserstein_matrix(diagrams_a, diagrams_b)
        ba = _wasserstein_matrix(diagrams_b, diagrams_a)

        np.testing.assert_allclose(ab, ba.T, rtol=1e-10)

    def test_positive_distances(self):
        """Distances between different diagrams are positive."""
        rng = np.random.default_rng(42)
        diagrams_a = [_compute_persistence_diagram(rng.standard_normal(64)) for _ in range(3)]
        diagrams_b = [_compute_persistence_diagram(rng.standard_normal(64)) for _ in range(3)]

        result = _wasserstein_matrix(diagrams_a, diagrams_b)

        assert np.all(result >= 0)

    def test_empty_diagrams_in_batch(self):
        """Handles empty diagrams correctly in batch."""
        diagrams_with_empty = [
            np.zeros((0, 2)),
            np.array([[0.0, 1.0]]),
        ]
        diagrams_nonempty = [
            np.array([[0.0, 2.0]]),
            np.zeros((0, 2)),
        ]

        result = _wasserstein_matrix(diagrams_with_empty, diagrams_nonempty)

        assert result.shape == (2, 2)
        # Empty vs empty = 0
        assert result[0, 1] == pytest.approx(0.0)
        # Empty vs non-empty = projection distance
        assert result[0, 0] > 0
        # Non-empty vs empty
        assert result[1, 1] > 0

    def test_empty_input_lists(self):
        """Handles empty input lists."""
        rng = np.random.default_rng(42)
        diagrams = [_compute_persistence_diagram(rng.standard_normal(32)) for _ in range(3)]

        # Empty trajectory list
        result = _wasserstein_matrix([], diagrams)
        assert result.shape == (0, 3)

        # Empty RPO list
        result = _wasserstein_matrix(diagrams, [])
        assert result.shape == (3, 0)

        # Both empty
        result = _wasserstein_matrix([], [])
        assert result.shape == (0, 0)
