"""Tests for Wasserstein distance computation."""

import numpy as np
import pytest

from ks_shadowing.pha.persistence import compute_persistence_diagram
from ks_shadowing.pha.wasserstein import (
    wasserstein_distance,
    wasserstein_matrix,
)


class TestWassersteinDistance:
    def test_self_distance_zero(self):
        """Distance from diagram to itself is zero."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(64)
        diagram = compute_persistence_diagram(field)

        dist = wasserstein_distance(diagram, diagram)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_empty_diagrams(self):
        """Distance between empty diagrams is zero."""
        empty = np.array([]).reshape(0, 2)
        dist = wasserstein_distance(empty, empty)
        assert dist == 0.0

    def test_symmetric(self):
        """Distance is symmetric: d(A, B) = d(B, A)."""
        rng = np.random.default_rng(42)
        diagram1 = compute_persistence_diagram(rng.standard_normal(64))
        diagram2 = compute_persistence_diagram(rng.standard_normal(64))

        assert wasserstein_distance(diagram1, diagram2) == pytest.approx(
            wasserstein_distance(diagram2, diagram1)
        )


class TestWassersteinMatrix:
    def test_batch_matches_individual(self):
        """Batch API matches individual wasserstein_distance calls."""
        rng = np.random.default_rng(42)
        traj_diagrams = [compute_persistence_diagram(rng.standard_normal(32)) for _ in range(5)]
        rpo_diagrams = [compute_persistence_diagram(rng.standard_normal(32)) for _ in range(4)]

        batch_result = wasserstein_matrix(traj_diagrams, rpo_diagrams)

        assert batch_result.shape == (5, 4)
        for i, traj in enumerate(traj_diagrams):
            for j, rpo in enumerate(rpo_diagrams):
                individual = wasserstein_distance(traj, rpo)
                assert batch_result[i, j] == pytest.approx(individual, rel=1e-10)

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

        result = wasserstein_matrix(diagrams_with_empty, diagrams_nonempty)

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
        diagrams = [compute_persistence_diagram(rng.standard_normal(32)) for _ in range(3)]

        # Empty trajectory list
        result = wasserstein_matrix([], diagrams)
        assert result.shape == (0, 3)

        # Empty RPO list
        result = wasserstein_matrix(diagrams, [])
        assert result.shape == (3, 0)

        # Both empty
        result = wasserstein_matrix([], [])
        assert result.shape == (0, 0)
