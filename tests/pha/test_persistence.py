"""Tests for persistence diagram computation."""

import numpy as np
import pytest

from ks_shadowing.pha.persistence import (
    apply_delay_embedding,
    compute_persistence_diagram,
    compute_trajectory_diagrams,
    compute_wasserstein_matrix,
    wasserstein_distance,
)


class TestComputePersistenceDiagram:
    def test_nontrivial_field_returns_pairs(self):
        """A random field with complex structure produces persistence pairs."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(64)

        diagram = compute_persistence_diagram(field)

        assert len(diagram) >= 1
        assert diagram.ndim == 2
        assert diagram.shape[1] == 2
        assert np.all(diagram[:, 0] < diagram[:, 1])

    def test_constant_fields_empty_diagram(self):
        """Constant periodic fields have no finite persistence pairs."""
        constant = np.ones(32)
        assert len(compute_persistence_diagram(constant)) == 0

    def test_simple_field_empty_diagram(self):
        """Simple periodic sinusoid yields no finite-death pairs."""
        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        sinusoid = np.sin(x)
        assert len(compute_persistence_diagram(sinusoid)) == 0

    def test_diagram_shift_invariance(self):
        """Shifted field produces identical persistence diagram."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(64)
        shifted_field = np.roll(field, 17)

        diagram1 = compute_persistence_diagram(field)
        diagram2 = compute_persistence_diagram(shifted_field)

        assert len(diagram1) == len(diagram2)
        if len(diagram1) > 0:
            sorted1 = diagram1[np.lexsort((diagram1[:, 1], diagram1[:, 0]))]
            sorted2 = diagram2[np.lexsort((diagram2[:, 1], diagram2[:, 0]))]
            np.testing.assert_allclose(sorted1, sorted2, rtol=1e-10)


class TestComputeTrajectoryDiagrams:
    def test_returns_valid_diagrams(self):
        """Returns one valid diagram per timestep."""
        trajectory = np.random.default_rng(42).standard_normal((10, 32))
        diagrams = compute_trajectory_diagrams(trajectory)

        assert len(diagrams) == 10
        for diagram in diagrams:
            assert diagram.ndim == 2
            assert diagram.shape[1] == 2


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


class TestComputeWassersteinMatrix:
    def test_output_shape_and_diagonal(self):
        """Output has correct shape; diagonal is zero for identical lists."""
        rng = np.random.default_rng(42)
        diagrams = [compute_persistence_diagram(rng.standard_normal(32)) for _ in range(4)]

        matrix = compute_wasserstein_matrix(diagrams, diagrams)

        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-10)


class TestApplyDelayEmbedding:
    def test_delay_one_unchanged(self):
        """With delay=1, output equals input."""
        matrix = np.random.default_rng(42).random((10, 8))
        result = apply_delay_embedding(matrix, delay=1)
        np.testing.assert_allclose(result, matrix)

    def test_sums_consecutive_entries(self):
        """Verifies the sum is computed correctly along the diagonal."""
        matrix = np.arange(12).reshape(4, 3).astype(float)
        # With delay=2, result[0, 0] = matrix[0, 0] + matrix[1, 1] = 0 + 4 = 4
        # result[0, 1] = matrix[0, 1] + matrix[1, 2] = 1 + 5 = 6
        # result[0, 2] = matrix[0, 2] + matrix[1, 0] = 2 + 3 = 5 (wraparound)
        result = apply_delay_embedding(matrix, delay=2)

        assert result.shape == (3, 3)
        assert result[0, 0] == pytest.approx(4.0)
        assert result[0, 1] == pytest.approx(6.0)
        assert result[0, 2] == pytest.approx(5.0)

    def test_invalid_delay_raises(self):
        """Delay < 1 or > trajectory length raises ValueError."""
        matrix = np.random.default_rng(42).random((10, 5))

        with pytest.raises(ValueError):
            apply_delay_embedding(matrix, delay=0)

        with pytest.raises(ValueError):
            apply_delay_embedding(matrix, delay=11)
