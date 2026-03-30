"""Tests for persistence diagram computation."""

import numpy as np
import pytest

from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha.persistence import (
    _apply_delay_embedding,
    _compute_persistence_diagram,
    _compute_trajectory_diagrams,
)


class TestComputePersistenceDiagram:
    def test_nontrivial_field_returns_pairs(self):
        """A random field with complex structure produces persistence pairs."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(64)

        diagram = _compute_persistence_diagram(field)

        assert len(diagram) >= 1
        assert diagram.ndim == 2
        assert diagram.shape[1] == 2
        assert np.all(diagram[:, 0] < diagram[:, 1])

    def test_constant_fields_empty_diagram(self):
        """Constant periodic fields have no finite persistence pairs."""
        constant = np.ones(32)
        assert len(_compute_persistence_diagram(constant)) == 0

    def test_simple_field_empty_diagram(self):
        """Simple periodic sinusoid yields no finite-death pairs."""
        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        sinusoid = np.sin(x)
        assert len(_compute_persistence_diagram(sinusoid)) == 0

    def test_diagram_shift_invariance(self):
        """Shifted field produces identical persistence diagram."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal(64)
        shifted_field = np.roll(field, 17)

        diagram1 = _compute_persistence_diagram(field)
        diagram2 = _compute_persistence_diagram(shifted_field)

        assert len(diagram1) == len(diagram2)
        if len(diagram1) > 0:
            sorted1 = diagram1[np.lexsort((diagram1[:, 1], diagram1[:, 0]))]
            sorted2 = diagram2[np.lexsort((diagram2[:, 1], diagram2[:, 0]))]
            np.testing.assert_allclose(sorted1, sorted2, rtol=1e-10)

    def test_two_minima_one_pair(self):
        """A field with exactly two local minima produces one persistence pair."""
        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        field = np.cos(2 * x)  # two minima, two maxima

        diagram = _compute_persistence_diagram(field)

        assert diagram.shape == (1, 2)
        np.testing.assert_allclose(diagram[0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(diagram[0, 1], 1.0, atol=1e-6)

    def test_pair_count_matches_local_minima(self):
        """Number of pairs equals number of local minima minus one."""
        x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        # 5 local minima
        field = np.sin(5 * x)

        diagram = _compute_persistence_diagram(field)

        assert len(diagram) == 4


class TestComputeTrajectoryDiagrams:
    def test_returns_valid_diagrams(self):
        """Returns one valid diagram per timestep."""
        rng = np.random.default_rng(42)
        modes = np.zeros((10, 17), dtype=np.complex128)
        modes[:, 1:16] = rng.standard_normal((10, 15)) + 1j * rng.standard_normal((10, 15))
        trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)
        diagrams = _compute_trajectory_diagrams(trajectory)

        assert len(diagrams) == 10
        for diagram in diagrams:
            assert diagram.ndim == 2
            assert diagram.shape[1] == 2


class TestChunkedTrajectoryDiagrams:
    def test_chunked_matches_unchunked(self):
        """Chunked diagram computation produces identical diagrams."""
        rng = np.random.default_rng(42)
        modes = np.zeros((50, 17), dtype=np.complex128)
        modes[:, 1:16] = rng.standard_normal((50, 15)) + 1j * rng.standard_normal((50, 15))
        trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)

        diagrams_default = _compute_trajectory_diagrams(trajectory)
        diagrams_chunked = _compute_trajectory_diagrams(trajectory, chunk_size=10)

        assert len(diagrams_default) == len(diagrams_chunked)
        for d1, d2 in zip(diagrams_default, diagrams_chunked, strict=True):
            np.testing.assert_allclose(d1, d2)


class TestApplyDelayEmbedding:
    def test_delay_one_unchanged(self):
        """With delay=1, output equals input."""
        matrix = np.random.default_rng(42).random((10, 8))
        result = _apply_delay_embedding(matrix, delay=1)
        np.testing.assert_allclose(result, matrix)

    def test_sums_consecutive_entries(self):
        """Verifies the sum is computed correctly along the diagonal."""
        matrix = np.arange(12).reshape(4, 3).astype(float)
        # With delay=2, result[0, 0] = matrix[0, 0] + matrix[1, 1] = 0 + 4 = 4
        # result[0, 1] = matrix[0, 1] + matrix[1, 2] = 1 + 5 = 6
        # result[0, 2] = matrix[0, 2] + matrix[1, 0] = 2 + 3 = 5 (wraparound)
        result = _apply_delay_embedding(matrix, delay=2)

        assert result.shape == (3, 3)
        assert result[0, 0] == pytest.approx(4.0)
        assert result[0, 1] == pytest.approx(6.0)
        assert result[0, 2] == pytest.approx(5.0)

    def test_invalid_delay_raises(self):
        """Delay < 1 or > trajectory length raises ValueError."""
        matrix = np.random.default_rng(42).random((10, 5))

        with pytest.raises(ValueError):
            _apply_delay_embedding(matrix, delay=0)

        with pytest.raises(ValueError):
            _apply_delay_embedding(matrix, delay=11)
