"""Tests for the KS integrator and its wrapper."""

import numpy as np
import pytest

from ks_shadowing import ksint


class TestKsint:
    def test_output_shape(self, sample_initial_state: np.ndarray):
        """Output shape matches `(steps // save_every + 1, 30)`."""
        result = ksint(sample_initial_state, dt=0.25, steps=100, save_every=10)
        assert result.shape == (11, 30)

    def test_initial_condition_preserved(self, sample_initial_state: np.ndarray):
        """First row of output equals input initial state."""
        result = ksint(sample_initial_state, dt=0.25, steps=10)
        np.testing.assert_array_equal(result[0], sample_initial_state)

    def test_deterministic(self, sample_initial_state: np.ndarray):
        """Same input produces identical output."""
        result1 = ksint(sample_initial_state, dt=0.25, steps=50)
        result2 = ksint(sample_initial_state, dt=0.25, steps=50)
        np.testing.assert_array_equal(result1, result2)

    def test_invalid_shape_raises(self):
        """Non-(30,) input raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            ksint(np.zeros(29), dt=0.25, steps=10)

        with pytest.raises(ValueError, match="shape"):
            ksint(np.zeros((30, 2)), dt=0.25, steps=10)

    def test_invalid_steps_raises(self, sample_initial_state: np.ndarray):
        """Non-positive steps raises ValueError."""
        with pytest.raises(ValueError, match="steps"):
            ksint(sample_initial_state, dt=0.25, steps=0)

    def test_save_every_exceeds_steps_raises(self, sample_initial_state: np.ndarray):
        """save_every > steps raises ValueError."""
        with pytest.raises(ValueError, match="save_every"):
            ksint(sample_initial_state, dt=0.25, steps=10, save_every=20)
