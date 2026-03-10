"""Tests for FFT utilities and spatial transforms."""

import numpy as np

from ks_shadowing.core.transforms import (
    interleaved_to_complex,
    tile_periodic,
    to_comoving_frame,
    to_physical,
)


class TestInterleavedToComplex:
    def test_output_shape(self, rng: np.random.Generator):
        """Output has shape (..., 17) for input (..., 30)."""
        coeffs = rng.standard_normal(30)
        result = interleaved_to_complex(coeffs)
        assert result.shape == (17,)

    def test_batched_input(self, rng: np.random.Generator):
        """Supports batched input with shape (batch, 30)."""
        coeffs_batch = rng.standard_normal((5, 30))
        result_batch = interleaved_to_complex(coeffs_batch)
        assert result_batch.shape == (5, 17)

    def test_zero_padding(self, rng: np.random.Generator):
        """Mode 0 and Nyquist mode are zero-padded."""
        coeffs = rng.standard_normal(30)
        result = interleaved_to_complex(coeffs)
        assert result[0] == 0
        assert result[-1] == 0

    def test_interleaving(self):
        """Correctly combines interleaved real and imaginary parts."""
        coeffs = np.array([1, 2, 3, 4] + [0] * 26, dtype=np.float64)
        result = interleaved_to_complex(coeffs)
        assert result[1] == 1 + 2j
        assert result[2] == 3 + 4j


class TestToPhysical:
    def test_output_shape(self, rng: np.random.Generator):
        """Output shape matches requested resolution."""
        coeffs = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        coeffs[0] = 0  # mode 0
        coeffs[-1] = 0  # Nyquist

        result = to_physical(coeffs, 2048)
        assert result.shape == (2048,)

        result_small = to_physical(coeffs, 64)
        assert result_small.shape == (64,)


class TestToComovingFrame:
    def test_output_shape(self, rng: np.random.Generator):
        """Output has same shape as input."""
        trajectory = rng.standard_normal((100, 64))
        result = to_comoving_frame(trajectory, drift_per_step=0.5)
        assert result.shape == trajectory.shape

    def test_zero_drift_unchanged(self, rng: np.random.Generator):
        """Zero drift rate leaves trajectory unchanged."""
        trajectory = rng.standard_normal((50, 128))
        result = to_comoving_frame(trajectory, drift_per_step=0.0)
        np.testing.assert_allclose(result, trajectory, rtol=1e-6, atol=1e-6)

    def test_integer_drift_matches_roll(self, rng: np.random.Generator):
        """Integer drift per step matches np.roll at each timestep."""
        resolution = 64
        num_steps = 10
        trajectory = rng.standard_normal((num_steps, resolution))
        drift_per_step = 3.0  # Exactly 3 grid cells per step

        result = to_comoving_frame(trajectory, drift_per_step)

        # At timestep i, field should be rolled by -drift_per_step * i
        for i in range(num_steps):
            expected = np.roll(trajectory[i], -int(drift_per_step * i))
            np.testing.assert_allclose(result[i], expected, rtol=1e-6, atol=1e-6)

    def test_periodic_signal_becomes_constant(self):
        """A drifting periodic signal becomes stationary in co-moving frame."""
        resolution = 64
        num_steps = 20
        drift_per_step = 2.0

        # Create a signal that drifts by drift_per_step per timestep
        x = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        base_signal = np.sin(x)

        trajectory = np.zeros((num_steps, resolution))
        for i in range(num_steps):
            trajectory[i] = np.roll(base_signal, int(drift_per_step * i))

        result = to_comoving_frame(trajectory, drift_per_step)

        # All timesteps should now be (approximately) the same
        for i in range(1, num_steps):
            np.testing.assert_allclose(result[i], result[0], rtol=1e-6, atol=1e-6)


class TestTilePeriodic:
    def test_tiles_to_cover_target(self, rng: np.random.Generator):
        """Tiles enough times to reach at least target_length."""
        period = 30
        field = rng.standard_normal((period, 64))

        result = tile_periodic(field, target_length=100)
        assert result.shape[0] == 120  # 4 tiles of 30

        result = tile_periodic(field, target_length=61)
        assert result.shape[0] == 90  # 3 tiles of 30

    def test_preserves_periodicity(self, rng: np.random.Generator):
        """Tiled result repeats the original period."""
        period = 25
        field = rng.standard_normal((period, 32))
        result = tile_periodic(field, target_length=100)

        for i in range(result.shape[0] - period):
            np.testing.assert_array_equal(result[i], result[i + period])
