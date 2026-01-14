"""Tests for FFT utilities and spatial transforms."""

import numpy as np
import pytest

from ks_shadowing import interleaved_to_complex, l2_distance_all_shifts, to_physical


class TestInterleavedToComplex:
    def test_output_shape(self, rng: np.random.Generator):
        """Output has shape (..., 17) for input (..., 30)."""
        # 30 interleaved -> 15 complex modes + mode 0 + Nyquist = 17
        coeffs = rng.standard_normal(30)
        result = interleaved_to_complex(coeffs)
        assert result.shape == (17,)

        # Batched input
        coeffs_batch = rng.standard_normal((5, 30))
        result_batch = interleaved_to_complex(coeffs_batch)
        assert result_batch.shape == (5, 17)

    def test_zero_padding(self, rng: np.random.Generator):
        """First and last coefficients (mode 0 and Nyquist) are zero."""
        coeffs = rng.standard_normal(30)
        result = interleaved_to_complex(coeffs)
        assert result[0] == 0
        assert result[-1] == 0

    def test_complex_values(self):
        """Correctly combines real and imaginary parts."""
        # [Re(a1), Im(a1), Re(a2), Im(a2), ...]
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


class TestL2DistanceAllShifts:
    def test_self_distance_zero_at_zero_shift(self, rng: np.random.Generator):
        """Distance from field to itself is zero at shift=0."""
        field = rng.standard_normal(128)
        distances = l2_distance_all_shifts(field, field)
        assert distances[0] == pytest.approx(0, abs=1e-10)

    def test_nonnegative(self, rng: np.random.Generator):
        """All distances are non-negative."""
        u = rng.standard_normal(128)
        v = rng.standard_normal(128)
        distances = l2_distance_all_shifts(u, v)
        assert np.all(distances >= 0)

    def test_output_length(self, rng: np.random.Generator):
        """Output length matches input length."""
        n = 256
        u = rng.standard_normal(n)
        v = rng.standard_normal(n)
        distances = l2_distance_all_shifts(u, v)
        assert len(distances) == n

    def test_shifted_field_minimum(self, rng: np.random.Generator):
        """Minimum distance occurs at the correct shift index."""
        field = rng.standard_normal(128)
        shift = 17
        shifted = np.roll(field, shift)

        distances = l2_distance_all_shifts(field, shifted)
        min_idx = np.argmin(distances)
        assert min_idx == shift
        assert distances[min_idx] == pytest.approx(0, abs=1e-10)
