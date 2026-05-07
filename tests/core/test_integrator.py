"""Tests for the KS integrator wrapper."""

import numpy as np
import pytest

from ks_shadowing.core.integrator import (
    _complex_to_interleaved,
    _interleaved_to_complex,
    ksint,
)


def test_complex_interleaved_roundtrip(sample_initial_state: np.ndarray) -> None:
    """Complex -> interleaved -> complex preserves the input."""
    interleaved = _complex_to_interleaved(sample_initial_state)
    recovered = _interleaved_to_complex(interleaved)
    np.testing.assert_array_equal(recovered, sample_initial_state)


def test_ksint_deterministic_and_preserves_initial(
    sample_initial_state: np.ndarray,
) -> None:
    """``ksint`` returns ``(steps + 1, 17)`` complex128 with row 0 equal to
    ``initial_state``, and is deterministic across calls."""
    result = ksint(sample_initial_state, dt=0.25, steps=50)
    assert result.shape == (51, 17)
    assert result.dtype == np.complex128
    np.testing.assert_array_equal(result[0], sample_initial_state)

    repeat = ksint(sample_initial_state, dt=0.25, steps=50)
    np.testing.assert_array_equal(result, repeat)


def test_ksint_invalid_shape_raises() -> None:
    """``ksint`` raises ``ValueError`` when ``initial_state`` is not shape
    ``(17,)``."""
    with pytest.raises(ValueError, match="shape"):
        ksint(np.zeros(16, dtype=np.complex128), dt=0.25, steps=10)
    with pytest.raises(ValueError, match="shape"):
        ksint(np.zeros((17, 2), dtype=np.complex128), dt=0.25, steps=10)


def test_ksint_save_interval(sample_initial_state: np.ndarray) -> None:
    """Integrating with ``save_interval=k`` returns every kth row of the
    full-resolution integration."""
    dt = 0.02
    steps = 46  # multiple of save_interval=23
    save_interval = 23

    full = ksint(sample_initial_state, dt, steps)
    sparse = ksint(sample_initial_state, dt, steps, save_interval=save_interval)

    assert sparse.shape == (steps // save_interval + 1, 17)
    np.testing.assert_allclose(sparse, full[::save_interval], rtol=1e-12, atol=1e-12)
