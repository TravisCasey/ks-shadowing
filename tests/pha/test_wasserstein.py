"""Tests for batched Wasserstein distance bindings."""

import numpy as np

from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha.persistence import KSPersistenceTrajectory
from ks_shadowing.pha.wasserstein import _wasserstein_column


def _flatten(diagrams: list) -> tuple:
    """Flatten ``diagrams`` via ``KSPersistenceTrajectory._flatten``."""
    return KSPersistenceTrajectory(diagrams=diagrams, dt=0.02)._flatten()


def test_flatten_offsets_for_variable_size_diagrams() -> None:
    """``_flatten`` produces cumulative ``offsets`` that index each diagram's
    starting row in the concatenated array."""
    diagrams = [
        np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
        np.array([[1.0, 3.0]], dtype=np.float64),
    ]
    flat, offsets = _flatten(diagrams)
    np.testing.assert_array_equal(offsets, [0, 2, 2, 3])
    assert flat.shape == (3, 2)
    np.testing.assert_array_equal(
        flat,
        np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]], dtype=np.float64),
    )


def test_wasserstein_column_self_zero(rng: np.random.Generator) -> None:
    """``_wasserstein_column`` returns 0 for a diagram compared against
    itself."""
    modes = np.zeros((4, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((4, 15)) + 1j * rng.standard_normal((4, 15))) * 0.1
    trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)
    diagrams = KSPersistenceTrajectory.from_trajectory(trajectory)
    flat, offsets = diagrams._flatten()

    column = _wasserstein_column(flat, offsets, diagrams.diagrams[0])
    assert column.shape == (4,)
    assert column[0] == 0.0


def test_wasserstein_column_empty_inputs() -> None:
    """``_wasserstein_column`` returns shape ``(0,)`` for an empty trajectory
    list, and accepts an empty ``diagram_b``."""
    flat_empty, offsets_empty = _flatten([])
    nonempty = np.array([[0.0, 1.0]], dtype=np.float64)
    assert _wasserstein_column(flat_empty, offsets_empty, nonempty).shape == (0,)

    flat_mix, offsets_mix = _flatten([np.zeros((0, 2)), np.array([[0.0, 1.0]], dtype=np.float64)])
    column = _wasserstein_column(flat_mix, offsets_mix, np.zeros((0, 2), dtype=np.float64))
    assert column.shape == (2,)
