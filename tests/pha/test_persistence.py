"""Tests for persistence diagram computation."""

import numpy as np
import pytest

from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha.detection import _apply_delay_embedding
from ks_shadowing.pha.persistence import (
    KSPersistenceTrajectory,
    _zeroth_persistence_diagram_periodic,
)


def test_zeroth_diagram_cos_two_minima() -> None:
    """``_zeroth_persistence_diagram_periodic`` on a discretized
    :math:`\\cos(2x)` returns one persistence pair ``(-1, 1)``."""
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    field = np.cos(2 * x)
    diagram = _zeroth_persistence_diagram_periodic(field)
    assert diagram.shape == (1, 2)
    np.testing.assert_allclose(diagram[0, 0], -1.0, atol=1e-6)
    np.testing.assert_allclose(diagram[0, 1], 1.0, atol=1e-6)


def test_zeroth_diagram_pair_count_scales() -> None:
    """A field with :math:`n` local minima produces :math:`n - 1` finite-death
    pairs (the essential class is excluded)."""
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    field = np.sin(5 * x)
    diagram = _zeroth_persistence_diagram_periodic(field)
    assert len(diagram) == 4


def test_apply_delay_embedding_explicit() -> None:
    """``_apply_delay_embedding(matrix, delay=2)`` averages entries along
    diagonals ``(t + l, (j + l) mod J)`` for ``l in range(delay)``."""
    matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
    expected = np.array(
        [[2.0, 3.0, 2.5], [5.0, 6.0, 5.5], [8.0, 9.0, 8.5]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(_apply_delay_embedding(matrix, delay=2), expected)


def test_apply_delay_embedding_invalid_delay_raises() -> None:
    """``_apply_delay_embedding`` raises ``ValueError`` when ``delay`` is
    less than 1 or exceeds the trajectory length."""
    matrix = np.zeros((10, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        _apply_delay_embedding(matrix, delay=0)
    with pytest.raises(ValueError):
        _apply_delay_embedding(matrix, delay=11)


def test_chunked_diagrams_match_unchunked(rng: np.random.Generator) -> None:
    """``KSPersistenceTrajectory.from_trajectory`` produces identical diagrams
    regardless of ``chunk_size``."""
    modes = np.zeros((50, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((50, 15)) + 1j * rng.standard_normal((50, 15))) * 0.1
    trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)

    default = KSPersistenceTrajectory.from_trajectory(trajectory)
    chunked = KSPersistenceTrajectory.from_trajectory(trajectory, chunk_size=10)

    assert len(default.diagrams) == len(chunked.diagrams)
    for a, b in zip(default.diagrams, chunked.diagrams, strict=True):
        np.testing.assert_array_equal(a, b)


def test_from_trajectory_higher_order_differs(rng: np.random.Generator) -> None:
    """``from_trajectory(..., order=1)`` produces diagrams of the derivative
    field, which differ from the order-0 diagrams on a generic trajectory."""
    modes = np.zeros((10, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((10, 15)) + 1j * rng.standard_normal((10, 15))) * 0.1
    trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)

    order0 = KSPersistenceTrajectory.from_trajectory(trajectory, order=0)
    order1 = KSPersistenceTrajectory.from_trajectory(trajectory, order=1)

    assert len(order0.diagrams) == len(order1.diagrams)
    assert any(
        a.shape != b.shape or not np.allclose(a, b)
        for a, b in zip(order0.diagrams, order1.diagrams, strict=True)
    )
