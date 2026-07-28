"""Tests for the 2D PHA pathfinding pipeline."""

import numpy as np
import pytest
from numpy.typing import NDArray

from ks_shadowing.pha.pathfinding import _extract_shadowing_events


def _diagonal_matrix(
    num_timesteps: int, period: int, distances: list[float]
) -> NDArray[np.float64]:
    """Return an ``(num_timesteps, period)`` matrix filled with 10.0 except
    for a diagonal of ``distances`` starting at ``(0, 0)``."""
    matrix = np.full((num_timesteps, period), 10.0, dtype=np.float64)
    for offset, distance in enumerate(distances):
        matrix[offset, offset % period] = distance
    return matrix


def test_eight_connected_with_phase_wraparound() -> None:
    """Close passes that span the ``phase=0`` boundary form a single event;
    duration matches the path length."""
    period = 5
    matrix = np.full((4, period), 10.0, dtype=np.float64)
    matrix[0, period - 2] = 0.5
    matrix[1, period - 1] = 0.5
    matrix[2, 0] = 0.5
    matrix[3, 1] = 0.5

    events = _extract_shadowing_events(matrix, rpo_index=0, threshold=1.0, min_duration=1)
    assert len(events) == 1
    assert events[0].end_timestep - events[0].start_timestep == 4


def test_diagonal_event_statistics() -> None:
    """A diagonal close-pass run produces an event with ``rpo_index``,
    ``start_timestep``, ``end_timestep``, ``start_phase``, ``mean_distance``,
    ``min_distance``, and a zero-filled ``shifts`` array of the correct
    length."""
    matrix = _diagonal_matrix(num_timesteps=5, period=5, distances=[0.2, 0.8, 0.4])

    events = _extract_shadowing_events(matrix, rpo_index=42, threshold=1.0, min_duration=1)
    assert len(events) == 1
    event = events[0]
    assert event.rpo_index == 42
    assert event.start_timestep == 0
    assert event.end_timestep == 3
    assert event.start_phase == 0
    assert event.mean_distance == pytest.approx((0.2 + 0.8 + 0.4) / 3)
    assert event.min_distance == 0.2
    assert len(event.shifts) == 3
    np.testing.assert_array_equal(event.shifts, np.zeros(3, dtype=np.int32))


def test_min_duration_and_disjoint_events() -> None:
    """Disjoint close-pass regions yield separate events; ``min_duration``
    excludes shorter ones."""
    matrix = np.full((10, 5), 10.0, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = 0.5
    matrix[7, 0] = matrix[8, 1] = 0.5

    events_all = _extract_shadowing_events(matrix, rpo_index=0, threshold=1.0, min_duration=1)
    assert len(events_all) == 2
    assert events_all[0].start_timestep == 0
    assert events_all[1].start_timestep == 7

    events_filtered = _extract_shadowing_events(matrix, rpo_index=0, threshold=1.0, min_duration=3)
    assert len(events_filtered) == 1
    assert events_filtered[0].start_timestep == 0


def test_no_events_above_threshold() -> None:
    """A distance matrix with all entries above ``threshold`` returns an
    empty event list."""
    matrix = np.full((10, 5), 10.0, dtype=np.float64)
    events = _extract_shadowing_events(matrix, rpo_index=0, threshold=1.0, min_duration=1)
    assert events == []
