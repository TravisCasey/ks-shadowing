"""Tests for the 3D SSA pathfinding pipeline."""

import numpy as np

from ks_shadowing.ssa.pathfinding import (
    _CLOSE_PASS_DTYPE,
    _extract_shadowing_events,
    _find_connected_components,
)


def _gen(*phase_distances: np.ndarray):
    """Yield ``(phase, chunk_start, dist_sq)`` from per-phase distance arrays.

    Each element of ``phase_distances`` becomes one phase; entries are
    squared internally to match the format expected by
    ``_extract_shadowing_events``.
    """

    def generate():
        for phase, distances in enumerate(phase_distances):
            yield phase, 0, distances**2

    return generate()


def test_26_connected_with_wraparounds() -> None:
    """Close passes adjacent only via phase or shift wraparound are grouped
    into a single component, producing one event after longest-path
    extraction."""
    resolution = 6
    dist_p0 = np.full((1, resolution), 10.0)
    dist_p0[0, 2] = 0.5
    dist_p3 = np.full((1, resolution), 10.0)
    dist_p3[0, 2] = 0.5
    filler = np.full((1, resolution), 10.0)

    events = _extract_shadowing_events(
        _gen(dist_p0, filler, filler, dist_p3),
        rpo_index=0,
        period=4,
        resolution=resolution,
        threshold=1.0,
        min_duration=1,
    )
    assert len(events) == 1

    dist_shift = np.full((1, resolution), 10.0)
    dist_shift[0, 0] = 0.5
    dist_shift[0, resolution - 1] = 0.5
    events = _extract_shadowing_events(
        _gen(dist_shift),
        rpo_index=0,
        period=1,
        resolution=resolution,
        threshold=1.0,
        min_duration=1,
    )
    assert len(events) == 1


def test_constant_shift_path_and_statistics() -> None:
    """A 3-step run at constant shift produces an event with the expected
    ``rpo_index``, range, ``start_phase``, ``shifts``, ``mean_distance``,
    and ``min_distance``."""
    resolution = 4
    distances = np.array(
        [[10.0, 0.2, 10.0, 10.0], [10.0, 0.8, 10.0, 10.0], [10.0, 0.4, 10.0, 10.0]],
        dtype=np.float64,
    )
    events = _extract_shadowing_events(
        _gen(distances),
        rpo_index=7,
        period=3,
        resolution=resolution,
        threshold=1.0,
        min_duration=1,
    )
    assert len(events) == 1
    event = events[0]
    assert event.rpo_index == 7
    assert event.start_timestep == 0
    assert event.end_timestep == 3
    assert event.start_phase == 0
    np.testing.assert_array_equal(event.shifts, [1, 1, 1])
    assert event.mean_distance == (0.2 + 0.8 + 0.4) / 3
    assert event.min_distance == 0.2


def test_shift_drift_and_break() -> None:
    """A shift sequence advancing by ``+1`` per timestep is a valid path; a
    ``+3`` jump breaks path continuity into shorter events."""
    resolution = 8
    valid = np.full((3, resolution), 10.0)
    valid[0, 2] = valid[1, 3] = valid[2, 4] = 0.5
    events = _extract_shadowing_events(
        _gen(valid),
        rpo_index=0,
        period=1,
        resolution=resolution,
        threshold=1.0,
        min_duration=1,
    )
    assert len(events) == 1
    np.testing.assert_array_equal(events[0].shifts, [2, 3, 4])

    broken = np.full((3, resolution), 10.0)
    broken[0, 2] = broken[1, 5] = broken[2, 6] = 0.5
    events = _extract_shadowing_events(
        _gen(broken),
        rpo_index=0,
        period=1,
        resolution=resolution,
        threshold=1.0,
        min_duration=1,
    )
    assert all(event.end_timestep - event.start_timestep < 3 for event in events)


def test_min_duration_filter() -> None:
    """Events shorter than ``min_duration`` are excluded."""
    distances = np.array([[0.5], [0.5]], dtype=np.float64)
    assert (
        _extract_shadowing_events(
            _gen(distances),
            rpo_index=0,
            period=1,
            resolution=1,
            threshold=1.0,
            min_duration=3,
        )
        == []
    )
    events = _extract_shadowing_events(
        _gen(distances),
        rpo_index=0,
        period=1,
        resolution=1,
        threshold=1.0,
        min_duration=2,
    )
    assert len(events) == 1


def test_diagonal_neighbors_form_one_component() -> None:
    """Close passes at ``(timestep=0, rpo_phase=0, shift=0)`` and
    ``(1, 1, 0)`` are adjacent (each coordinate differs by at most 1) and
    group into one component."""
    close_passes = np.array(
        [(0, 0, 0, 0.1), (1, 1, 0, 0.1)],
        dtype=_CLOSE_PASS_DTYPE,
    )

    components = _find_connected_components(close_passes, period=10, resolution=8)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_phase_jump_of_two_separates_components() -> None:
    """Close passes at ``(timestep=0, rpo_phase=0, shift=0)`` and
    ``(1, 2, 0)`` differ by 2 in ``rpo_phase`` and form two separate
    components."""
    close_passes = np.array(
        [(0, 0, 0, 0.1), (1, 2, 0, 0.1)],
        dtype=_CLOSE_PASS_DTYPE,
    )

    components = _find_connected_components(close_passes, period=10, resolution=8)
    assert len(components) == 2
