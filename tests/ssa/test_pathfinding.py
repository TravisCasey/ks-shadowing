"""Tests for shadowing event pathfinding (3D version)."""

import numpy as np
import pytest

from ks_shadowing import RPO
from ks_shadowing.ssa.pathfinding import (
    _CLOSE_PASS_DTYPE_3D,
    _collect_close_passes_3d,
    _extract_shadowing_events_3d,
    _find_connected_components_3d,
)
from ks_shadowing.ssa.rpo import _RPOStateSpace


def make_dist_sq_generator(*distance_arrays):
    """Create a squared distance generator from distance arrays.

    Each array becomes one phase. Arrays contain distances (not squared);
    they are squared internally to match the expected generator format.
    """

    def generator():
        for phase, distances in enumerate(distance_arrays):
            yield phase, distances**2

    return generator


def make_mock_rpo_state_space(rpo_index: int, period: int, resolution: int) -> _RPOStateSpace:
    """Create a mock _RPOStateSpace for testing pathfinding logic."""
    mock_rpo = RPO(
        index=rpo_index,
        fourier_coeffs=np.zeros(30),
        period=1.0,
        time_steps=period,
        spatial_shift=0.0,
    )
    mock_trajectory = np.zeros((period, resolution))
    return _RPOStateSpace(rpo=mock_rpo, trajectory=mock_trajectory)


def make_passes(*entries: tuple[int, int, int, float]) -> np.ndarray:
    """Helper to create structured array of close passes."""
    passes = np.empty(len(entries), dtype=_CLOSE_PASS_DTYPE_3D)
    for i, (t, p, s, d) in enumerate(entries):
        passes[i] = (t, p, s, d)
    return passes


class TestCollectClosePasses3D:
    def test_collects_below_threshold(self):
        """Collects entries below threshold from squared distance generator."""
        # Squared distances for two phases, 3 timesteps, 4 shifts
        phase0_sq = np.array([[0.25, 2.25, 0.09, 4.0], [4.0, 4.0, 4.0, 4.0], [0.01, 4.0, 4.0, 4.0]])
        phase1_sq = np.array([[1.0, 0.64, 1.44, 0.16], [4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0]])

        def gen():
            yield 0, phase0_sq
            yield 1, phase1_sq

        passes = _collect_close_passes_3d(gen(), threshold=1.0)
        assert len(passes) == 5
        assert all(passes["timestep"][i] != 1 for i in range(len(passes)))

    def test_empty_when_all_above_threshold(self):
        """Returns empty array when no entries below threshold."""

        def gen():
            yield 0, np.full((2, 4), 100.0)  # All squared distances > 1.0^2

        passes = _collect_close_passes_3d(gen(), threshold=1.0)
        assert len(passes) == 0


class TestFindConnectedComponents3D:
    def test_adjacent_entries_same_component(self):
        """Adjacent entries are grouped together."""
        passes = make_passes(
            (0, 0, 0, 0.5),
            (1, 1, 0, 0.5),
            (2, 2, 1, 0.5),
        )
        components = _find_connected_components_3d(passes, period=10, resolution=10)
        assert len(components) == 1

    def test_disjoint_entries_separate_components(self):
        """Non-adjacent entries are in separate components."""
        passes = make_passes(
            (0, 0, 0, 0.5),
            (10, 5, 5, 0.5),
        )
        components = _find_connected_components_3d(passes, period=10, resolution=10)
        assert len(components) == 2

    def test_wraparound_connectivity(self):
        """Phase and shift wraparound connects boundary elements."""
        # Phase wraparound
        passes = make_passes(
            (0, 0, 0, 0.5),
            (1, 9, 0, 0.5),
        )
        assert len(_find_connected_components_3d(passes, period=10, resolution=10)) == 1

        # Shift wraparound
        passes = make_passes(
            (0, 0, 0, 0.5),
            (0, 0, 9, 0.5),
        )
        assert len(_find_connected_components_3d(passes, period=10, resolution=10)) == 1


class TestExtractShadowingEvents3D:
    def test_empty_generator(self):
        """Empty generator returns no events."""
        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=10, resolution=8)
        events = _extract_shadowing_events_3d(iter([]), rpo_data, threshold=1.0)
        assert events == []

    def test_simple_path(self):
        """Detects a path with constant shift across timesteps."""
        distances = np.full((3, 4), 10.0)
        distances[0, 1] = distances[1, 1] = distances[2, 1] = 0.5

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=3, resolution=4)
        generator = make_dist_sq_generator(distances, np.full((3, 4), 10.0), np.full((3, 4), 10.0))
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0)

        assert len(events) == 1
        assert events[0].end_timestep - events[0].start_timestep == 3
        assert all(shift == 1 for shift in events[0].shifts)

    def test_shift_drift(self):
        """Detects path where shift increases by 1 each timestep."""
        distances = np.full((3, 8), 10.0)
        distances[0, 2] = distances[1, 3] = distances[2, 4] = 0.5

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=3, resolution=8)
        generator = make_dist_sq_generator(distances, np.full((3, 8), 10.0), np.full((3, 8), 10.0))
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0)

        assert len(events) == 1
        assert list(events[0].shifts) == [2, 3, 4]

    def test_large_shift_jump_breaks_path(self):
        """Shift change > 1 breaks path continuity."""
        distances = np.full((3, 8), 10.0)
        distances[0, 2] = distances[1, 5] = distances[2, 6] = 0.5  # Jump from 2 to 5

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=3, resolution=8)
        generator = make_dist_sq_generator(distances, np.full((3, 8), 10.0), np.full((3, 8), 10.0))
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0)

        assert all(event.end_timestep - event.start_timestep < 3 for event in events)

    def test_shift_wraparound(self):
        """Shift wraps from resolution-1 to 0."""
        distances = np.full((2, 8), 10.0)
        distances[0, 7] = distances[1, 0] = 0.5

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=2, resolution=8)
        generator = make_dist_sq_generator(distances, np.full((2, 8), 10.0))
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0)

        assert len(events) == 1
        assert events[0].end_timestep - events[0].start_timestep == 2

    def test_min_duration_filter(self):
        """Events shorter than min_duration are excluded."""
        distances = np.array([[0.5], [0.5]])

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=1, resolution=1)
        generator = make_dist_sq_generator(distances)
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0, min_duration=3)
        assert len(events) == 0

        generator = make_dist_sq_generator(distances)
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0, min_duration=2)
        assert len(events) == 1

    def test_event_statistics(self):
        """Mean and min distance are computed correctly from path distances."""
        distances = np.array([[0.2], [0.8], [0.4]])

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=1, resolution=1)
        generator = make_dist_sq_generator(distances)
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0)

        assert len(events) == 1
        assert events[0].mean_distance == pytest.approx((0.2 + 0.8 + 0.4) / 3)
        assert events[0].min_distance == pytest.approx(0.2)

    def test_multiple_disjoint_events(self):
        """Separate close regions yield separate events."""
        distances = np.array([[0.5], [0.5], [10.0], [10.0], [0.5], [0.5]])

        rpo_data = make_mock_rpo_state_space(rpo_index=0, period=1, resolution=1)
        generator = make_dist_sq_generator(distances)
        events = _extract_shadowing_events_3d(generator(), rpo_data, threshold=1.0)

        assert len(events) == 2
        assert events[0].start_timestep == 0
        assert events[1].start_timestep == 4
