"""Tests for shadowing event pathfinding (2D version)."""

import numpy as np
import pytest

from ks_shadowing import RPO
from ks_shadowing.pha.pathfinding import (
    _CLOSE_PASS_DTYPE_2D,
    _collect_close_passes_2d,
    _ComponentPathFinder2D,
    _extract_shadowing_events_2d,
    _find_connected_components_2d,
)
from ks_shadowing.pha.persistence import _RPOPersistence


def make_passes_2d(*entries: tuple[int, int, float]) -> np.ndarray:
    """Helper to create structured array of 2D close passes."""
    passes = np.empty(len(entries), dtype=_CLOSE_PASS_DTYPE_2D)
    for i, (t, p, d) in enumerate(entries):
        passes[i] = (t, p, d)
    return passes


def make_mock_rpo_persistence(rpo_index: int, period: int) -> _RPOPersistence:
    """Create a mock _RPOPersistence for testing pathfinding logic."""
    mock_rpo = RPO(
        index=rpo_index,
        fourier_coeffs=np.zeros(30),
        period=1.0,
        time_steps=period,
        spatial_shift=0.0,
    )
    mock_diagrams = [np.array([]).reshape(0, 2) for _ in range(period)]
    return _RPOPersistence(rpo=mock_rpo, diagrams=mock_diagrams)


class TestCollectClosePasses2D:
    def test_collects_correct_entries(self):
        """Collects entries below threshold with correct indices."""
        dist_matrix = np.full((3, 4), 10.0)
        dist_matrix[1, 2] = 0.5
        dist_matrix[0, 0] = 0.3

        passes = _collect_close_passes_2d(dist_matrix, threshold=1.0)

        assert len(passes) == 2
        assert all(passes["distance"] < 1.0)

        # Check specific entry
        match = passes[passes["timestep"] == 1]
        assert len(match) == 1
        assert match[0]["phase"] == 2
        assert match[0]["distance"] == pytest.approx(0.5)

    def test_empty_when_all_above_threshold(self):
        """Returns empty array when no entries below threshold."""
        dist_matrix = np.full((3, 4), 10.0)
        passes = _collect_close_passes_2d(dist_matrix, threshold=1.0)
        assert len(passes) == 0


class TestFindConnectedComponents2D:
    def test_connectivity_types(self):
        """Tests diagonal, horizontal, vertical, and phase wraparound connectivity."""
        # Diagonal
        diagonal = make_passes_2d((0, 0, 0.5), (1, 1, 0.5), (2, 2, 0.5))
        assert len(_find_connected_components_2d(diagonal, 10, 10)) == 1

        # Horizontal (same timestep, adjacent phase)
        horizontal = make_passes_2d((5, 3, 0.5), (5, 4, 0.5))
        assert len(_find_connected_components_2d(horizontal, 10, 10)) == 1

        # Vertical (adjacent timestep, same phase)
        vertical = make_passes_2d((5, 3, 0.5), (6, 3, 0.5))
        assert len(_find_connected_components_2d(vertical, 10, 10)) == 1

        # Phase wraparound
        wraparound = make_passes_2d((0, 0, 0.5), (0, 9, 0.5))
        assert len(_find_connected_components_2d(wraparound, 10, 10)) == 1

    def test_disjoint_entries_separate_components(self):
        """Non-adjacent entries are in separate components."""
        passes = make_passes_2d((0, 0, 0.5), (10, 5, 0.5))
        components = _find_connected_components_2d(passes, 10, 20)
        assert len(components) == 2


class TestComponentPathFinder2D:
    def test_simple_diagonal_path(self):
        """Finds path where both timestep and phase advance by 1."""
        passes = make_passes_2d(
            (0, 0, 0.5),
            (1, 1, 0.4),
            (2, 2, 0.3),
        )
        finder = _ComponentPathFinder2D(passes, period=10)
        result = finder.find_longest_path()
        assert result is not None

        path, mean_dist, min_dist = result
        assert len(path) == 3
        assert min_dist == pytest.approx(0.3)
        assert mean_dist == pytest.approx((0.5 + 0.4 + 0.3) / 3)

    def test_phase_wraparound_in_path(self):
        """Path continues across phase boundary."""
        passes = make_passes_2d(
            (0, 8, 0.5),
            (1, 9, 0.4),
            (2, 0, 0.3),
        )
        finder = _ComponentPathFinder2D(passes, period=10)
        result = finder.find_longest_path()

        assert result is not None
        path, _, _ = result
        assert len(path) == 3

    def test_no_valid_path(self):
        """When phases don't match expected +1 pattern, each point is its own path."""
        passes = make_passes_2d(
            (0, 0, 0.5),
            (1, 0, 0.4),
            (2, 0, 0.3),
        )
        finder = _ComponentPathFinder2D(passes, period=10)
        result = finder.find_longest_path()

        assert result is not None
        path, _, _ = result
        assert len(path) == 1

    def test_multiple_branches_chooses_longest(self):
        """When multiple paths exist, chooses the longest."""
        passes = make_passes_2d(
            (0, 0, 0.5),
            (1, 1, 0.5),
            (2, 2, 0.5),
            (3, 3, 0.5),  # Length 4
            (0, 5, 0.1),
            (1, 6, 0.1),  # Length 2
        )
        finder = _ComponentPathFinder2D(passes, period=10)
        result = finder.find_longest_path()

        assert result is not None
        path, _, _ = result
        assert len(path) == 4

    def test_ties_broken_by_mean_distance(self):
        """Equal length paths are broken by mean distance."""
        passes = make_passes_2d(
            (0, 0, 0.9),
            (1, 1, 0.9),  # Mean 0.9
            (10, 5, 0.1),
            (11, 6, 0.1),  # Mean 0.1
        )
        finder = _ComponentPathFinder2D(passes, period=10)
        result = finder.find_longest_path()

        assert result is not None
        path, mean_dist, _ = result
        assert len(path) == 2
        assert mean_dist == pytest.approx(0.1)


class TestExtractShadowingEvents2D:
    def test_simple_diagonal_event(self):
        """Detects a diagonal path with correct statistics and zero shifts."""
        dist_matrix = np.full((5, 5), 10.0)
        dist_matrix[0, 0] = 0.2
        dist_matrix[1, 1] = 0.8
        dist_matrix[2, 2] = 0.4

        rpo_data = make_mock_rpo_persistence(rpo_index=42, period=5)
        events = _extract_shadowing_events_2d(dist_matrix, rpo_data, threshold=1.0, min_duration=1)
        assert len(events) == 1

        event = events[0]
        assert event.start_timestep == 0
        assert event.end_timestep == 3
        assert event.start_phase == 0
        assert event.rpo_index == 42
        assert event.mean_distance == pytest.approx((0.2 + 0.8 + 0.4) / 3)
        assert event.min_distance == pytest.approx(0.2)
        assert len(event.shifts) == 3
        np.testing.assert_array_equal(event.shifts, [0, 0, 0])

    def test_empty_when_all_above_threshold(self):
        """Returns no events when all distances above threshold."""
        dist_matrix = np.full((10, 5), 10.0)
        rpo_data = make_mock_rpo_persistence(rpo_index=0, period=5)
        events = _extract_shadowing_events_2d(dist_matrix, rpo_data, threshold=1.0, min_duration=1)
        assert events == []

    def test_min_duration_filter(self):
        """Events shorter than min_duration are excluded."""
        dist_matrix = np.full((5, 5), 10.0)
        dist_matrix[0, 0] = 0.5
        dist_matrix[1, 1] = 0.5

        rpo_data = make_mock_rpo_persistence(rpo_index=0, period=5)

        assert (
            len(_extract_shadowing_events_2d(dist_matrix, rpo_data, threshold=1.0, min_duration=3))
            == 0
        )
        assert (
            len(_extract_shadowing_events_2d(dist_matrix, rpo_data, threshold=1.0, min_duration=2))
            == 1
        )

    def test_multiple_disjoint_events(self):
        """Separate close regions yield separate events."""
        dist_matrix = np.full((10, 5), 10.0)
        dist_matrix[0, 0] = 0.5
        dist_matrix[1, 1] = 0.5
        dist_matrix[2, 2] = 0.5
        dist_matrix[7, 0] = 0.5
        dist_matrix[8, 1] = 0.5
        dist_matrix[9, 2] = 0.5

        rpo_data = make_mock_rpo_persistence(rpo_index=0, period=5)
        events = _extract_shadowing_events_2d(dist_matrix, rpo_data, threshold=1.0, min_duration=1)

        assert len(events) == 2
        assert events[0].start_timestep == 0
        assert events[1].start_timestep == 7

    def test_phase_wraparound_event(self):
        """Event path wraps around phase boundary."""
        dist_matrix = np.full((4, 5), 10.0)
        dist_matrix[0, 3] = 0.5
        dist_matrix[1, 4] = 0.5
        dist_matrix[2, 0] = 0.5
        dist_matrix[3, 1] = 0.5

        rpo_data = make_mock_rpo_persistence(rpo_index=0, period=5)
        events = _extract_shadowing_events_2d(dist_matrix, rpo_data, threshold=1.0, min_duration=1)

        assert len(events) == 1
        assert events[0].end_timestep - events[0].start_timestep == 4
