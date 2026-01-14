"""Tests for shadowing event detection utilities."""

import numpy as np
import pytest

from ks_shadowing.detection import PathBuffer, extract_shadowing_events


class TestPathBuffer:
    def test_start_path(self):
        """start_path initializes a new path correctly."""
        buf = PathBuffer.empty(2, 5)
        buf.start_path(rpo=1, phase=3, distance=2.5, time=7)

        assert buf.path_length[1, 3] == 1
        assert buf.distance_sum[1, 3] == 2.5
        assert buf.min_distance[1, 3] == 2.5
        assert buf.start_time[1, 3] == 7
        assert buf.is_active(1, 3)
        assert not buf.is_active(0, 0)

    def test_extend_from(self):
        """extend_from correctly extends a path from previous buffer."""
        prev = PathBuffer.empty(1, 5)
        prev.start_path(rpo=0, phase=2, distance=1.0, time=10)

        curr = PathBuffer.empty(1, 5)
        curr.extend_from(prev, rpo=0, phase=3, pred_phase=2, distance=0.5)

        assert curr.path_length[0, 3] == 2
        assert curr.distance_sum[0, 3] == 1.5
        assert curr.min_distance[0, 3] == 0.5
        assert curr.start_time[0, 3] == 10

    def test_to_event(self):
        """to_event creates correct ShadowingEvent."""
        buf = PathBuffer.empty(1, 5)
        buf.path_length[0, 2] = 4
        buf.distance_sum[0, 2] = 8.0
        buf.min_distance[0, 2] = 1.5
        buf.start_time[0, 2] = 100

        event = buf.to_event(0, 2)
        assert event.rpo_index == 0
        assert event.start_time == 100
        assert event.end_time == 104
        assert event.duration == 4
        assert event.mean_distance == 2.0
        assert event.min_distance == 1.5


class TestExtractShadowingEvents:
    def test_empty_generator(self):
        """Empty generator returns no events."""
        events = extract_shadowing_events(iter([]), [10], threshold=1.0)
        assert events == []

    def test_empty_rpo_list(self):
        """Empty RPO list returns no events."""
        events = extract_shadowing_events(iter([np.array([[]])]), [], threshold=1.0)
        assert events == []

    def test_all_above_threshold(self):
        """No events when all distances exceed threshold."""

        def gen():
            for _ in range(5):
                yield np.array([[10.0, 10.0, 10.0]])

        events = extract_shadowing_events(gen(), [3], threshold=1.0)
        assert events == []

    def test_simple_contiguous_path(self):
        """Detects a simple contiguous shadowing path."""
        # Single RPO with period 3, trajectory of 5 timesteps
        # Path: phase 0 -> 1 -> 2 -> 0 -> 1 (wraps around)

        def gen():
            # t=0: phase 0 below threshold
            yield np.array([[0.5, 10.0, 10.0]])
            # t=1: phase 1 below threshold
            yield np.array([[10.0, 0.5, 10.0]])
            # t=2: phase 2 below threshold
            yield np.array([[10.0, 10.0, 0.5]])
            # t=3: phase 0 below threshold (wraparound)
            yield np.array([[0.5, 10.0, 10.0]])
            # t=4: phase 1 below threshold
            yield np.array([[10.0, 0.5, 10.0]])

        events = extract_shadowing_events(gen(), [3], threshold=1.0)
        assert len(events) == 1
        assert events[0].start_time == 0
        assert events[0].end_time == 5
        assert events[0].duration == 5
        assert events[0].rpo_index == 0

    def test_min_duration_filter(self):
        """Short paths are filtered by min_duration."""

        def gen():
            yield np.array([[0.5]])
            yield np.array([[0.5]])
            yield np.array([[10.0]])  # break

        events = extract_shadowing_events(gen(), [1], threshold=1.0, min_duration=3)
        assert len(events) == 0

        events = extract_shadowing_events(
            gen(),
            [1],
            threshold=1.0,
            min_duration=2,
        )
        assert len(events) == 1

    def test_multiple_rpos(self):
        """Handles multiple RPOs with different periods."""

        def gen():
            # RPO 0: period 2, RPO 1: period 3
            # Only RPO 1 has a valid path
            yield np.array([[10.0, 10.0, np.inf], [0.5, 10.0, 10.0]])
            yield np.array([[10.0, 10.0, np.inf], [10.0, 0.5, 10.0]])
            yield np.array([[10.0, 10.0, np.inf], [10.0, 10.0, 0.5]])

        events = extract_shadowing_events(gen(), [2, 3], threshold=1.0)
        assert len(events) == 1
        assert events[0].rpo_index == 1
        assert events[0].duration == 3

    def test_event_statistics(self):
        """Mean and min distance are computed correctly."""

        def gen():
            yield np.array([[0.2]])
            yield np.array([[0.8]])
            yield np.array([[0.4]])

        events = extract_shadowing_events(gen(), [1], threshold=1.0)
        assert len(events) == 1
        assert events[0].mean_distance == pytest.approx((0.2 + 0.8 + 0.4) / 3)
        assert events[0].min_distance == pytest.approx(0.2)

    def test_multiple_disjoint_events(self):
        """Detects multiple separate shadowing events."""

        def gen():
            yield np.array([[0.5]])  # t=0: event 1 start
            yield np.array([[0.5]])  # t=1
            yield np.array([[10.0]])  # t=2: break
            yield np.array([[10.0]])  # t=3: break
            yield np.array([[0.5]])  # t=4: event 2 start
            yield np.array([[0.5]])  # t=5

        events = extract_shadowing_events(gen(), [1], threshold=1.0)
        assert len(events) == 2
        assert events[0].start_time == 0
        assert events[0].end_time == 2
        assert events[1].start_time == 4
        assert events[1].end_time == 6
