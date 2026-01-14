"""Tests for shadowing event detection."""

import numpy as np
import pytest

from ks_shadowing.detection import ShadowingEvent, extract_shadowing_events


class TestShadowingEvent:
    def test_duration(self):
        """Duration is end_time - start_time."""
        event = ShadowingEvent(
            rpo_index=0,
            start_time=10,
            end_time=15,
            mean_distance=1.0,
            min_distance=0.5,
        )
        assert event.duration == 5


class TestExtractShadowingEvents:
    def test_empty_generator(self):
        """Empty generator returns no events."""
        events = extract_shadowing_events(iter([]), rpo_index=0, period=10, threshold=1.0)
        assert events == []

    def test_all_above_threshold(self):
        """No events when all distances exceed threshold."""

        def gen():
            for _ in range(5):
                yield np.full((3, 8), 10.0)

        events = extract_shadowing_events(gen(), rpo_index=0, period=3, threshold=1.0)
        assert events == []

    def test_simple_path_same_shift(self):
        """Detects path where shift stays constant."""

        # Period=2, 4 shifts, path at shift=1
        def gen():
            arr = np.full((2, 4), 10.0)
            arr[0, 1] = 0.5  # t=0, phase=0, shift=1
            yield arr.copy()
            arr = np.full((2, 4), 10.0)
            arr[1, 1] = 0.5  # t=1, phase=1, shift=1
            yield arr.copy()
            arr = np.full((2, 4), 10.0)
            arr[0, 1] = 0.5  # t=2, phase=0, shift=1 (wraparound)
            yield arr.copy()

        events = extract_shadowing_events(gen(), rpo_index=0, period=2, threshold=1.0)
        assert len(events) == 1
        assert events[0].duration == 3

    def test_path_with_shift_drift(self):
        """Detects path where shift drifts by 1 each step."""

        # Period=3, 8 shifts
        def gen():
            arr = np.full((3, 8), 10.0)
            arr[0, 2] = 0.5  # t=0, phase=0, shift=2
            yield arr.copy()
            arr = np.full((3, 8), 10.0)
            arr[1, 3] = 0.5  # t=1, phase=1, shift=3 (drift +1)
            yield arr.copy()
            arr = np.full((3, 8), 10.0)
            arr[2, 4] = 0.5  # t=2, phase=2, shift=4 (drift +1)
            yield arr.copy()

        events = extract_shadowing_events(gen(), rpo_index=0, period=3, threshold=1.0)
        assert len(events) == 1
        assert events[0].duration == 3

    def test_shift_jump_breaks_path(self):
        """Large shift jump (>1) breaks the path."""

        def gen():
            arr = np.full((3, 8), 10.0)
            arr[0, 2] = 0.5  # t=0, phase=0, shift=2
            yield arr.copy()
            arr = np.full((3, 8), 10.0)
            arr[1, 5] = 0.5  # t=1, phase=1, shift=5 (jump of 3 - too far!)
            yield arr.copy()
            arr = np.full((3, 8), 10.0)
            arr[2, 6] = 0.5  # t=2, phase=2, shift=6
            yield arr.copy()

        events = extract_shadowing_events(gen(), rpo_index=0, period=3, threshold=1.0)
        # Should get separate short events, not one continuous path
        assert all(e.duration < 3 for e in events)

    def test_shift_wraparound(self):
        """Shift wraparound works (shift 0 connects to shift K-1)."""

        def gen():
            arr = np.full((2, 8), 10.0)
            arr[0, 7] = 0.5  # t=0, phase=0, shift=7
            yield arr.copy()
            arr = np.full((2, 8), 10.0)
            arr[1, 0] = 0.5  # t=1, phase=1, shift=0 (wraparound from 7)
            yield arr.copy()

        events = extract_shadowing_events(gen(), rpo_index=0, period=2, threshold=1.0)
        assert len(events) == 1
        assert events[0].duration == 2

    def test_phase_wraparound(self):
        """Phase wraparound works at period boundary."""

        # Period=2, path goes phase 0 -> 1 -> 0 -> 1
        def gen():
            for t in range(4):
                arr = np.full((2, 4), 10.0)
                arr[t % 2, 1] = 0.5
                yield arr

        events = extract_shadowing_events(gen(), rpo_index=0, period=2, threshold=1.0)
        assert len(events) == 1
        assert events[0].duration == 4

    def test_min_duration_filter(self):
        """Short paths are filtered by min_duration."""

        def gen():
            yield np.array([[0.5, 10.0]])
            yield np.array([[10.0, 0.5]])
            yield np.full((1, 2), 10.0)

        events = extract_shadowing_events(
            gen(), rpo_index=0, period=1, threshold=1.0, min_duration=3
        )
        assert len(events) == 0

        events = extract_shadowing_events(
            gen(), rpo_index=0, period=1, threshold=1.0, min_duration=2
        )
        assert len(events) == 1

    def test_event_statistics(self):
        """Mean and min distance are computed correctly."""

        def gen():
            yield np.array([[0.2, 10.0]])
            yield np.array([[0.8, 10.0]])
            yield np.array([[0.4, 10.0]])

        events = extract_shadowing_events(gen(), rpo_index=0, period=1, threshold=1.0)
        assert len(events) == 1
        assert events[0].mean_distance == pytest.approx((0.2 + 0.8 + 0.4) / 3)
        assert events[0].min_distance == pytest.approx(0.2)

    def test_multiple_disjoint_events(self):
        """Detects multiple separate shadowing events."""

        def gen():
            yield np.array([[0.5, 10.0]])  # t=0: event 1
            yield np.array([[0.5, 10.0]])  # t=1
            yield np.full((1, 2), 10.0)  # t=2: break
            yield np.full((1, 2), 10.0)  # t=3: break
            yield np.array([[0.5, 10.0]])  # t=4: event 2
            yield np.array([[0.5, 10.0]])  # t=5

        events = extract_shadowing_events(gen(), rpo_index=0, period=1, threshold=1.0)
        assert len(events) == 2
        assert events[0].start_time == 0
        assert events[0].end_time == 2
        assert events[1].start_time == 4
        assert events[1].end_time == 6
