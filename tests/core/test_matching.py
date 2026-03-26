"""Tests for matched event computation."""

import numpy as np

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.matching import find_matched_events


def _make_event(rpo_index: int, start: int, end: int, mean_distance: float = 0.1) -> ShadowingEvent:
    """Build a minimal ShadowingEvent for matching tests."""
    return ShadowingEvent(
        rpo_index=rpo_index,
        start_timestep=start,
        end_timestep=end,
        mean_distance=mean_distance,
        min_distance=mean_distance,
        start_phase=0,
        shifts=np.zeros(end - start, dtype=np.int32),
    )


class TestFindMatchedEvents:
    """Tests for find_matched_events."""

    def test_empty_ssa_events(self) -> None:
        """No matches when SSA event list is empty."""
        pha = [_make_event(0, 10, 20)]
        assert find_matched_events([], pha) == []

    def test_empty_pha_events(self) -> None:
        """No matches when PHA event list is empty."""
        ssa = [_make_event(0, 10, 20)]
        assert find_matched_events(ssa, []) == []

    def test_different_rpos_no_match(self) -> None:
        """Events on different RPOs do not match."""
        ssa = [_make_event(0, 10, 20)]
        pha = [_make_event(1, 10, 20)]
        assert find_matched_events(ssa, pha) == []

    def test_adjacent_no_overlap(self) -> None:
        """Adjacent ranges [0,5) and [5,10) do not overlap."""
        ssa = [_make_event(0, 0, 5)]
        pha = [_make_event(0, 5, 10)]
        assert find_matched_events(ssa, pha) == []

    def test_exact_overlap(self) -> None:
        """Identical ranges produce one match with IoU = 1."""
        ssa = [_make_event(0, 10, 20)]
        pha = [_make_event(0, 10, 20)]
        matches = find_matched_events(ssa, pha)
        assert len(matches) == 1
        assert matches[0].intersection_length == 10
        assert matches[0].union_length == 10

    def test_partial_overlap(self) -> None:
        """Partially overlapping ranges have correct intersection and union."""
        ssa = [_make_event(0, 0, 10)]
        pha = [_make_event(0, 5, 15)]
        matches = find_matched_events(ssa, pha)
        assert len(matches) == 1
        assert matches[0].intersection_length == 5
        assert matches[0].union_length == 15

    def test_one_to_many(self) -> None:
        """One SSA event can match multiple PHA events."""
        ssa = [_make_event(0, 0, 30)]
        pha = [_make_event(0, 5, 10), _make_event(0, 20, 25)]
        matches = find_matched_events(ssa, pha)
        assert len(matches) == 2
        assert all(m.ssa_event is ssa[0] for m in matches)

    def test_many_to_one(self) -> None:
        """Multiple SSA events can match one PHA event."""
        ssa = [_make_event(0, 0, 5), _make_event(0, 10, 15)]
        pha = [_make_event(0, 0, 15)]
        matches = find_matched_events(ssa, pha)
        assert len(matches) == 2
        assert all(m.pha_event is pha[0] for m in matches)

    def test_multiple_rpos(self) -> None:
        """Events across multiple RPOs are matched independently."""
        ssa = [_make_event(0, 0, 10), _make_event(1, 0, 10)]
        pha = [_make_event(0, 5, 15), _make_event(2, 0, 10)]
        matches = find_matched_events(ssa, pha)
        assert len(matches) == 1
        assert matches[0].ssa_event.rpo_index == 0
        assert matches[0].pha_event.rpo_index == 0
