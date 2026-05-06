"""Tests for matched event computation."""

import numpy as np

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.matching import match_events


def _event(rpo_index: int, start: int, end: int) -> ShadowingEvent:
    """Minimal ShadowingEvent for matching tests."""
    return ShadowingEvent(
        rpo_index=rpo_index,
        start_timestep=start,
        end_timestep=end,
        mean_distance=0.1,
        min_distance=0.1,
        start_phase=0,
        shifts=np.zeros(end - start, dtype=np.int32),
    )


def test_adjacent_ranges_dont_overlap() -> None:
    """Events with abutting half-open ranges ``[0, 5)`` and ``[5, 10)`` do
    not match."""
    assert match_events([_event(0, 0, 5)], [_event(0, 5, 10)]) == []


def test_partial_overlap_arithmetic() -> None:
    """Partially overlapping ``[0, 10)`` and ``[5, 15)`` produce a single
    match with ``intersection_length=5`` and ``union_length=15``."""
    matches = match_events([_event(0, 0, 10)], [_event(0, 5, 15)])
    assert len(matches) == 1
    assert matches[0].intersection_length == 5
    assert matches[0].union_length == 15


def test_one_to_many_within_rpo_and_isolated_across_rpos() -> None:
    """One SSA event matches multiple PHA events on the same RPO; events on
    different RPOs do not match each other."""
    ssa = [_event(0, 0, 30), _event(1, 0, 10)]
    pha = [_event(0, 5, 10), _event(0, 20, 25), _event(2, 0, 10)]

    matches = match_events(ssa, pha)
    assert len(matches) == 2
    assert all(match.ssa_event is ssa[0] for match in matches)
    matched_pha_ids = {id(match.pha_event) for match in matches}
    assert matched_pha_ids == {id(pha[0]), id(pha[1])}
