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
    """Partially overlapping ``[0, 10)`` and ``[5, 15)`` produce a single match
    with composite lengths 10, intersection 5, and union 15."""
    matches = match_events([_event(0, 0, 10)], [_event(0, 5, 15)])
    assert len(matches) == 1
    match = matches[0]
    assert match.ssa_length == 10
    assert match.pha_length == 10
    assert match.intersection_length == 5
    assert match.union_length == 15


def test_grouping_isolation_and_gap_arithmetic() -> None:
    """Overlapping events on one RPO group into a single match, events on other
    RPOs form separate matches sorted by RPO, and a gap in one side's coverage
    counts toward the union but not the intersection."""
    ssa = [_event(0, 0, 30), _event(1, 0, 10)]
    pha = [_event(0, 5, 10), _event(0, 20, 25), _event(2, 0, 10), _event(1, 2, 8)]

    matches = match_events(ssa, pha)
    assert len(matches) == 2
    first, second = matches

    assert [id(event) for event in first.ssa_events] == [id(ssa[0])]
    assert [id(event) for event in first.pha_events] == [id(pha[0]), id(pha[1])]
    assert first.ssa_length == 30
    assert first.pha_length == 10
    assert first.intersection_length == 10
    assert first.union_length == 30

    assert [id(event) for event in second.ssa_events] == [id(ssa[1])]
    assert [id(event) for event in second.pha_events] == [id(pha[3])]
    assert second.intersection_length == 6
    assert second.union_length == 10


def test_transitive_chaining() -> None:
    """A PHA event overlapping two SSA events chains all overlapping events on
    the RPO into a single match with composite arithmetic over both sides."""
    ssa = [_event(0, 0, 10), _event(0, 20, 30)]
    pha = [_event(0, 0, 5), _event(0, 8, 22)]

    matches = match_events(ssa, pha)
    assert len(matches) == 1
    match = matches[0]
    assert [id(event) for event in match.ssa_events] == [id(ssa[0]), id(ssa[1])]
    assert [id(event) for event in match.pha_events] == [id(pha[0]), id(pha[1])]
    assert match.ssa_length == 20
    assert match.pha_length == 19
    assert match.intersection_length == 9
    assert match.union_length == 30
