"""Regression tests for the public plotting helpers.

The lean-test policy applies (see CLAUDE.md). Covers
``events_to_union_mask`` boundary semantics and the lab-frame
alignment produced by ``align_rpo_to_window``.
"""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.results import load_results
from ks_shadowing.core.rpo import load_rpos
from ks_shadowing.plotting import (
    align_rpo_to_window,
    events_to_union_mask,
    select_event_by_rank,
)

FIXTURE_SSA = Path(__file__).resolve().parent.parent / "examples/data/ssa_r2048.h5"
RPO_FILE = Path(__file__).resolve().parent.parent / "data/rpos_selected.npz"


def _event(start: int, end: int) -> ShadowingEvent:
    return ShadowingEvent(
        rpo_index=0,
        start_timestep=start,
        end_timestep=end,
        mean_distance=0.0,
        min_distance=0.0,
        start_phase=0,
        shifts=np.zeros(end - start, dtype=np.int32),
    )


def test_events_to_union_mask_unions_overlapping_intervals() -> None:
    """The mask is True for any timestep covered by at least one event,
    with ``end_timestep`` exclusive (so a timestep at exactly
    ``end_timestep`` is not covered)."""
    events = [_event(2, 5), _event(4, 8)]
    mask = events_to_union_mask(events, num_timesteps=10)
    expected = np.array([False, False, True, True, True, True, True, True, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_events_to_union_mask_empty_list() -> None:
    """Empty event list produces an all-False mask of the requested length."""
    mask = events_to_union_mask([], num_timesteps=5)
    assert mask.shape == (5,)
    assert not mask.any()


def test_align_rpo_matches_trajectory_at_best_event() -> None:
    """At the lowest-mean-distance event in the SSA fixture, each row of
    the aligned RPO panel must be close (in L2 per row) to the
    corresponding trajectory row. The 3 * mean_distance bound is a loose
    sanity check that catches sign errors and native-phase mapping bugs;
    any correctly aligned panel falls well under it.
    """
    if not FIXTURE_SSA.exists():
        pytest.skip(f"SSA fixture not present: {FIXTURE_SSA}")

    _, trajectory, events = load_results(FIXTURE_SSA)
    assert events, "fixture must contain at least one event"
    event = select_event_by_rank(events)

    rpos = load_rpos(RPO_FILE)
    rpo = rpos[event.rpo_index]

    aligned = align_rpo_to_window(rpo, event, event.start_timestep, event.end_timestep, trajectory)

    trajectory_slice = trajectory[event.start_timestep : event.end_timestep]
    trajectory_physical = trajectory_slice.to_physical()

    per_row_l2 = np.linalg.norm(aligned - trajectory_physical, axis=1)
    mean_l2 = float(per_row_l2.mean())

    assert mean_l2 <= 3.0 * event.mean_distance, (
        f"mean per-row L2 {mean_l2:.4f} exceeds 3 * mean_distance "
        f"({event.mean_distance:.4f}); alignment is likely off (sign error "
        f"or wrong native-phase mapping)."
    )
