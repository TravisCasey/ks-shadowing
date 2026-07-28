"""End-to-end tests for the public SSA detection API."""

import numpy as np
import pytest

from ks_shadowing import ssa
from ks_shadowing.core import INTEGRATION_DT
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


@pytest.fixture
def short_trajectory(small_rpos: list[RPO]) -> KSTrajectory:
    """200-timestep trajectory at resolution 32 seeded from the shortest RPO."""
    rpo = small_rpos[0]
    return KSTrajectory.from_initial_state(
        rpo.modes, dt=INTEGRATION_DT, num_timesteps=200, resolution=32
    )


def test_detect_deterministic_and_sorted(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """``ssa.detect`` is deterministic across calls and returns events sorted
    by ``(start_timestep, rpo_index)``."""
    threshold = 5.0
    events_a = ssa.detect(
        short_trajectory, small_rpos, threshold=threshold, min_duration=10, n_jobs=1
    ).events
    events_b = ssa.detect(
        short_trajectory, small_rpos, threshold=threshold, min_duration=10, n_jobs=1
    ).events

    assert len(events_a) == len(events_b)
    for left, right in zip(events_a, events_b, strict=True):
        assert left.rpo_index == right.rpo_index
        assert left.start_timestep == right.start_timestep
        assert left.end_timestep == right.end_timestep

    sort_keys = [(event.start_timestep, event.rpo_index) for event in events_a]
    assert sort_keys == sorted(sort_keys)


def test_auto_detect_threshold_matches_quantile(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """The threshold returned by ``ssa.auto_detect`` equals
    ``np.quantile(ssa.compute_min_distances(...), threshold_quantile)``."""
    quantile = 0.4
    min_distances = ssa.compute_min_distances(short_trajectory, small_rpos, n_jobs=1)
    expected_threshold = float(np.quantile(min_distances, quantile))

    threshold = ssa.auto_detect(
        short_trajectory,
        small_rpos,
        threshold_quantile=quantile,
        min_duration=10,
        n_jobs=1,
    ).threshold
    assert threshold == pytest.approx(expected_threshold)


def test_detect_native_mode(sample_initial_state: np.ndarray, small_rpos: list[RPO]) -> None:
    """``ssa.detect`` runs end-to-end at ``native=True`` with an inferred
    downsample of 2 and returns events with valid bounds, ``shifts`` shape,
    and ``shifts`` dtype."""
    trajectory = KSTrajectory.from_initial_state(
        sample_initial_state,
        dt=INTEGRATION_DT,
        num_timesteps=50,
        resolution=32,
        save_interval=2,
    )

    events = ssa.detect(
        trajectory,
        small_rpos[:1],
        threshold=1e6,
        native=True,
        min_duration=1,
        n_jobs=1,
    ).events

    for event in events:
        assert 0 <= event.start_timestep < event.end_timestep <= len(trajectory)
        assert event.shifts.shape == (event.end_timestep - event.start_timestep,)
        assert event.shifts.dtype == np.int32
