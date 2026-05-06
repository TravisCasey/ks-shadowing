"""End-to-end tests for the public PHA detection API."""

import numpy as np
import pytest

from ks_shadowing import pha
from ks_shadowing.core import TRAJECTORY_DT
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


@pytest.fixture
def short_trajectory(small_rpos: list[RPO]) -> KSTrajectory:
    """200-timestep trajectory at resolution 16 seeded from the shortest RPO."""
    rpo = small_rpos[0]
    return KSTrajectory.from_initial_state(
        rpo.modes, dt=TRAJECTORY_DT, num_timesteps=200, resolution=16
    )


def test_detect_deterministic_and_sorted(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """``pha.detect`` is deterministic across calls, returns events sorted by
    ``(start_timestep, rpo_index)``, and populates ``shifts`` with length
    equal to the event duration."""
    threshold = 0.5
    events_a = pha.detect(
        short_trajectory,
        small_rpos,
        delay=2,
        threshold=threshold,
        min_duration=10,
        n_jobs=1,
    )
    events_b = pha.detect(
        short_trajectory,
        small_rpos,
        delay=2,
        threshold=threshold,
        min_duration=10,
        n_jobs=1,
    )

    assert len(events_a) == len(events_b)
    for left, right in zip(events_a, events_b, strict=True):
        assert left.rpo_index == right.rpo_index
        assert left.start_timestep == right.start_timestep
        assert left.end_timestep == right.end_timestep
        np.testing.assert_array_equal(left.shifts, right.shifts)

    sort_keys = [(event.start_timestep, event.rpo_index) for event in events_a]
    assert sort_keys == sorted(sort_keys)

    for event in events_a:
        assert len(event.shifts) == event.end_timestep - event.start_timestep


def test_auto_detect_threshold_matches_quantile(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """The threshold returned by ``pha.auto_detect`` equals
    ``np.quantile(pha.compute_min_distances(...), threshold_quantile)``."""
    quantile = 0.4
    delay = 2
    min_distances = pha.compute_min_distances(short_trajectory, small_rpos, delay=delay, n_jobs=1)
    finite = min_distances[np.isfinite(min_distances)]
    expected_threshold = float(np.quantile(finite, quantile))

    _, threshold = pha.auto_detect(
        short_trajectory,
        small_rpos,
        delay=delay,
        threshold_quantile=quantile,
        min_duration=10,
        n_jobs=1,
    )
    assert threshold == pytest.approx(expected_threshold)
