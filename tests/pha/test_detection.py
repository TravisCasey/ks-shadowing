"""End-to-end tests for the public PHA detection API."""

import numpy as np
import pytest

from ks_shadowing import pha
from ks_shadowing.core import INTEGRATION_DT
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


@pytest.fixture
def short_trajectory(small_rpos: list[RPO]) -> KSTrajectory:
    """200-timestep trajectory at resolution 16 seeded from the shortest RPO."""
    rpo = small_rpos[0]
    return KSTrajectory.from_initial_state(
        rpo.modes, dt=INTEGRATION_DT, num_timesteps=200, resolution=16
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


def test_detect_native_mode(sample_initial_state: np.ndarray, small_rpos: list[RPO]) -> None:
    """``pha.detect`` runs end-to-end at ``native=True, downsample=2`` and
    returns events with valid bounds, ``shifts`` shape, and ``shifts`` dtype."""
    trajectory = KSTrajectory.from_initial_state(
        sample_initial_state, dt=INTEGRATION_DT, num_timesteps=100, resolution=16
    )

    events = pha.detect(
        trajectory,
        small_rpos[:1],
        delay=2,
        threshold=1e6,
        downsample=2,
        native=True,
        min_duration=1,
        n_jobs=1,
    )

    for event in events:
        assert 0 <= event.start_timestep < event.end_timestep <= len(trajectory)
        assert event.shifts.shape == (event.end_timestep - event.start_timestep,)
        assert event.shifts.dtype == np.int32


def test_derivatives_affects_min_distances(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """``compute_min_distances`` at ``derivatives=2`` differs from
    ``derivatives=1`` on a generic trajectory, confirming the derivatives
    kwarg is wired through the pipeline."""
    base = pha.compute_min_distances(short_trajectory, small_rpos, n_jobs=1)
    enriched = pha.compute_min_distances(short_trajectory, small_rpos, derivatives=2, n_jobs=1)
    assert base.shape == enriched.shape
    # Different derivative orders produce different Wasserstein scores
    # generically; equality would mean the kwarg is being ignored.
    finite_mask = np.isfinite(base) & np.isfinite(enriched)
    assert finite_mask.any()
    assert not np.allclose(base[finite_mask], enriched[finite_mask])
