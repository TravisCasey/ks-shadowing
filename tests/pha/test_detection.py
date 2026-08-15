"""End-to-end tests for the public PHA detection API."""

import numpy as np
import pytest
from numpy.typing import NDArray

from ks_shadowing import pha
from ks_shadowing.core import INTEGRATION_DT
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha.detection import _apply_delay_embedding, _center_events


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
    ).events
    events_b = pha.detect(
        short_trajectory,
        small_rpos,
        delay=2,
        threshold=threshold,
        min_duration=10,
        n_jobs=1,
    ).events

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

    threshold = pha.auto_detect(
        short_trajectory,
        small_rpos,
        delay=delay,
        threshold_quantile=quantile,
        min_duration=10,
        n_jobs=1,
    ).threshold
    assert threshold == pytest.approx(expected_threshold)


def test_detect_native_mode(
    sample_initial_state: NDArray[np.complex128], small_rpos: list[RPO]
) -> None:
    """``pha.detect`` runs end-to-end at ``native=True`` with an inferred
    downsample of 2 and returns events with valid bounds, ``shifts`` shape,
    and ``shifts`` dtype, and events relabeled to the centered window limits
    at ``delay=3``."""
    trajectory = KSTrajectory.from_initial_state(
        sample_initial_state,
        dt=INTEGRATION_DT,
        num_timesteps=100,
        resolution=32,
        save_interval=2,
    )

    events = pha.detect(
        trajectory,
        small_rpos[:1],
        delay=3,
        threshold=1e6,
        native=True,
        min_duration=1,
        n_jobs=1,
    ).events

    assert events
    for event in events:
        assert 0 <= event.start_timestep < event.end_timestep <= len(trajectory)
        assert event.shifts.shape == (event.end_timestep - event.start_timestep,)
        assert event.shifts.dtype == np.int32
    assert min(event.start_timestep for event in events) == 1
    assert max(event.end_timestep for event in events) == len(trajectory) - 1


def test_derivatives_affects_min_distances(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """``compute_min_distances`` at ``max_derivative_order=1`` differs from
    ``max_derivative_order=0`` on a generic trajectory, confirming the
    kwarg is wired through the pipeline."""
    base = pha.compute_min_distances(short_trajectory, small_rpos, n_jobs=1)
    enriched = pha.compute_min_distances(
        short_trajectory, small_rpos, max_derivative_order=1, n_jobs=1
    )
    assert base.shape == enriched.shape
    # Different derivative orders produce different Wasserstein scores
    # generically; equality would mean the kwarg is being ignored.
    finite_mask = np.isfinite(base) & np.isfinite(enriched)
    assert finite_mask.any()
    assert not np.allclose(base[finite_mask], enriched[finite_mask])


def test_apply_delay_embedding_explicit() -> None:
    """``_apply_delay_embedding(matrix, delay=2)`` averages entries along
    diagonals ``(t + l, (j + l) mod J)`` for ``l in range(delay)``."""
    matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
    expected = np.array(
        [[2.0, 3.0, 2.5], [5.0, 6.0, 5.5], [8.0, 9.0, 8.5]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(_apply_delay_embedding(matrix, delay=2), expected)


def test_apply_delay_embedding_invalid_delay_raises() -> None:
    """``_apply_delay_embedding`` raises ``ValueError`` when ``delay`` is
    less than 1 or exceeds the trajectory length."""
    matrix = np.zeros((10, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        _apply_delay_embedding(matrix, delay=0)
    with pytest.raises(ValueError):
        _apply_delay_embedding(matrix, delay=11)


def test_center_events_shifts_timesteps_and_phase() -> None:
    """``_center_events`` shifts ``start_timestep``/``end_timestep`` by the
    centering offset and ``start_phase`` by the same offset modulo ``period``,
    flooring the offset for even ``delay``, and returns events unchanged at
    ``delay=1``."""
    event = ShadowingEvent(
        rpo_index=3,
        start_timestep=10,
        end_timestep=20,
        mean_distance=0.5,
        min_distance=0.1,
        start_phase=48,
        shifts=np.zeros(10, dtype=np.int32),
    )

    (odd,) = _center_events([event], delay=5, period=50)
    assert odd.start_timestep == 12
    assert odd.end_timestep == 22
    assert odd.start_phase == 0  # 48 + 2 wraps modulo 50
    assert odd.rpo_index == 3
    assert odd.mean_distance == 0.5

    (even,) = _center_events([event], delay=8, period=50)
    assert even.start_timestep == 13  # offset (8 - 1) // 2 == 3
    assert even.start_phase == 1

    (identity,) = _center_events([event], delay=1, period=50)
    assert (identity.start_timestep, identity.end_timestep, identity.start_phase) == (10, 20, 48)


def test_min_distances_infinite_at_window_edges(
    short_trajectory: KSTrajectory, small_rpos: list[RPO]
) -> None:
    """``compute_min_distances`` returns infinity for the leading
    ``(delay - 1) // 2`` and trailing ``delay // 2`` timesteps and finite
    values everywhere between."""
    delay = 4  # even window: leading edge 1, trailing edge 2
    min_distances = pha.compute_min_distances(short_trajectory, small_rpos, delay=delay, n_jobs=1)
    leading = (delay - 1) // 2
    trailing = delay // 2

    assert min_distances.shape == (len(short_trajectory),)
    assert np.all(np.isinf(min_distances[:leading]))
    assert np.all(np.isinf(min_distances[len(short_trajectory) - trailing :]))
    assert np.all(np.isfinite(min_distances[leading : len(short_trajectory) - trailing]))
