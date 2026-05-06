"""Tests for PHA shift reconstruction."""

from itertools import pairwise

import numpy as np

from ks_shadowing.pha.shifts import _find_optimal_shifts


def test_dp_continuity_constraint(rng: np.random.Generator) -> None:
    """Returned shifts ``s[t]`` satisfy ``(s[t+1] - s[t]) mod resolution
    in {0, 1, resolution - 1}``."""
    resolution = 16
    distances = rng.random((20, resolution))
    shifts = _find_optimal_shifts(distances, resolution=resolution)
    assert len(shifts) == 20
    for previous, current in pairwise(shifts):
        diff = (int(current) - int(previous)) % resolution
        assert diff in (0, 1, resolution - 1)


def test_dp_prefers_low_distance_path() -> None:
    """``_find_optimal_shifts`` selects the globally minimum-cost path even
    when the locally cheapest first step is on a different shift."""
    resolution = 10
    distances = np.full((4, resolution), 10.0, dtype=np.float64)
    distances[0, 0] = 2.0
    distances[1, 0] = 0.1
    distances[2, 0] = 0.1
    distances[3, 0] = 0.1
    distances[0, 5] = 0.1
    distances[1, 5] = 2.0
    distances[2, 5] = 2.0
    distances[3, 5] = 2.0

    shifts = _find_optimal_shifts(distances, resolution=resolution)
    np.testing.assert_array_equal(shifts, np.zeros(4, dtype=np.int32))


def test_dp_wraparound() -> None:
    """The shift sequence wraps from ``resolution - 1`` to 0 under the
    continuity constraint."""
    resolution = 8
    distances = np.full((3, resolution), 10.0, dtype=np.float64)
    distances[0, resolution - 1] = 0.1
    distances[1, 0] = 0.1
    distances[2, 1] = 0.1

    shifts = _find_optimal_shifts(distances, resolution=resolution)
    np.testing.assert_array_equal(shifts, [resolution - 1, 0, 1])
