"""Tests for shift reconstruction for PHA events."""

import numpy as np

from ks_shadowing import RPO
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.pha.shifts import (
    _compute_event_shifts,
    _find_optimal_shifts,
)


class TestFindOptimalShifts:
    def test_single_timestep(self):
        """Single timestep returns the minimum distance shift."""
        distances = np.array([[0.5, 0.1, 0.9, 0.3]])
        shifts = _find_optimal_shifts(distances, resolution=4)

        assert len(shifts) == 1
        assert shifts[0] == 1

    def test_shift_continuity_constraint(self):
        """Optimal shifts change by at most 1 between consecutive timesteps."""
        rng = np.random.default_rng(42)
        distances = rng.random((20, 16))
        shifts = _find_optimal_shifts(distances, resolution=16)

        assert len(shifts) == 20
        for i in range(1, len(shifts)):
            diff = (shifts[i] - shifts[i - 1]) % 16
            assert diff in (0, 1, 15)  # 15 is -1 mod 16

    def test_prefers_low_distance_path(self):
        """When multiple valid paths exist, chooses lowest total distance."""
        distances = np.full((4, 10), 10.0)
        # Path through shift 0: 2.0 + 0.1 + 0.1 + 0.1 = 2.3
        distances[0, 0] = 2.0
        distances[1, 0] = 0.1
        distances[2, 0] = 0.1
        distances[3, 0] = 0.1
        # Path through shift 5: 0.1 + 2.0 + 2.0 + 2.0 = 6.1
        distances[0, 5] = 0.1
        distances[1, 5] = 2.0
        distances[2, 5] = 2.0
        distances[3, 5] = 2.0

        shifts = _find_optimal_shifts(distances, resolution=10)
        # Should choose the path through shift 0 despite higher initial cost
        assert np.all(shifts == np.zeros(4, dtype=np.int32))

    def test_wraparound_in_shifts(self):
        """Shift sequence can wrap around from resolution-1 to 0."""
        distances = np.full((3, 8), 10.0)
        distances[0, 7] = 0.1
        distances[1, 0] = 0.1
        distances[2, 1] = 0.1

        shifts = _find_optimal_shifts(distances, resolution=8)
        assert list(shifts) == [7, 0, 1]


class TestComputeEventShifts:
    def test_computes_valid_shifts(self):
        """Computes shifts with correct continuity, dtype, and preserves other fields."""
        rpo = RPO(
            index=5,
            fourier_coeffs=np.zeros(30),
            period=1.0,
            time_steps=10,
            spatial_shift=0.0,
        )
        resolution = 64

        original_shifts = np.array([1, 2, 3, 4], dtype=np.int32)
        event = ShadowingEvent(
            rpo_index=5,
            start_timestep=2,
            end_timestep=6,
            mean_distance=0.15,
            min_distance=0.08,
            start_phase=3,
            shifts=original_shifts,
        )

        trajectory_fourier = np.zeros((20, 30))
        result = _compute_event_shifts(event, trajectory_fourier, rpo, resolution)

        # Check shifts are valid
        assert len(result.shifts) == 4
        assert result.shifts.dtype == np.int32
        for i in range(1, len(result.shifts)):
            diff = (result.shifts[i] - result.shifts[i - 1]) % resolution
            assert diff in (0, 1, resolution - 1)

        # Check other fields preserved
        assert result.rpo_index == 5
        assert result.start_timestep == 2
        assert result.end_timestep == 6
        assert result.mean_distance == 0.15
        assert result.min_distance == 0.08
        assert result.start_phase == 3

        # Check original unchanged (frozen dataclass uses replace)
        np.testing.assert_array_equal(event.shifts, [1, 2, 3, 4])
