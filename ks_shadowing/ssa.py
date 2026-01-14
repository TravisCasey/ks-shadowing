"""State Space Approach (SSA) for shadowing detection.

The SSA detects shadowing by computing L2 distances in physical space between
trajectory snapshots and RPO phases, optimizing over all spatial shifts using
FFT-based cross-correlation.
"""

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import DEFAULT_RESOLUTION
from .detection import ShadowingEvent, extract_shadowing_events
from .integrator import ksint
from .rpo import RPO
from .transforms import interleaved_to_complex, min_distance_over_shifts, to_physical


def state_space_distances(
    trajectory_physical: NDArray[np.float64],
    rpo_trajectories: list[NDArray[np.float64]],
) -> Iterator[NDArray[np.float64]]:
    """Yield distance arrays for timesteps of the trajectory against the RPOs

    Each yielded array is of the shape `(rpo_count, max_period)`, where
    `max_period` is the maximum period of all the RPOs. Any entries for a given
    RPO that exceeds the period of that RPO is filled with infinity.

    The yielded arrays (one for each timestep of `trajectory_physical`) contain
    the minimum L2 distance over all spatial shifts of the trajectory snapshot
    against a phase of the RPO: one entry for each phase of each RPO.
    """
    rpo_count = len(rpo_trajectories)
    if rpo_count != 0:
        max_period = max(traj.shape[0] for traj in rpo_trajectories)

    for snapshot in trajectory_physical:
        distances = np.full((rpo_count, max_period), np.inf, dtype=np.float64)
        for rpo_index, traj in enumerate(rpo_trajectories):
            phase_distances = min_distance_over_shifts(snapshot, traj)
            distances[rpo_index, : traj.shape[0]] = phase_distances

        yield distances


class SSADetector:
    """State Space Approach shadowing detector.

    Precomputes RPO trajectories in physical space and compares each against
    chaotic trajectories to detect shadowing events in state space.
    """

    def __init__(
        self,
        rpos: Sequence[RPO],
        dt: float,
        resolution: int = DEFAULT_RESOLUTION,
    ):
        """Initialize detector with precomputed RPO trajectories."""
        self.dt: float = dt
        self.resolution: int = resolution

        # Precompute each RPO's trajectory in physical space
        self.rpo_trajectories: list[NDArray[np.float64]] = []
        for rpo in rpos:
            # Exclude endpoint of RPO trajectory as it is periodic
            fourier_traj: NDArray[np.float64] = ksint(rpo.fourier_coeffs, dt, rpo.time_steps)[:-1]
            complex_coeffs: NDArray[np.complex128] = interleaved_to_complex(fourier_traj)
            self.rpo_trajectories.append(to_physical(complex_coeffs, resolution))

    @property
    def rpo_periods(self) -> list[int]:
        """List of period lengths (number of phases) for each RPO."""
        return [traj.shape[0] for traj in self.rpo_trajectories]

    def detect(
        self,
        trajectory_fourier: NDArray[np.floating],
        threshold: float,
        min_duration: int = 1,
    ) -> list[ShadowingEvent]:
        """Detect shadowing events in trajectory.

        Uses streaming DP to find contiguous time intervals where the trajectory
        shadows an RPO with distance below the given threshold.
        """
        complex_coeffs = interleaved_to_complex(trajectory_fourier)
        trajectory_physical = to_physical(complex_coeffs, self.resolution)

        return extract_shadowing_events(
            state_space_distances(trajectory_physical, self.rpo_trajectories),
            self.rpo_periods,
            threshold,
            min_duration,
        )

    def compute_min_distances(
        self, trajectory_fourier: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Compute minimum distance to any RPO at each trajectory timestep.

        Returns shape `(M,)` where each entry is the minimum over all RPOs,
        phases, and shifts. Useful for distance threshold selection in detect
        methods.
        """
        complex_coeffs = interleaved_to_complex(trajectory_fourier)
        trajectory_physical = to_physical(complex_coeffs, self.resolution)

        min_dists = np.empty(len(trajectory_physical), dtype=np.float64)
        for timestep, distances in enumerate(
            state_space_distances(trajectory_physical, self.rpo_trajectories)
        ):
            min_dists[timestep] = np.min(distances)

        return min_dists

    def auto_detect(
        self,
        trajectory_fourier: NDArray[np.floating],
        f_close: float = 0.4,
        min_duration: int = 1,
    ) -> tuple[list[ShadowingEvent], float]:
        """Detect shadowing events with automatic threshold selection.

        Computes the threshold as the `f_close` quantile of minimum distances,
        then detects events. Returns both the events and the computed threshold.
        """
        min_distances = self.compute_min_distances(trajectory_fourier)
        threshold = float(np.quantile(min_distances, f_close))
        events = self.detect(trajectory_fourier, threshold, min_duration)
        return events, threshold

    def state_space_distance_array(
        self, trajectory_fourier: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Return full 3D distance array for debugging or other analysis.

        Returns shape `(len, rpo_count, max_period)` where `len` is the
        trajectory length. Warning: This can be very, very large for long
        trajectories. Prefer the `state_space_distances` generator.
        """
        complex_coeffs = interleaved_to_complex(trajectory_fourier)
        trajectory_physical = to_physical(complex_coeffs, self.resolution)

        return np.array(list(state_space_distances(trajectory_physical, self.rpo_trajectories)))
