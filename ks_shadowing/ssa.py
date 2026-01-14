"""State Space Approach (SSA) for shadowing detection.

The SSA detects shadowing by computing L2 distances in physical space between
trajectory snapshots and RPO phases with compatible spatial shifting.
"""

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import DEFAULT_RESOLUTION
from .detection import ShadowingEvent, extract_shadowing_events
from .integrator import ksint
from .rpo import RPO
from .transforms import interleaved_to_complex, l2_distance_all_shifts, to_physical


def compute_distances_to_rpo(
    trajectory_physical: NDArray[np.float64],
    rpo_physical: NDArray[np.float64],
) -> Iterator[NDArray[np.float64]]:
    """Yield distance arrays for each trajectory timestep against one RPO.

    Each yielded array has shape `(period, spatial_resolution)` containing the
    L2 distance from the trajectory snapshot to each RPO phase at each spatial
    shift.

    Args:
        trajectory_physical: Shape `(trajectory_length, spatial_resolution)`
            trajectory in physical space.
        rpo_physical: Shape `(period, resolution)` RPO trajectory in physical
            space.

    Yields:
        `(period, spatial_resolution)` distance arrays, one per trajectory
            timestep.
    """
    for snapshot in trajectory_physical:
        yield l2_distance_all_shifts(snapshot, rpo_physical)


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
        self.dt = dt
        self.resolution = resolution
        self.rpos = list(rpos)

        # Precompute each RPO's trajectory in physical space
        self.rpo_trajectories: list[NDArray[np.float64]] = []
        for rpo in rpos:
            fourier_traj: NDArray[np.float64] = ksint(rpo.fourier_coeffs, dt, rpo.time_steps)[:-1]
            complex_coeffs: NDArray[np.complex128] = interleaved_to_complex(fourier_traj)
            self.rpo_trajectories.append(to_physical(complex_coeffs, resolution))

    def detect(
        self,
        trajectory_fourier: NDArray[np.floating],
        threshold: float,
        min_duration: int = 1,
    ) -> list[ShadowingEvent]:
        """Detect shadowing events in trajectory for all RPOs.

        Uses streaming DP to find contiguous paths in the 3D distance space
        where the trajectory shadows an RPO, enforcing both phase advancement
        and spatial continuity constraints.
        """
        complex_coeffs: NDArray[np.complex128] = interleaved_to_complex(trajectory_fourier)
        trajectory_physical: NDArray[np.float64] = to_physical(complex_coeffs, self.resolution)

        all_events: list[ShadowingEvent] = []

        for rpo_index, traj in enumerate(self.rpo_trajectories):
            period: int = traj.shape[0]
            events = extract_shadowing_events(
                compute_distances_to_rpo(trajectory_physical, traj),
                rpo_index,
                period,
                threshold,
                min_duration,
            )
            all_events.extend(events)

        all_events.sort(key=lambda e: (e.start_time, e.rpo_index))
        return all_events

    def compute_min_distances(
        self, trajectory_fourier: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Compute minimum distance to any RPO at each trajectory timestep.

        Returns shape `(trajectory_length,)` where each entry is the minimum
        over all RPOs, phases, and shifts. Useful for threshold selection.
        """
        complex_coeffs: NDArray[np.complex128] = interleaved_to_complex(trajectory_fourier)
        trajectory_physical: NDArray[np.float64] = to_physical(complex_coeffs, self.resolution)

        min_dists: NDArray[np.float64] = np.full(len(trajectory_physical), np.inf, dtype=np.float64)

        for rpo_traj in self.rpo_trajectories:
            for t, distances in enumerate(compute_distances_to_rpo(trajectory_physical, rpo_traj)):
                min_dists[t] = min(min_dists[t], np.min(distances))

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
        min_distances: NDArray[np.float64] = self.compute_min_distances(trajectory_fourier)
        threshold: float = float(np.quantile(min_distances, f_close))
        events = self.detect(trajectory_fourier, threshold, min_duration)
        return events, threshold
