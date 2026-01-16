"""State Space Approach (SSA) shadowing detector."""

from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray
from scipy import fft
from tqdm import tqdm

from ks_shadowing.core import DOMAIN_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO, RPOTrajectory
from ks_shadowing.core.transforms import (
    interleaved_to_physical,
    tile_periodic,
    to_comoving_frame,
)
from ks_shadowing.core.util import resolve_n_jobs
from ks_shadowing.ssa.pathfinding import extract_shadowing_events


def compute_distances_sq(
    trajectory_physical: NDArray[np.float64],
    rpo_data: RPOTrajectory,
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Yield squared distance arrays for each phase offset using co-moving frame.

    Both trajectory and RPO are transformed to the RPO's co-moving frame where
    the RPO becomes truly periodic. Yields `(phase, dist_sq)` tuples where
    `dist_sq` has shape `(num_timesteps, resolution)`.

    Uses FFT cross-correlation to compute distances to all shifts in O(N log N).
    Squared distances are returned to allow thresholding without sqrt overhead.
    """
    num_timesteps = trajectory_physical.shape[0]
    period = rpo_data.time_steps
    resolution = rpo_data.resolution

    # Drift rate in grid cells per timestep
    drift_per_step = (rpo_data.spatial_shift / DOMAIN_SIZE) * resolution / period

    # Transform both to co-moving frame
    trajectory_comoving = to_comoving_frame(trajectory_physical, drift_per_step)
    rpo_comoving = to_comoving_frame(rpo_data.trajectory, drift_per_step)

    # Tile RPO to cover trajectory length plus phase offsets
    rpo_tiled = tile_periodic(rpo_comoving, num_timesteps + period - 1)

    # Precompute FFTs and squared norms
    traj_fft = fft.rfft(trajectory_comoving, axis=-1)
    rpo_fft = fft.rfft(rpo_tiled, axis=-1)
    norm_traj_sq = np.sum(trajectory_comoving**2, axis=-1)
    norm_rpo_sq = np.sum(rpo_tiled**2, axis=-1)

    # Yield squared distances for each phase offset
    for phase in range(period):
        rpo_slice_fft = rpo_fft[phase : phase + num_timesteps]
        rpo_slice_norm_sq = norm_rpo_sq[phase : phase + num_timesteps]

        cross_corr = fft.irfft(np.conj(traj_fft) * rpo_slice_fft, resolution, axis=-1)
        dist_sq = norm_traj_sq[:, np.newaxis] + rpo_slice_norm_sq[:, np.newaxis] - 2 * cross_corr
        yield phase, np.maximum(dist_sq, 0.0)


# Module-level worker functions for multiprocessing (must be pickleable)


def _detect_single_rpo(
    rpo_data: RPOTrajectory,
    trajectory_physical: NDArray[np.float64],
    threshold: float,
    min_duration: int,
) -> list[ShadowingEvent]:
    """Worker function for parallel detection."""
    return extract_shadowing_events(
        compute_distances_sq(trajectory_physical, rpo_data),
        rpo_data,
        threshold,
        min_duration,
    )


def _min_dist_single_rpo(
    rpo_data: RPOTrajectory,
    trajectory_physical: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Worker function for parallel min distance computation."""
    min_dists_sq = np.full(len(trajectory_physical), np.inf, dtype=np.float64)
    for _, dist_sq in compute_distances_sq(trajectory_physical, rpo_data):
        # Min over shift dimension, keeping timestep dimension
        phase_min_sq = np.min(dist_sq, axis=1)
        np.minimum(min_dists_sq, phase_min_sq, out=min_dists_sq)
    return np.sqrt(min_dists_sq)


class SSADetector:
    """State Space Approach shadowing detector.

    Detects shadowing events by computing L2 distances between a trajectory
    and RPO phases in the RPO's co-moving reference frame, then finding
    connected components of close passes with valid paths through them.

    The co-moving frame transformation is essential because RPOs are *relative*
    periodic orbits: they drift spatially over each period. In the co-moving
    frame, this drift is removed and the RPO becomes truly periodic.
    """

    def __init__(
        self,
        rpos: Sequence[RPO],
        dt: float,
        resolution: int,
    ):
        """Initialize detector with precomputed RPO trajectories.

        Each RPO is integrated using its native timestep to preserve numerical
        accuracy. The resulting trajectories are stored in physical space.

        Args:
            rpos: Sequence of RPO objects to detect shadowing against.
            dt: Timestep of trajectories that will be analyzed.
            resolution: Spatial resolution for physical space representation.
        """
        self.dt = dt
        self.resolution = resolution
        self.rpos = list(rpos)
        self.rpo_data_list = [RPOTrajectory.from_rpo(rpo, resolution) for rpo in rpos]

    def detect(
        self,
        trajectory_fourier: NDArray[np.floating],
        threshold: float,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> list[ShadowingEvent]:
        """Detect shadowing events for all RPOs.

        Args:
            trajectory_fourier: Trajectory in Fourier space, shape
                `(num_timesteps, 30)`.
            threshold: Maximum L2 distance for a point to be considered close.
            min_duration: Minimum event duration in timesteps.
            show_progress: Whether to display a progress bar.
            n_jobs: Number of parallel workers. Use -1 for all CPUs.

        Returns:
            List of `ShadowingEvent` sorted by `(start_timestep, rpo_index)`.
        """
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        n_workers = resolve_n_jobs(n_jobs)

        if n_workers == 1:
            events = self._detect_sequential(
                trajectory_physical, threshold, min_duration, show_progress
            )
        else:
            events = self._detect_parallel(
                trajectory_physical, threshold, min_duration, show_progress, n_workers
            )

        events.sort(key=lambda e: (e.start_timestep, e.rpo_index))
        return events

    def _detect_sequential(
        self,
        trajectory_physical: NDArray[np.float64],
        threshold: float,
        min_duration: int,
        show_progress: bool,
    ) -> list[ShadowingEvent]:
        """Run detection sequentially over all RPOs."""
        events: list[ShadowingEvent] = []
        iterator: Iterable[RPOTrajectory] = self.rpo_data_list

        if show_progress:
            iterator = tqdm(iterator, total=len(self.rpo_data_list), desc="Detecting", leave=False)

        for rpo_data in iterator:
            rpo_events = extract_shadowing_events(
                compute_distances_sq(trajectory_physical, rpo_data),
                rpo_data,
                threshold,
                min_duration,
            )
            events.extend(rpo_events)

        return events

    def _detect_parallel(
        self,
        trajectory_physical: NDArray[np.float64],
        threshold: float,
        min_duration: int,
        show_progress: bool,
        n_workers: int,
    ) -> list[ShadowingEvent]:
        """Run detection in parallel over RPOs."""
        events: list[ShadowingEvent] = []
        worker = partial(
            _detect_single_rpo,
            trajectory_physical=trajectory_physical,
            threshold=threshold,
            min_duration=min_duration,
        )

        with Pool(n_workers) as pool:
            results = pool.imap_unordered(worker, self.rpo_data_list)

            if show_progress:
                results = tqdm(
                    results, total=len(self.rpo_data_list), desc="Detecting", leave=False
                )

            for rpo_events in results:
                events.extend(rpo_events)

        return events

    def compute_min_distances(
        self,
        trajectory_fourier: NDArray[np.floating],
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> NDArray[np.float64]:
        """Compute minimum distance to any RPO at each trajectory timestep.

        For each timestep, finds the minimum L2 distance over all RPOs, all
        phase offsets, and all spatial shifts. Useful for threshold selection.

        Returns:
            Array of shape `(num_timesteps,)` with minimum distances.
        """
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        n_workers = resolve_n_jobs(n_jobs)

        if n_workers == 1:
            return self._min_distances_sequential(trajectory_physical, show_progress)
        else:
            return self._min_distances_parallel(trajectory_physical, show_progress, n_workers)

    def _min_distances_sequential(
        self,
        trajectory_physical: NDArray[np.float64],
        show_progress: bool,
    ) -> NDArray[np.float64]:
        """Compute min distances sequentially."""
        min_dists_sq = np.full(len(trajectory_physical), np.inf, dtype=np.float64)
        iterator = self.rpo_data_list

        if show_progress:
            iterator = tqdm(iterator, desc="Min distances", leave=False)

        for rpo_data in iterator:
            for _, dist_sq in compute_distances_sq(trajectory_physical, rpo_data):
                # Min over shift dimension
                phase_min_sq = np.min(dist_sq, axis=1)
                np.minimum(min_dists_sq, phase_min_sq, out=min_dists_sq)

        return np.sqrt(min_dists_sq)

    def _min_distances_parallel(
        self,
        trajectory_physical: NDArray[np.float64],
        show_progress: bool,
        n_workers: int,
    ) -> NDArray[np.float64]:
        """Compute min distances in parallel."""
        min_dists = np.full(len(trajectory_physical), np.inf, dtype=np.float64)
        worker = partial(
            _min_dist_single_rpo,
            trajectory_physical=trajectory_physical,
        )

        with Pool(n_workers) as pool:
            results = pool.imap_unordered(worker, self.rpo_data_list)

            if show_progress:
                results = tqdm(
                    results, total=len(self.rpo_data_list), desc="Min distances", leave=False
                )

            for rpo_min in results:
                np.minimum(min_dists, rpo_min, out=min_dists)

        return min_dists

    def auto_detect(
        self,
        trajectory_fourier: NDArray[np.floating],
        f_close: float = 0.4,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> tuple[list[ShadowingEvent], float]:
        """Detect shadowing events with automatic threshold selection.

        The threshold is set to the `f_close` quantile of minimum distances
        across the trajectory. For example, `f_close=0.4` means 40% of timesteps
        will have minimum distance below the threshold.

        Returns:
            Tuple of (events, threshold).
        """
        min_distances = self.compute_min_distances(
            trajectory_fourier, show_progress=show_progress, n_jobs=n_jobs
        )
        threshold = float(np.quantile(min_distances, f_close))
        events = self.detect(
            trajectory_fourier, threshold, min_duration, show_progress, n_jobs=n_jobs
        )
        return events, threshold
