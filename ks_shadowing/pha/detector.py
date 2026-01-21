"""Persistent Homology Approach (PHA) shadowing detector."""

from collections.abc import Iterable, Sequence
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical
from ks_shadowing.core.util import resolve_n_jobs
from ks_shadowing.pha.pathfinding import extract_shadowing_events_2d
from ks_shadowing.pha.persistence import (
    RPOPersistence,
    apply_delay_embedding,
    compute_trajectory_diagrams,
    compute_wasserstein_matrix,
)


def _detect_single_rpo(
    rpo_data: RPOPersistence,
    traj_diagrams: list[NDArray[np.float64]],
    threshold: float,
    min_duration: int,
    delay: int,
) -> list[ShadowingEvent]:
    """Worker function for parallel detection."""
    # Compute Wasserstein distance matrix
    wasserstein_matrix = compute_wasserstein_matrix(traj_diagrams, rpo_data.diagrams)

    # Apply time-delay embedding
    embedded_matrix = apply_delay_embedding(wasserstein_matrix, delay)

    # Extract events
    return extract_shadowing_events_2d(
        embedded_matrix,
        rpo_data,
        threshold,
        min_duration,
        delay,
    )


def _min_dist_single_rpo(
    rpo_data: RPOPersistence,
    traj_diagrams: list[NDArray[np.float64]],
    delay: int,
) -> NDArray[np.float64]:
    """Worker function for parallel min distance computation."""
    # Compute Wasserstein distance matrix
    wasserstein_matrix = compute_wasserstein_matrix(traj_diagrams, rpo_data.diagrams)

    # Apply time-delay embedding
    embedded_matrix = apply_delay_embedding(wasserstein_matrix, delay)

    # Minimum over all phases for each timestep
    min_dists = np.min(embedded_matrix, axis=1)

    # Pad to original trajectory length (delay embedding reduces length)
    num_traj = len(traj_diagrams)
    result = np.full(num_traj, np.inf, dtype=np.float64)
    result[: len(min_dists)] = min_dists

    return result


class PHADetector:
    """Persistent Homology Approach shadowing detector.

    Detects shadowing events by computing Wasserstein distances between
    persistence diagrams of trajectory and RPO snapshots. Uses time-delay
    embedding.

    Unlike SSA, PHA quotients out the continuous spatial symmetry via
    persistence diagrams, eliminating the need for explicit shift optimization.
    This results in a 2D search space (timestep x phase) instead of 3D.

    Note: Events returned by PHA have zero-filled `shifts` arrays. To compute
    actual shifts for visualization, use `compute_event_shifts` from
    the `shifts` module.
    """

    def __init__(
        self,
        rpos: Sequence[RPO],
        dt: float,
        resolution: int,
        delay: int,
    ):
        """Initialize detector with precomputed RPO persistence diagrams.

        Each RPO is integrated using its native timestep and converted to
        persistence diagrams for comparison.

        Args:
            rpos: Sequence of RPO objects to detect shadowing against.
            dt: Timestep of trajectories that will be analyzed.
            resolution: Spatial resolution for physical space representation.
            delay: Time-delay embedding window size.
        """
        self.dt = dt
        self.resolution = resolution
        self.delay = delay
        self.rpos = list(rpos)
        self.rpo_data_list = [RPOPersistence.from_rpo(rpo, resolution) for rpo in rpos]

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
            threshold: Maximum Wasserstein distance for a point to be considered close.
            min_duration: Minimum event duration in timesteps.
            show_progress: Whether to display a progress bar.
            n_jobs: Number of parallel workers. Use -1 for all CPUs.

        Returns:
            List of `ShadowingEvent` sorted by `(start_timestep, rpo_index)`.
            Note: The `shifts` field will be zeros; use `compute_shifts_for_event`
            to populate it for visualization.
        """
        # Convert trajectory to physical space and compute persistence diagrams
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        traj_diagrams = compute_trajectory_diagrams(trajectory_physical)

        n_workers = resolve_n_jobs(n_jobs)

        if n_workers == 1:
            events = self._detect_sequential(traj_diagrams, threshold, min_duration, show_progress)
        else:
            events = self._detect_parallel(
                traj_diagrams, threshold, min_duration, show_progress, n_workers
            )

        events.sort(key=lambda e: (e.start_timestep, e.rpo_index))
        return events

    def _detect_sequential(
        self,
        traj_diagrams: list[NDArray[np.float64]],
        threshold: float,
        min_duration: int,
        show_progress: bool,
    ) -> list[ShadowingEvent]:
        """Run detection sequentially over all RPOs."""
        events: list[ShadowingEvent] = []
        iterator: Iterable[RPOPersistence] = self.rpo_data_list

        if show_progress:
            iterator = tqdm(iterator, total=len(self.rpo_data_list), desc="Detecting", leave=False)

        for rpo_data in iterator:
            # Compute Wasserstein distance matrix
            wasserstein_matrix = compute_wasserstein_matrix(traj_diagrams, rpo_data.diagrams)

            # Apply time-delay embedding
            embedded_matrix = apply_delay_embedding(wasserstein_matrix, self.delay)

            # Extract events
            rpo_events = extract_shadowing_events_2d(
                embedded_matrix,
                rpo_data,
                threshold,
                min_duration,
                self.delay,
            )
            events.extend(rpo_events)

        return events

    def _detect_parallel(
        self,
        traj_diagrams: list[NDArray[np.float64]],
        threshold: float,
        min_duration: int,
        show_progress: bool,
        n_workers: int,
    ) -> list[ShadowingEvent]:
        """Run detection in parallel over RPOs."""
        events: list[ShadowingEvent] = []
        worker = partial(
            _detect_single_rpo,
            traj_diagrams=traj_diagrams,
            threshold=threshold,
            min_duration=min_duration,
            delay=self.delay,
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
        """Compute minimum Wasserstein distance to any RPO at each trajectory timestep.

        For each timestep, finds the minimum distance over all RPOs and all
        phase offsets (after time-delay embedding). Useful for threshold selection.

        Note: Due to time-delay embedding, the last `(delay - 1)` timesteps will
        have infinite distance (not enough future data for embedding).

        Returns:
            Array of shape `(num_timesteps,)` with minimum distances.
        """
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        traj_diagrams = compute_trajectory_diagrams(trajectory_physical)
        n_workers = resolve_n_jobs(n_jobs)

        if n_workers == 1:
            return self._min_distances_sequential(traj_diagrams, show_progress)
        else:
            return self._min_distances_parallel(traj_diagrams, show_progress, n_workers)

    def _min_distances_sequential(
        self,
        traj_diagrams: list[NDArray[np.float64]],
        show_progress: bool,
    ) -> NDArray[np.float64]:
        """Compute min distances sequentially."""
        num_traj = len(traj_diagrams)
        min_dists = np.full(num_traj, np.inf, dtype=np.float64)
        iterator = self.rpo_data_list

        if show_progress:
            iterator = tqdm(iterator, desc="Min distances", leave=False)

        for rpo_data in iterator:
            wasserstein_matrix = compute_wasserstein_matrix(traj_diagrams, rpo_data.diagrams)
            embedded_matrix = apply_delay_embedding(wasserstein_matrix, self.delay)

            # Minimum over all phases
            rpo_min = np.min(embedded_matrix, axis=1)
            min_dists[: len(rpo_min)] = np.minimum(min_dists[: len(rpo_min)], rpo_min)

        return min_dists

    def _min_distances_parallel(
        self,
        traj_diagrams: list[NDArray[np.float64]],
        show_progress: bool,
        n_workers: int,
    ) -> NDArray[np.float64]:
        """Compute min distances in parallel."""
        num_traj = len(traj_diagrams)
        min_dists = np.full(num_traj, np.inf, dtype=np.float64)
        worker = partial(
            _min_dist_single_rpo,
            traj_diagrams=traj_diagrams,
            delay=self.delay,
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
        # Filter out infinite distances (from delay embedding edge effects)
        finite_distances = min_distances[np.isfinite(min_distances)]
        threshold = float(np.quantile(finite_distances, f_close))
        events = self.detect(
            trajectory_fourier, threshold, min_duration, show_progress, n_jobs=n_jobs
        )
        return events, threshold
