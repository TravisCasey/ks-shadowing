"""Persistent Homology Approach (PHAA) shadowing detector.

Detects shadowing by computing Wasserstein distances between trajectory
snapshots and RPO phases in the space of persistence diagrams, using
`GUDHI <https://gudhi.inria.fr/>`_ for efficient computation of persistence
diagrams and a custom `Hera <https://github.com/anigmetov/hera>`_ harness for
batched Wasserstein computations.
"""

from collections.abc import Iterable, Sequence
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical
from ks_shadowing.core.util import _resolve_n_jobs
from ks_shadowing.pha.pathfinding import _extract_shadowing_events_2d
from ks_shadowing.pha.persistence import (
    _apply_delay_embedding,
    _compute_trajectory_diagrams,
    _RPOPersistence,
)
from ks_shadowing.pha.shifts import _compute_event_shifts
from ks_shadowing.pha.wasserstein import _wasserstein_matrix


def _detect_single_rpo(
    rpo_data: _RPOPersistence,
    traj_diagrams: list[NDArray[np.float64]],
    threshold: float,
    min_duration: int,
    delay: int,
) -> list[ShadowingEvent]:
    """Worker function for parallel detection."""
    wass_matrix = _wasserstein_matrix(traj_diagrams, rpo_data.diagrams)
    embedded_matrix = _apply_delay_embedding(wass_matrix, delay)
    return _extract_shadowing_events_2d(
        embedded_matrix,
        rpo_data,
        threshold,
        min_duration,
    )


def _min_dist_single_rpo(
    rpo_data: _RPOPersistence,
    traj_diagrams: list[NDArray[np.float64]],
    delay: int,
) -> NDArray[np.float64]:
    """Worker function for parallel min distance computation."""
    wass_matrix = _wasserstein_matrix(traj_diagrams, rpo_data.diagrams)
    embedded_matrix = _apply_delay_embedding(wass_matrix, delay)

    # Minimum over all phases for each timestep
    min_dists = np.min(embedded_matrix, axis=1)

    # Pad to original trajectory length (delay embedding reduces length)
    num_traj = len(traj_diagrams)
    result = np.full(num_traj, np.inf, dtype=np.float64)
    result[: len(min_dists)] = min_dists

    return result


class PHADetector:
    r"""Persistent Homology Approach shadowing detector.

    Detects shadowing events by computing Wasserstein distances between
    persistence diagrams of trajectory and RPO snapshots.

    Unlike SSA, PHA quotients out the continuous spatial symmetry via
    persistence diagrams, eliminating the need for explicit shift optimization.
    This results in a 2D search space ``(timestep, phase)`` instead of 3D.

    The detection pipeline operates as follows:

    1. Convert trajectory and RPO to physical space and compute sublevel-set
       persistence diagrams (GUDHI periodic cubical complex) for each timestep.
    2. Compute the full Wasserstein distance matrix between trajectory and RPO
       diagrams using the Hera C++ batch API.
    3. Apply time-delay embedding: sum :math:`w` consecutive diagonal entries
       to increase robustness by comparing windows of timesteps rather than
       individual snapshots.
    4. Collect all entries below the threshold as "close passes" in the 2D
       ``(timestep, phase)`` grid.
    5. Group close passes into 8-connected components using a single-pass sweep
       with dense label arrays and union-find.
    6. Extract the longest valid path through each component, where both
       trajectory timestep and RPO phase advance by exactly 1 per step.
    7. Each path becomes a :class:`~ks_shadowing.core.event.ShadowingEvent`.
    8. Compute spatial shifts for each event post-hoc using
       :math:`L_2` distances and dynamic programming.

    Parameters
    ----------
    rpos : Sequence[RPO]
        RPO objects to detect shadowing against. Each is integrated using
        its native timestep to preserve numerical accuracy.
    dt : float
        Timestep of trajectories that will be analyzed.
    resolution : int
        Spatial resolution for physical-space representation.
    delay : int
        Time-delay embedding window size.
    """

    def __init__(
        self,
        rpos: Sequence[RPO],
        dt: float,
        resolution: int,
        delay: int,
    ):
        self.dt = dt
        self.resolution = resolution
        self.delay = delay
        self.rpos = list(rpos)
        self.rpo_data = [_RPOPersistence.from_rpo(rpo, resolution) for rpo in rpos]

    def detect(
        self,
        trajectory_fourier: NDArray[np.float64],
        threshold: float,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> list[ShadowingEvent]:
        r"""Detect shadowing events for all RPOs.

        Parameters
        ----------
        trajectory_fourier : NDArray[np.float64], shape (num_timesteps, 30)
            Trajectory in interleaved Fourier format from
            :func:`~ks_shadowing.core.integrator.ksint`.
        threshold : float
            Maximum Wasserstein distance for a point to be considered close.
        min_duration : int, optional
            Minimum event duration in timesteps. Default is 1.
        show_progress : bool, optional
            Whether to display a progress bar. Default is ``False``.
        n_jobs : int, optional
            Number of parallel workers. Use -1 for all CPUs. Default is 1.

        Returns
        -------
        list[ShadowingEvent]
            Events sorted by ``(start_timestep, rpo_index)``.
        """
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        traj_diagrams = _compute_trajectory_diagrams(trajectory_physical)

        n_workers = _resolve_n_jobs(n_jobs)

        if n_workers == 1:
            events = self._detect_sequential(traj_diagrams, threshold, min_duration, show_progress)
        else:
            events = self._detect_parallel(
                traj_diagrams, threshold, min_duration, show_progress, n_workers
            )

        # Compute spatial shifts for each event. PHA quotients out shifts
        # during detection, so they are reconstructed post-hoc using L2
        # distances in the co-moving frame.
        rpo_by_index = {rpo.index: rpo for rpo in self.rpos}
        events = [
            _compute_event_shifts(
                event, trajectory_fourier, rpo_by_index[event.rpo_index], self.resolution
            )
            for event in events
        ]

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
        iterator: Iterable[_RPOPersistence] = self.rpo_data

        if show_progress:
            iterator = tqdm(iterator, total=len(self.rpo_data), desc="Detecting", leave=False)

        for rpo_data in iterator:
            wass_matrix = _wasserstein_matrix(traj_diagrams, rpo_data.diagrams)
            embedded_matrix = _apply_delay_embedding(wass_matrix, self.delay)
            rpo_events = _extract_shadowing_events_2d(
                embedded_matrix,
                rpo_data,
                threshold,
                min_duration,
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
            results = pool.imap_unordered(worker, self.rpo_data)

            if show_progress:
                results = tqdm(results, total=len(self.rpo_data), desc="Detecting", leave=False)

            for rpo_events in results:
                events.extend(rpo_events)

        return events

    def compute_min_distances(
        self,
        trajectory_fourier: NDArray[np.float64],
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> NDArray[np.float64]:
        r"""Compute minimum Wasserstein distance to any RPO at each timestep.

        For each timestep, finds the minimum distance over all RPOs and all
        phase offsets (after time-delay embedding). Useful for threshold
        selection.

        Due to time-delay embedding, the last ``(delay - 1)`` timesteps will
        have infinite distance (not enough future data for embedding).

        Parameters
        ----------
        trajectory_fourier : NDArray[np.float64], shape (num_timesteps, 30)
            Trajectory in interleaved Fourier format.
        show_progress : bool, optional
            Whether to display a progress bar. Default is ``False``.
        n_jobs : int, optional
            Number of parallel workers. Use -1 for all CPUs. Default is 1.

        Returns
        -------
        NDArray[np.float64], shape (num_timesteps,)
            Minimum Wasserstein distance to any RPO at each timestep.
        """
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        traj_diagrams = _compute_trajectory_diagrams(trajectory_physical)
        n_workers = _resolve_n_jobs(n_jobs)

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
        iterator = self.rpo_data

        if show_progress:
            iterator = tqdm(iterator, desc="Min distances", leave=False)

        for rpo_data in iterator:
            wass_matrix = _wasserstein_matrix(traj_diagrams, rpo_data.diagrams)
            embedded_matrix = _apply_delay_embedding(wass_matrix, self.delay)

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
            results = pool.imap_unordered(worker, self.rpo_data)

            if show_progress:
                results = tqdm(results, total=len(self.rpo_data), desc="Min distances", leave=False)

            for rpo_min in results:
                np.minimum(min_dists, rpo_min, out=min_dists)

        return min_dists

    def auto_detect(
        self,
        trajectory_fourier: NDArray[np.float64],
        f_close: float = 0.4,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> tuple[list[ShadowingEvent], float]:
        """Detect shadowing events with automatic threshold selection.

        The threshold is set to the ``f_close`` quantile of minimum distances
        across the trajectory. For example, ``f_close=0.4`` means 40% of
        timesteps will have minimum distance below the threshold.

        Parameters
        ----------
        trajectory_fourier : NDArray[np.float64], shape (num_timesteps, 30)
            Trajectory in interleaved Fourier format.
        f_close : float, optional
            Quantile for threshold selection. Default is 0.4.
        min_duration : int, optional
            Minimum event duration in timesteps. Default is 1.
        show_progress : bool, optional
            Whether to display a progress bar. Default is ``False``.
        n_jobs : int, optional
            Number of parallel workers. Use -1 for all CPUs. Default is 1.

        Returns
        -------
        events : list[ShadowingEvent]
            Detected events sorted by ``(start_timestep, rpo_index)``.
        threshold : float
            The automatically selected threshold.
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
