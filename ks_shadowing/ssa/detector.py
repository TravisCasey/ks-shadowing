"""State Space Approach (SSA) shadowing detector.

Detects shadowing by computing :math:`L_2` distances between trajectory
snapshots and RPO phases in physical space, using FFT cross-correlation to
optimize over spatial shifts.
"""

from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray
from scipy import fft
from tqdm import tqdm

from ks_shadowing.core import DOMAIN_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.parallel import _resolve_n_jobs
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import (
    _tile_periodic,
    interleaved_to_physical,
    to_comoving_frame,
)
from ks_shadowing.ssa.pathfinding import _extract_shadowing_events_3d
from ks_shadowing.ssa.rpo import _RPOStateSpace


def _compute_distances_sq(
    trajectory_physical: NDArray[np.float64],
    rpo_data: _RPOStateSpace,
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
    rpo_tiled = _tile_periodic(rpo_comoving, num_timesteps + period - 1)

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


def _detect_single_rpo(
    rpo_data: _RPOStateSpace,
    trajectory_physical: NDArray[np.float64],
    threshold: float,
    min_duration: int,
) -> list[ShadowingEvent]:
    """Worker function for parallel detection."""
    return _extract_shadowing_events_3d(
        _compute_distances_sq(trajectory_physical, rpo_data),
        rpo_data,
        threshold,
        min_duration,
    )


def _min_dist_single_rpo(
    rpo_data: _RPOStateSpace,
    trajectory_physical: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Worker function for parallel min distance computation."""
    min_dists_sq = np.full(len(trajectory_physical), np.inf, dtype=np.float64)
    for _, dist_sq in _compute_distances_sq(trajectory_physical, rpo_data):
        # Min over shift dimension, keeping timestep dimension
        phase_min_sq = np.min(dist_sq, axis=1)
        np.minimum(min_dists_sq, phase_min_sq, out=min_dists_sq)
    return np.sqrt(min_dists_sq)


class SSADetector:
    r"""State Space Approach shadowing detector.

    Detects shadowing events by computing :math:`L_2` distances between a
    trajectory and RPO phases in physical space, then extracting contiguous
    shadowing episodes via connected component analysis and longest-path
    finding.

    The distance computation operates in a 3D space of
    ``(timestep, phase, shift)``, where ``phase`` indexes the RPO period and
    ``shift`` indexes the spatial translation. Both trajectory and RPO are first
    transformed to the RPO's co-moving reference frame, which removes the RPO's
    spatial drift and makes it truly periodic. FFT cross-correlation then
    computes distances to all spatial shifts simultaneously in
    :math:`O(N \log N)` time per timestep.

    The full distance array has shape ``(num_timesteps, period, resolution)``
    but is never materialized at once. Instead, it is generated one phase
    slice at a time: a ``(num_timesteps, resolution)`` array per phase
    offset. FFTs for the trajectory and tiled RPO are precomputed once and
    reused across all phase slices.

    From these distances, all entries below the threshold are collected as
    "close passes" and grouped into connected components using 26-connectivity
    in the 3D grid (with wraparound in the phase and shift dimensions). Each
    component represents a cluster of nearby ``(timestep, phase, shift)``
    points. A longest path is then extracted from each component, subject to the
    constraint that ``timestep`` advances by exactly 1, phase remains constant
    (trajectory and RPO co-evolve), and spatial shift changes by at most 1 per
    step. Each such path becomes a
    :class:`~ks_shadowing.core.event.ShadowingEvent`.

    Parameters
    ----------
    rpos : Sequence[RPO]
        RPO objects to detect shadowing against. Each is integrated using its
        native timestep to preserve numerical accuracy.
    dt : float
        Timestep of trajectories that will be analyzed.
    resolution : int
        Spatial resolution for physical-space representation.
    """

    def __init__(
        self,
        rpos: Sequence[RPO],
        dt: float,
        resolution: int,
    ):
        self.dt = dt
        self.resolution = resolution
        self.rpos = list(rpos)
        self.rpo_data = [_RPOStateSpace.from_rpo(rpo, resolution) for rpo in rpos]

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
            Maximum :math:`L_2` distance for a point to be considered close.
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
        n_workers = _resolve_n_jobs(n_jobs)

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
        iterator: Iterable[_RPOStateSpace] = self.rpo_data

        if show_progress:
            iterator = tqdm(iterator, total=len(self.rpo_data), desc="Detecting", leave=False)

        for rpo_data in iterator:
            rpo_events = _extract_shadowing_events_3d(
                _compute_distances_sq(trajectory_physical, rpo_data),
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
        r"""Compute minimum :math:`L_2` distance to any RPO at each timestep.

        For each timestep, finds the minimum distance over all RPOs, all
        phase offsets, and all spatial shifts. Useful for threshold selection.

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
            Minimum distance to any RPO at each timestep.
        """
        trajectory_physical = interleaved_to_physical(trajectory_fourier, self.resolution)
        n_workers = _resolve_n_jobs(n_jobs)

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
        iterator = self.rpo_data

        if show_progress:
            iterator = tqdm(iterator, desc="Min distances", leave=False)

        for rpo_data in iterator:
            for _, dist_sq in _compute_distances_sq(trajectory_physical, rpo_data):
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
        threshold = float(np.quantile(min_distances, f_close))
        events = self.detect(
            trajectory_fourier, threshold, min_duration, show_progress, n_jobs=n_jobs
        )
        return events, threshold
