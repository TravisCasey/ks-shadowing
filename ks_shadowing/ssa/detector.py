"""State Space Approach (SSA) shadowing detector.

Detects shadowing by computing :math:`L_2` distances between trajectory
snapshots and RPO phases using 17-mode FFT cross-correlation to optimize
over spatial shifts.
"""

from collections.abc import Iterator, Sequence
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ks_shadowing.core import DEFAULT_CHUNK_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.parallel import _resolve_n_jobs
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq
from ks_shadowing.ssa.pathfinding import _extract_shadowing_events_3d
from ks_shadowing.ssa.rpo import _RPOStateSpace


def _compute_distances_sq(
    trajectory: KSTrajectory,
    rpo_data: _RPOStateSpace,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
    r"""Yield squared distance arrays for each phase offset using co-moving frame.

    Both trajectory and RPO are transformed to the RPO's co-moving frame where
    the RPO becomes truly periodic. The trajectory is processed in chunks via
    :meth:`~ks_shadowing.core.trajectory.KSTrajectory.chunks_fourier` to
    control the output size of
    :func:`~ks_shadowing.core.trajectory.shift_distances_sq`.

    Yields ``(phase, chunk_start, dist_sq_chunk)`` tuples where
    ``dist_sq_chunk`` has shape ``(chunk_len, resolution)``.

    Parameters
    ----------
    trajectory : KSTrajectory
        Trajectory in spectral form.
    rpo_data : _RPOStateSpace
        Precomputed RPO trajectory in state space.
    chunk_size : int, optional
        Maximum number of trajectory timesteps per chunk. Default is
        ``DEFAULT_CHUNK_SIZE``.
    """
    period = rpo_data.time_steps
    resolution = rpo_data.resolution
    drift_rate = rpo_data.spatial_shift / period

    traj_comoving = trajectory.to_comoving(drift_rate)
    rpo_comoving = rpo_data.trajectory.to_comoving(drift_rate)
    rpo_tiled = rpo_comoving.tile(len(traj_comoving) + period - 1)

    for chunk_start, chunk_modes in traj_comoving.chunks_fourier(chunk_size):
        chunk_len = chunk_modes.shape[0]
        for phase in range(period):
            rpo_slice = rpo_tiled.modes[chunk_start + phase : chunk_start + phase + chunk_len]
            dist_sq = shift_distances_sq(chunk_modes, rpo_slice, resolution)
            yield phase, chunk_start, np.maximum(dist_sq, 0.0)


# Module-level state for pool workers (set by initializer)
_shared_traj_shm_name: str | None = None
_shared_traj_shape: tuple[int, int] = (0, 0)
_shared_traj_resolution: int = 0
_shared_traj_dt: float = 0.0


def _ssa_pool_initializer(
    shm_name: str, shape: tuple[int, int], resolution: int, dt: float
) -> None:
    """Store shared memory metadata for trajectory modes array."""
    global _shared_traj_shm_name, _shared_traj_shape  # noqa: PLW0603
    global _shared_traj_resolution, _shared_traj_dt  # noqa: PLW0603
    _shared_traj_shm_name = shm_name
    _shared_traj_shape = shape
    _shared_traj_resolution = resolution
    _shared_traj_dt = dt


def _reconstruct_trajectory_from_shm() -> KSTrajectory:
    """Reconstruct a KSTrajectory from the shared memory buffer."""
    shm = SharedMemory(name=_shared_traj_shm_name)
    modes = np.ndarray(_shared_traj_shape, dtype=np.complex128, buffer=shm.buf)
    return KSTrajectory(modes=modes, dt=_shared_traj_dt, resolution=_shared_traj_resolution), shm


def _detect_single_rpo(
    args: tuple[_RPOStateSpace, float, int, int],
) -> list[ShadowingEvent]:
    """Worker function for parallel detection using shared trajectory."""
    rpo_data, threshold, min_duration, chunk_size = args
    trajectory, shm = _reconstruct_trajectory_from_shm()
    try:
        return _extract_shadowing_events_3d(
            _compute_distances_sq(trajectory, rpo_data, chunk_size),
            rpo_data,
            threshold,
            min_duration,
        )
    finally:
        shm.close()


def _min_dist_single_rpo(
    args: tuple[_RPOStateSpace, int],
) -> NDArray[np.float64]:
    """Worker function for parallel min distance using shared trajectory."""
    rpo_data, chunk_size = args
    trajectory, shm = _reconstruct_trajectory_from_shm()
    try:
        min_dists_sq = np.full(len(trajectory), np.inf, dtype=np.float64)
        for _, chunk_start, dist_sq in _compute_distances_sq(trajectory, rpo_data, chunk_size):
            chunk_end = chunk_start + dist_sq.shape[0]
            phase_min_sq = np.min(dist_sq, axis=1)
            np.minimum(
                min_dists_sq[chunk_start:chunk_end],
                phase_min_sq,
                out=min_dists_sq[chunk_start:chunk_end],
            )
        return np.sqrt(min_dists_sq)
    finally:
        shm.close()


class SSADetector:
    r"""State Space Approach shadowing detector.

    Detects shadowing events by computing :math:`L_2` distances between a
    trajectory and RPO phases using 17-mode FFT cross-correlation, then
    extracting contiguous shadowing episodes via connected component analysis
    and longest-path finding.

    The distance computation operates in a 3D space of
    ``(timestep, phase, shift)``, where ``phase`` indexes the RPO period and
    ``shift`` indexes the spatial translation. Both trajectory and RPO are first
    transformed to the RPO's co-moving reference frame, which removes the RPO's
    spatial drift and makes it truly periodic.
    :func:`~ks_shadowing.core.trajectory.shift_distances_sq` then computes
    distances to all spatial shifts simultaneously using 17-mode FFT products.

    The full distance array has shape ``(num_timesteps, period, resolution)``
    but is never materialized at once. Instead, it is generated one phase
    slice at a time: a ``(num_timesteps, resolution)`` array per phase
    offset.

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
    chunk_size : int, optional
        Maximum number of trajectory timesteps to process at once in the
        distance computation. Controls peak memory usage. Default is
        ``DEFAULT_CHUNK_SIZE``.
    """

    def __init__(
        self,
        rpos: Sequence[RPO],
        dt: float,
        resolution: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.dt = dt
        self.resolution = resolution
        self.chunk_size = chunk_size
        self.rpos = list(rpos)
        self.rpo_data = sorted(
            [_RPOStateSpace.from_rpo(rpo, resolution) for rpo in rpos],
            key=lambda rd: rd.time_steps,
            reverse=True,
        )

    def detect(
        self,
        trajectory: KSTrajectory,
        threshold: float,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> list[ShadowingEvent]:
        r"""Detect shadowing events for all RPOs.

        Parameters
        ----------
        trajectory : KSTrajectory
            Trajectory in spectral form.
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
        n_workers = _resolve_n_jobs(n_jobs)

        if n_workers == 1:
            events = self._detect_sequential(trajectory, threshold, min_duration, show_progress)
        else:
            events = self._detect_parallel(
                trajectory, threshold, min_duration, show_progress, n_workers
            )

        events.sort(key=lambda e: (e.start_timestep, e.rpo_index))
        return events

    def _detect_sequential(
        self,
        trajectory: KSTrajectory,
        threshold: float,
        min_duration: int,
        show_progress: bool,
    ) -> list[ShadowingEvent]:
        """Run detection sequentially with two-level progress (RPO, then phases)."""
        events: list[ShadowingEvent] = []
        num_timesteps = len(trajectory)
        num_chunks = (num_timesteps + self.chunk_size - 1) // self.chunk_size

        rpo_iter = iter(self.rpo_data)
        if show_progress:
            rpo_iter = tqdm(rpo_iter, total=len(self.rpo_data), desc="Detecting", leave=False)

        for rpo_data in rpo_iter:
            generator = _compute_distances_sq(trajectory, rpo_data, self.chunk_size)
            if show_progress:
                total_yields = num_chunks * rpo_data.time_steps
                generator = tqdm(generator, total=total_yields, desc="  Phases", leave=False)

            rpo_events = _extract_shadowing_events_3d(generator, rpo_data, threshold, min_duration)
            events.extend(rpo_events)

        return events

    def _detect_parallel(
        self,
        trajectory: KSTrajectory,
        threshold: float,
        min_duration: int,
        show_progress: bool,
        n_workers: int,
    ) -> list[ShadowingEvent]:
        """Run detection in parallel over RPOs."""
        shm = SharedMemory(create=True, size=max(1, trajectory.modes.nbytes))
        try:
            view = np.ndarray(trajectory.modes.shape, dtype=np.complex128, buffer=shm.buf)
            view[:] = trajectory.modes

            tasks = [
                (rpo_data, threshold, min_duration, self.chunk_size) for rpo_data in self.rpo_data
            ]

            with Pool(
                n_workers,
                initializer=_ssa_pool_initializer,
                initargs=(shm.name, trajectory.modes.shape, self.resolution, self.dt),
            ) as pool:
                results = pool.imap_unordered(_detect_single_rpo, tasks)
                if show_progress:
                    results = tqdm(results, total=len(self.rpo_data), desc="Detecting", leave=False)

                events: list[ShadowingEvent] = []
                for rpo_events in results:
                    events.extend(rpo_events)
        finally:
            shm.close()
            shm.unlink()

        return events

    def compute_min_distances(
        self,
        trajectory: KSTrajectory,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> NDArray[np.float64]:
        r"""Compute minimum :math:`L_2` distance to any RPO at each timestep.

        For each timestep, finds the minimum distance over all RPOs, all
        phase offsets, and all spatial shifts. Useful for threshold selection.

        Parameters
        ----------
        trajectory : KSTrajectory
            Trajectory in spectral form.
        show_progress : bool, optional
            Whether to display a progress bar. Default is ``False``.
        n_jobs : int, optional
            Number of parallel workers. Use -1 for all CPUs. Default is 1.

        Returns
        -------
        NDArray[np.float64], shape (num_timesteps,)
            Minimum distance to any RPO at each timestep.
        """
        n_workers = _resolve_n_jobs(n_jobs)

        if n_workers == 1:
            return self._min_distances_sequential(trajectory, show_progress)
        else:
            return self._min_distances_parallel(trajectory, show_progress, n_workers)

    def _min_distances_sequential(
        self,
        trajectory: KSTrajectory,
        show_progress: bool,
    ) -> NDArray[np.float64]:
        """Compute min distances sequentially with two-level progress."""
        min_dists_sq = np.full(len(trajectory), np.inf, dtype=np.float64)
        num_timesteps = len(trajectory)
        num_chunks = (num_timesteps + self.chunk_size - 1) // self.chunk_size

        rpo_iter = iter(self.rpo_data)
        if show_progress:
            rpo_iter = tqdm(rpo_iter, total=len(self.rpo_data), desc="Min distances", leave=False)

        for rpo_data in rpo_iter:
            generator = _compute_distances_sq(trajectory, rpo_data, self.chunk_size)
            if show_progress:
                total_yields = num_chunks * rpo_data.time_steps
                generator = tqdm(generator, total=total_yields, desc="  Phases", leave=False)

            for _, chunk_start, dist_sq in generator:
                chunk_end = chunk_start + dist_sq.shape[0]
                phase_min_sq = np.min(dist_sq, axis=1)
                np.minimum(
                    min_dists_sq[chunk_start:chunk_end],
                    phase_min_sq,
                    out=min_dists_sq[chunk_start:chunk_end],
                )

        return np.sqrt(min_dists_sq)

    def _min_distances_parallel(
        self,
        trajectory: KSTrajectory,
        show_progress: bool,
        n_workers: int,
    ) -> NDArray[np.float64]:
        """Compute min distances in parallel."""
        shm = SharedMemory(create=True, size=max(1, trajectory.modes.nbytes))
        try:
            view = np.ndarray(trajectory.modes.shape, dtype=np.complex128, buffer=shm.buf)
            view[:] = trajectory.modes

            with Pool(
                n_workers,
                initializer=_ssa_pool_initializer,
                initargs=(shm.name, trajectory.modes.shape, self.resolution, self.dt),
            ) as pool:
                tasks = [(rpo_data, self.chunk_size) for rpo_data in self.rpo_data]
                results = pool.imap_unordered(_min_dist_single_rpo, tasks)
                if show_progress:
                    results = tqdm(
                        results,
                        total=len(self.rpo_data),
                        desc="Min distances",
                        leave=False,
                    )

                min_dists = np.full(len(trajectory), np.inf, dtype=np.float64)
                for rpo_min in results:
                    np.minimum(min_dists, rpo_min, out=min_dists)
        finally:
            shm.close()
            shm.unlink()

        return min_dists

    def auto_detect(
        self,
        trajectory: KSTrajectory,
        threshold_quantile: float = 0.4,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
    ) -> tuple[list[ShadowingEvent], float]:
        """Detect shadowing events with automatic threshold selection.

        The threshold is set to the ``threshold_quantile`` quantile of minimum
        distances across the trajectory. For example, ``threshold_quantile=0.4``
        means 40% of timesteps will have minimum distance below the threshold.

        Parameters
        ----------
        trajectory : KSTrajectory
            Trajectory in spectral form.
        threshold_quantile : float, optional
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
            trajectory, show_progress=show_progress, n_jobs=n_jobs
        )
        threshold = float(np.quantile(min_distances, threshold_quantile))
        events = self.detect(trajectory, threshold, min_duration, show_progress, n_jobs=n_jobs)
        return events, threshold
