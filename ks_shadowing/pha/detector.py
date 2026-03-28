"""Persistent Homology Approach (PHA) shadowing detector.

Detects shadowing by computing Wasserstein distances between trajectory
snapshots and RPO phases in the space of persistence diagrams, using a custom
`Hera <https://github.com/anigmetov/hera>`_ harness for batched Wasserstein
computations.
"""

from collections.abc import Sequence
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.parallel import _resolve_n_jobs
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical
from ks_shadowing.pha.pathfinding import _extract_shadowing_events_2d
from ks_shadowing.pha.persistence import (
    _apply_delay_embedding,
    _compute_trajectory_diagrams,
    _RPOPersistence,
)
from ks_shadowing.pha.shifts import _compute_event_shifts
from ks_shadowing.pha.wasserstein import _flatten_diagrams, _wasserstein_column

# Module-level state for pool workers (set by initializer)
_shared_points_shm_name: str | None = None
_shared_offsets_shm_name: str | None = None
_shared_num_traj: int = 0
_shared_points_shape: tuple[int, int] = (0, 2)


def _pha_pool_initializer(
    points_shm_name: str,
    offsets_shm_name: str,
    num_traj: int,
    points_shape: tuple[int, int],
) -> None:
    """Attach to shared memory containing pre-flattened trajectory diagrams."""
    global _shared_points_shm_name, _shared_offsets_shm_name  # noqa: PLW0603
    global _shared_num_traj, _shared_points_shape  # noqa: PLW0603
    _shared_points_shm_name = points_shm_name
    _shared_offsets_shm_name = offsets_shm_name
    _shared_num_traj = num_traj
    _shared_points_shape = points_shape


def _compute_single_column(
    args: tuple[int, int, NDArray[np.float64]],
) -> tuple[int, int, NDArray[np.float64]]:
    """Compute one Wasserstein column using shared pre-flattened trajectory data.

    Parameters
    ----------
    args : tuple
        ``(rpo_list_index, phase_index, rpo_diagram)`` where ``rpo_diagram``
        is a single RPO phase's persistence diagram.

    Returns
    -------
    tuple[int, int, NDArray[np.float64]]
        ``(rpo_list_index, phase_index, column)`` where ``column`` has shape
        ``(I,)``.
    """
    rpo_list_index, phase_index, rpo_diagram = args

    points_shm = SharedMemory(name=_shared_points_shm_name)
    offsets_shm = SharedMemory(name=_shared_offsets_shm_name)
    try:
        traj_points = np.ndarray(_shared_points_shape, dtype=np.float64, buffer=points_shm.buf)
        traj_offsets = np.ndarray((_shared_num_traj + 1,), dtype=np.int64, buffer=offsets_shm.buf)

        column = _wasserstein_column(traj_points, traj_offsets, _shared_num_traj, rpo_diagram)
    finally:
        points_shm.close()
        offsets_shm.close()

    return rpo_list_index, phase_index, column


class PHADetector:
    r"""Persistent Homology Approach shadowing detector.

    Detects shadowing events by computing Wasserstein distances between
    persistence diagrams of trajectory and RPO snapshots.

    Unlike SSA, PHA quotients out the continuous spatial symmetry via
    persistence diagrams, eliminating the need for explicit shift optimization.
    This results in a 2D search space ``(timestep, phase)`` instead of 3D.

    The detection pipeline operates as follows:

    1. Convert trajectory and RPO to physical space and compute sublevel-set
       persistence diagrams for each timestep.
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
        """Run detection sequentially, one Wasserstein column at a time."""
        traj_points, traj_offsets = _flatten_diagrams(traj_diagrams)
        num_traj = len(traj_diagrams)
        events: list[ShadowingEvent] = []
        total_phases = sum(len(rd.diagrams) for rd in self.rpo_data)

        phase_iter = (
            (rpo_data, diagram) for rpo_data in self.rpo_data for diagram in rpo_data.diagrams
        )
        if show_progress:
            phase_iter = tqdm(phase_iter, total=total_phases, desc="Detecting", leave=False)

        columns: list[NDArray[np.float64]] = []
        current_rpo: _RPOPersistence | None = None

        for rpo_data, diagram in phase_iter:
            if current_rpo is not rpo_data:
                # Process completed RPO
                if current_rpo is not None and columns:
                    wass_matrix = np.column_stack(columns)
                    embedded = _apply_delay_embedding(wass_matrix, self.delay)
                    events.extend(
                        _extract_shadowing_events_2d(embedded, current_rpo, threshold, min_duration)
                    )
                columns = []
                current_rpo = rpo_data

            columns.append(_wasserstein_column(traj_points, traj_offsets, num_traj, diagram))

        # Process last RPO
        if current_rpo is not None and columns:
            wass_matrix = np.column_stack(columns)
            embedded = _apply_delay_embedding(wass_matrix, self.delay)
            events.extend(
                _extract_shadowing_events_2d(embedded, current_rpo, threshold, min_duration)
            )

        return events

    def _detect_parallel(
        self,
        traj_diagrams: list[NDArray[np.float64]],
        threshold: float,
        min_duration: int,
        show_progress: bool,
        n_workers: int,
    ) -> list[ShadowingEvent]:
        """Run detection in parallel, one Wasserstein column per task."""
        traj_points, traj_offsets = _flatten_diagrams(traj_diagrams)
        num_traj = len(traj_diagrams)

        points_shm = SharedMemory(create=True, size=max(1, traj_points.nbytes))
        offsets_shm = SharedMemory(create=True, size=max(1, traj_offsets.nbytes))

        try:
            points_view = np.ndarray(traj_points.shape, dtype=np.float64, buffer=points_shm.buf)
            points_view[:] = traj_points
            offsets_view = np.ndarray(traj_offsets.shape, dtype=np.int64, buffer=offsets_shm.buf)
            offsets_view[:] = traj_offsets

            tasks: list[tuple[int, int, NDArray[np.float64]]] = [
                (rpo_index, phase_index, diagram)
                for rpo_index, rpo_data in enumerate(self.rpo_data)
                for phase_index, diagram in enumerate(rpo_data.diagrams)
            ]

            with Pool(
                n_workers,
                initializer=_pha_pool_initializer,
                initargs=(
                    points_shm.name,
                    offsets_shm.name,
                    num_traj,
                    traj_points.shape,
                ),
            ) as pool:
                results = pool.imap_unordered(_compute_single_column, tasks)

                if show_progress:
                    results = tqdm(results, total=len(tasks), desc="Detecting", leave=False)

                # Collect columns grouped by RPO
                columns_by_rpo: dict[int, dict[int, NDArray[np.float64]]] = {}
                for rpo_index, phase_index, column in results:
                    columns_by_rpo.setdefault(rpo_index, {})[phase_index] = column
        finally:
            points_shm.close()
            points_shm.unlink()
            offsets_shm.close()
            offsets_shm.unlink()

        # Assemble matrices and run pathfinding in parent (cheap)
        events: list[ShadowingEvent] = []
        for rpo_index, rpo_data in enumerate(self.rpo_data):
            phase_columns = columns_by_rpo.get(rpo_index)
            if phase_columns is None:
                continue
            num_phases = len(rpo_data.diagrams)
            wass_matrix = np.empty((num_traj, num_phases), dtype=np.float64)
            for j in range(num_phases):
                wass_matrix[:, j] = phase_columns[j]

            embedded = _apply_delay_embedding(wass_matrix, self.delay)
            events.extend(_extract_shadowing_events_2d(embedded, rpo_data, threshold, min_duration))

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
        """Compute min distances sequentially, one Wasserstein column at a time."""
        traj_points, traj_offsets = _flatten_diagrams(traj_diagrams)
        num_traj = len(traj_diagrams)
        min_dists = np.full(num_traj, np.inf, dtype=np.float64)
        total_phases = sum(len(rd.diagrams) for rd in self.rpo_data)

        phase_iter = (
            (rpo_data, diagram) for rpo_data in self.rpo_data for diagram in rpo_data.diagrams
        )
        if show_progress:
            phase_iter = tqdm(phase_iter, total=total_phases, desc="Min distances", leave=False)

        columns: list[NDArray[np.float64]] = []
        current_rpo: _RPOPersistence | None = None

        for rpo_data, diagram in phase_iter:
            if current_rpo is not rpo_data:
                # Process completed RPO
                if current_rpo is not None and columns:
                    wass_matrix = np.column_stack(columns)
                    embedded = _apply_delay_embedding(wass_matrix, self.delay)
                    rpo_min = np.min(embedded, axis=1)
                    min_dists[: len(rpo_min)] = np.minimum(min_dists[: len(rpo_min)], rpo_min)
                columns = []
                current_rpo = rpo_data

            columns.append(_wasserstein_column(traj_points, traj_offsets, num_traj, diagram))

        # Process last RPO
        if current_rpo is not None and columns:
            wass_matrix = np.column_stack(columns)
            embedded = _apply_delay_embedding(wass_matrix, self.delay)
            rpo_min = np.min(embedded, axis=1)
            min_dists[: len(rpo_min)] = np.minimum(min_dists[: len(rpo_min)], rpo_min)

        return min_dists

    def _min_distances_parallel(
        self,
        traj_diagrams: list[NDArray[np.float64]],
        show_progress: bool,
        n_workers: int,
    ) -> NDArray[np.float64]:
        """Compute min distances in parallel, one Wasserstein column per task."""
        traj_points, traj_offsets = _flatten_diagrams(traj_diagrams)
        num_traj = len(traj_diagrams)
        min_dists = np.full(num_traj, np.inf, dtype=np.float64)

        points_shm = SharedMemory(create=True, size=max(1, traj_points.nbytes))
        offsets_shm = SharedMemory(create=True, size=max(1, traj_offsets.nbytes))

        try:
            points_view = np.ndarray(traj_points.shape, dtype=np.float64, buffer=points_shm.buf)
            points_view[:] = traj_points
            offsets_view = np.ndarray(traj_offsets.shape, dtype=np.int64, buffer=offsets_shm.buf)
            offsets_view[:] = traj_offsets

            tasks: list[tuple[int, int, NDArray[np.float64]]] = [
                (rpo_index, phase_index, diagram)
                for rpo_index, rpo_data in enumerate(self.rpo_data)
                for phase_index, diagram in enumerate(rpo_data.diagrams)
            ]

            with Pool(
                n_workers,
                initializer=_pha_pool_initializer,
                initargs=(
                    points_shm.name,
                    offsets_shm.name,
                    num_traj,
                    traj_points.shape,
                ),
            ) as pool:
                results = pool.imap_unordered(_compute_single_column, tasks)

                if show_progress:
                    results = tqdm(
                        results,
                        total=len(tasks),
                        desc="Min distances",
                        leave=False,
                    )

                columns_by_rpo: dict[int, dict[int, NDArray[np.float64]]] = {}
                for rpo_index, phase_index, column in results:
                    columns_by_rpo.setdefault(rpo_index, {})[phase_index] = column
        finally:
            points_shm.close()
            points_shm.unlink()
            offsets_shm.close()
            offsets_shm.unlink()

        # Assemble matrices, apply delay embedding, reduce
        for rpo_index, rpo_data in enumerate(self.rpo_data):
            phase_columns = columns_by_rpo.get(rpo_index)
            if phase_columns is None:
                continue
            num_phases = len(rpo_data.diagrams)
            wass_matrix = np.empty((num_traj, num_phases), dtype=np.float64)
            for j in range(num_phases):
                wass_matrix[:, j] = phase_columns[j]

            embedded = _apply_delay_embedding(wass_matrix, self.delay)
            rpo_min = np.min(embedded, axis=1)
            min_dists[: len(rpo_min)] = np.minimum(min_dists[: len(rpo_min)], rpo_min)

        return min_dists

    def auto_detect(  # noqa: PLR0913
        self,
        trajectory_fourier: NDArray[np.float64],
        threshold_quantile: float = 0.4,
        min_duration: int = 1,
        show_progress: bool = False,
        n_jobs: int = 1,
        downsample: int = 1,
    ) -> tuple[list[ShadowingEvent], float]:
        """Detect shadowing events with automatic threshold selection.

        The threshold is set to the ``threshold_quantile`` quantile of minimum
        distances across the trajectory. For example, ``threshold_quantile=0.4``
        means 40% of timesteps will have minimum distance below the threshold.

        Parameters
        ----------
        trajectory_fourier : NDArray[np.float64], shape (num_timesteps, 30)
            Trajectory in interleaved Fourier format.
        threshold_quantile : float, optional
            Quantile for threshold selection. Default is 0.4.
        min_duration : int, optional
            Minimum event duration in timesteps. Default is 1.
        show_progress : bool, optional
            Whether to display a progress bar. Default is ``False``.
        n_jobs : int, optional
            Number of parallel workers. Use -1 for all CPUs. Default is 1.
        downsample : int, optional
            Stride for subsampling the trajectory during threshold estimation.
            Only every ``downsample``-th timestep is used for computing minimum
            distances. The full trajectory is still used for detection.
            Default is 1 (no subsampling).

        Returns
        -------
        events : list[ShadowingEvent]
            Detected events sorted by ``(start_timestep, rpo_index)``.
        threshold : float
            The automatically selected threshold.
        """
        threshold_trajectory = trajectory_fourier[::downsample]
        min_distances = self.compute_min_distances(
            threshold_trajectory, show_progress=show_progress, n_jobs=n_jobs
        )
        # Filter out infinite distances (from delay embedding edge effects)
        finite_distances = min_distances[np.isfinite(min_distances)]
        threshold = float(np.quantile(finite_distances, threshold_quantile))
        events = self.detect(
            trajectory_fourier, threshold, min_duration, show_progress, n_jobs=n_jobs
        )
        return events, threshold
