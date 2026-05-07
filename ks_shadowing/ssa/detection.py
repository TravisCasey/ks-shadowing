"""State Space Approach (SSA) shadowing detection.

Detects shadowing between a Kuramoto-Sivashinsky trajectory and a collection of
relative periodic orbits (RPOs) by computing :math:`L_2` distances in physical
space. For each RPO the detection grid is 3D
``(trajectory timestep, RPO phase, spatial shift)``; both the trajectory and
the RPO are first transformed to the RPO's co-moving reference frame, where the
RPO is truly periodic, and :func:`~ks_shadowing.core.trajectory.shift_distances_sq`
evaluates all spatial shifts simultaneously via 17-mode FFT cross-correlation.
Pathfinding over the 3D grid is delegated to
:mod:`~ks_shadowing.ssa.pathfinding`.

The full three-dimensional distance array is never materialized: distances are
generated one phase-chunk slice at a time and streamed straight into the
pathfinding pipeline. Trajectory modes are published once into
:class:`~multiprocessing.shared_memory.SharedMemory` and each worker rebuilds a
:class:`~ks_shadowing.core.trajectory.KSTrajectory` view over that buffer,
parallelizing detection across RPOs.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ks_shadowing.core import DEFAULT_CHUNK_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.parallel import (
    _forkserver_pool,
    _resolve_n_jobs,
    _shared_memory_view,
)
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq
from ks_shadowing.ssa.pathfinding import _extract_shadowing_events


def detect(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    threshold: float,
    min_duration: int = 1,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ShadowingEvent]:
    """Detect shadowing events between ``trajectory`` and ``rpos``.

    Transforms the trajectory and each RPO to the RPO's co-moving frame, then
    computes :math:`L_2` distances at all ``(timestep, phase, shift)``
    combinations using 17-mode FFT cross-correlation. Grid entries below
    ``threshold`` are grouped into 26-connected components and the longest
    valid path through each component becomes a
    :class:`~ks_shadowing.core.event.ShadowingEvent`.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory to scan for shadowing events.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to shadow against. Each RPO is integrated at
        its own native timestep to preserve numerical accuracy.
    threshold : float
        Maximum :math:`L_2` distance for a grid entry to count as a close pass.
        Typically set by quantile with :func:`compute_min_distances`. This
        flow is automated via :func:`auto_detect`.
    min_duration : int, optional
        Minimum event duration in timesteps. Default is 1.
    show_progress : bool, optional
        Whether to display a ``tqdm`` progress bar over RPOs. Default is
        ``False``.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all available CPUs. Default
        is 1.
    chunk_size : int, optional
        Maximum number of trajectory timesteps processed at once in the
        distance computation. Default is
        :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    list[ShadowingEvent]
        Events sorted by ``(start_timestep, rpo_index)``.
    """
    rpo_trajectory_pairs = _compute_rpo_trajectory_pairs(rpos, trajectory.resolution)
    n_workers = _resolve_n_jobs(n_jobs)

    return _detect_from_pairs(
        trajectory,
        rpo_trajectory_pairs,
        threshold,
        min_duration,
        show_progress,
        n_workers,
        chunk_size,
    )


def compute_min_distances(
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> NDArray[np.float64]:
    """Compute the minimum :math:`L_2` distance to any RPO at each trajectory
    timestep.

    For each timestep, returns the minimum over all RPOs, all phase offsets,
    and all spatial shifts. Useful for threshold selection by quantile; see
    :func:`auto_detect`.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory whose minimum distances are computed.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to compare against.
    show_progress : bool, optional
        Whether to display a ``tqdm`` progress bar over RPOs. Default is
        ``False``.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all available CPUs. Default
        is 1.
    chunk_size : int, optional
        Maximum number of trajectory timesteps processed at once in the
        distance computation. Default is
        :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    NDArray[np.float64], shape (num_timesteps,)
        Minimum :math:`L_2` distance to any RPO at each timestep.
    """
    rpo_trajectory_pairs = _compute_rpo_trajectory_pairs(rpos, trajectory.resolution)
    n_workers = _resolve_n_jobs(n_jobs)

    return _min_distances_from_pairs(
        trajectory, rpo_trajectory_pairs, show_progress, n_workers, chunk_size
    )


def auto_detect(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    threshold_quantile: float = 0.4,
    min_duration: int = 1,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[ShadowingEvent], float]:
    """Detect shadowing events with a threshold chosen automatically from the
    distribution of per-timestep minimum :math:`L_2` distances.

    The threshold is set to the ``threshold_quantile`` quantile of per-timestep
    minimum distances: for example, ``threshold_quantile=0.4`` selects a
    threshold such that roughly 40% of trajectory timesteps have a minimum
    :math:`L_2` distance to some RPO phase below the threshold.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory to scan for shadowing events.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to detect shadowing against.
    threshold_quantile : float, optional
        Quantile of per-timestep minimum distances used as the detection
        threshold. Default is 0.4.
    min_duration : int, optional
        Minimum event duration in timesteps. Default is 1.
    show_progress : bool, optional
        Whether to display a ``tqdm`` progress bar over RPOs. Default is
        ``False``.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all available CPUs. Default
        is 1.
    chunk_size : int, optional
        Maximum number of trajectory timesteps processed at once in the
        distance computation. Default is
        :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    events : list[ShadowingEvent]
        Detected events sorted by ``(start_timestep, rpo_index)``.
    threshold : float
        The automatically selected threshold.
    """
    rpo_trajectory_pairs = _compute_rpo_trajectory_pairs(rpos, trajectory.resolution)
    n_workers = _resolve_n_jobs(n_jobs)

    min_distances = _min_distances_from_pairs(
        trajectory, rpo_trajectory_pairs, show_progress, n_workers, chunk_size
    )
    threshold = float(np.quantile(min_distances, threshold_quantile))

    events = _detect_from_pairs(
        trajectory,
        rpo_trajectory_pairs,
        threshold,
        min_duration,
        show_progress,
        n_workers,
        chunk_size,
    )
    return events, threshold


def _compute_rpo_trajectory_pairs(
    rpos: Sequence[RPO],
    resolution: int,
) -> list[tuple[RPO, KSTrajectory]]:
    """Integrate each RPO to a spectral trajectory over one period.

    Returned pairs are sorted by RPO period descending so that the
    longest-running RPOs are dispatched first.
    """
    pairs: list[tuple[RPO, KSTrajectory]] = []
    for rpo in rpos:
        rpo_trajectory = KSTrajectory.from_initial_state(
            rpo.modes, rpo.dt, rpo.time_steps, resolution
        )
        pairs.append((rpo, rpo_trajectory))

    pairs.sort(key=lambda pair: pair[0].time_steps, reverse=True)
    return pairs


def _detect_from_pairs(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpo_trajectory_pairs: list[tuple[RPO, KSTrajectory]],
    threshold: float,
    min_duration: int,
    show_progress: bool,
    n_workers: int,
    chunk_size: int,
) -> list[ShadowingEvent]:
    """Detect events across all RPOs.

    Dispatches via a forkserver-backed pool when ``n_workers > 1``, sharing
    the trajectory modes through
    :class:`~multiprocessing.shared_memory.SharedMemory`; otherwise iterates
    sequentially in-process. Returned events are sorted by
    ``(start_timestep, rpo_index)``.
    """
    # Per-RPO cost is dominated by the phase loop in ``_compute_distances_sq``,
    # which scales linearly in ``rpo.time_steps``. Weighting the progress bar
    # by this cost (rather than RPO count) gives a percent-complete that
    # tracks actual work, since ``imap_unordered`` returns short jobs after
    # long ones with descending-period ordering.
    cost_by_rpo_index = {rpo.index: rpo.time_steps for rpo, _ in rpo_trajectory_pairs}
    events: list[ShadowingEvent] = []

    if n_workers == 1:
        with tqdm(
            total=sum(cost_by_rpo_index.values()),
            desc="Detecting",
            unit="phase",
            disable=not show_progress,
        ) as bar:
            for rpo, rpo_trajectory in rpo_trajectory_pairs:
                events.extend(
                    _extract_shadowing_events(
                        _compute_distances_sq(trajectory, rpo, rpo_trajectory, chunk_size),
                        rpo.index,
                        rpo.time_steps,
                        trajectory.resolution,
                        threshold,
                        min_duration,
                    )
                )
                bar.update(cost_by_rpo_index[rpo.index])
    else:
        with (
            _shared_memory_view(trajectory.modes) as modes_shm,
            _forkserver_pool(n_workers) as pool,
            tqdm(
                total=sum(cost_by_rpo_index.values()),
                desc="Detecting",
                unit="phase",
                disable=not show_progress,
            ) as bar,
        ):
            worker_inputs = [
                _DetectWorkerInputs(
                    modes_shm_name=modes_shm.name,
                    modes_shape=trajectory.modes.shape,
                    trajectory_dt=trajectory.dt,
                    resolution=trajectory.resolution,
                    rpo=rpo,
                    rpo_trajectory=rpo_trajectory,
                    threshold=threshold,
                    min_duration=min_duration,
                    chunk_size=chunk_size,
                )
                for rpo, rpo_trajectory in rpo_trajectory_pairs
            ]
            for rpo_index, rpo_events in pool.imap_unordered(_detect_single_rpo, worker_inputs):
                events.extend(rpo_events)
                bar.update(cost_by_rpo_index[rpo_index])

    events.sort(key=lambda event: (event.start_timestep, event.rpo_index))
    return events


def _min_distances_from_pairs(
    trajectory: KSTrajectory,
    rpo_trajectory_pairs: list[tuple[RPO, KSTrajectory]],
    show_progress: bool,
    n_workers: int,
    chunk_size: int,
) -> NDArray[np.float64]:
    """Compute per-timestep minimum :math:`L_2` distances across all RPOs.

    Mirrors :func:`_detect_from_pairs`'s parallel/sequential dispatch: a
    forkserver-backed pool when ``n_workers > 1``, otherwise an in-process
    loop. Each worker (or sequential iteration) returns a per-timestep
    minimum distance array for its RPO, which is reduced elementwise into
    the running global minimum.
    """
    # See ``_detect_from_pairs`` for the rationale behind weighting the bar
    # by ``rpo.time_steps``.
    cost_by_rpo_index = {rpo.index: rpo.time_steps for rpo, _ in rpo_trajectory_pairs}
    min_distances = np.full(len(trajectory), np.inf, dtype=np.float64)

    if n_workers == 1:
        with tqdm(
            total=sum(cost_by_rpo_index.values()),
            desc="Min distances",
            unit="phase",
            disable=not show_progress,
        ) as bar:
            for rpo, rpo_trajectory in rpo_trajectory_pairs:
                min_distances_sq = np.full(len(trajectory), np.inf, dtype=np.float64)
                for _, chunk_start, dist_sq in _compute_distances_sq(
                    trajectory, rpo, rpo_trajectory, chunk_size
                ):
                    chunk_end = chunk_start + dist_sq.shape[0]
                    phase_min_sq = np.min(dist_sq, axis=1)
                    np.minimum(
                        min_distances_sq[chunk_start:chunk_end],
                        phase_min_sq,
                        out=min_distances_sq[chunk_start:chunk_end],
                    )

                np.minimum(min_distances, np.sqrt(min_distances_sq), out=min_distances)
                bar.update(cost_by_rpo_index[rpo.index])
    else:
        with (
            _shared_memory_view(trajectory.modes) as modes_shm,
            _forkserver_pool(n_workers) as pool,
            tqdm(
                total=sum(cost_by_rpo_index.values()),
                desc="Min distances",
                unit="phase",
                disable=not show_progress,
            ) as bar,
        ):
            worker_inputs = [
                _MinDistanceWorkerInputs(
                    modes_shm_name=modes_shm.name,
                    modes_shape=trajectory.modes.shape,
                    trajectory_dt=trajectory.dt,
                    resolution=trajectory.resolution,
                    rpo=rpo,
                    rpo_trajectory=rpo_trajectory,
                    chunk_size=chunk_size,
                )
                for rpo, rpo_trajectory in rpo_trajectory_pairs
            ]
            for rpo_index, rpo_min_distances in pool.imap_unordered(
                _min_distances_single_rpo, worker_inputs
            ):
                np.minimum(min_distances, rpo_min_distances, out=min_distances)
                bar.update(cost_by_rpo_index[rpo_index])

    return min_distances


class _DetectWorkerInputs(NamedTuple):
    """Inputs to :func:`_detect_single_rpo` for one RPO.

    Attributes
    ----------
    modes_shm_name : str
        Name of the :class:`~multiprocessing.shared_memory.SharedMemory` block
        holding the trajectory modes.
    modes_shape : tuple[int, ...]
        Shape of the trajectory modes array in shared memory.
    trajectory_dt : float
        Trajectory integration timestep.
    resolution : int
        Spatial resolution of the trajectory.
    rpo : :class:`~ks_shadowing.core.rpo.RPO`
        Source RPO metadata.
    rpo_trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Integrated RPO trajectory over one period.
    threshold : float
        Maximum :math:`L_2` distance for a grid entry to count as a close pass.
    min_duration : int
        Minimum event duration in timesteps.
    chunk_size : int
        Maximum number of trajectory timesteps processed at once in the
        distance computation.
    """

    modes_shm_name: str
    modes_shape: tuple[int, ...]
    trajectory_dt: float
    resolution: int
    rpo: RPO
    rpo_trajectory: KSTrajectory
    threshold: float
    min_duration: int
    chunk_size: int


class _MinDistanceWorkerInputs(NamedTuple):
    """Inputs to :func:`_min_distances_single_rpo` for one RPO.

    Attributes
    ----------
    modes_shm_name : str
        Name of the :class:`~multiprocessing.shared_memory.SharedMemory` block
        holding the trajectory modes.
    modes_shape : tuple[int, ...]
        Shape of the trajectory modes array in shared memory.
    trajectory_dt : float
        Trajectory integration timestep.
    resolution : int
        Spatial resolution of the trajectory.
    rpo : :class:`~ks_shadowing.core.rpo.RPO`
        Source RPO metadata.
    rpo_trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Integrated RPO trajectory over one period.
    chunk_size : int
        Maximum number of trajectory timesteps processed at once in the
        distance computation.
    """

    modes_shm_name: str
    modes_shape: tuple[int, ...]
    trajectory_dt: float
    resolution: int
    rpo: RPO
    rpo_trajectory: KSTrajectory
    chunk_size: int


@contextmanager
def _attach_trajectory(
    modes_shm_name: str,
    modes_shape: tuple[int, ...],
    trajectory_dt: float,
    resolution: int,
) -> Iterator[KSTrajectory]:
    """Worker-side: attach to the shared trajectory modes and yield a
    :class:`~ks_shadowing.core.trajectory.KSTrajectory` view over the buffer.

    The shared-memory handle is closed (but not unlinked) on exit; the parent
    owns the lifetime of the block.
    """
    modes_shm = SharedMemory(name=modes_shm_name)
    try:
        modes = np.ndarray(modes_shape, dtype=np.complex128, buffer=modes_shm.buf)
        yield KSTrajectory(modes=modes, dt=trajectory_dt, resolution=resolution)
    finally:
        modes_shm.close()


def _detect_single_rpo(inputs: _DetectWorkerInputs) -> tuple[int, list[ShadowingEvent]]:
    """Detect events for a single RPO, worker-side.

    Attaches a :class:`~ks_shadowing.core.trajectory.KSTrajectory` view over
    the shared-memory trajectory modes and runs the full per-RPO pipeline:
    streaming distance computation, close-pass collection, connected-component
    grouping, and longest-path extraction.

    Parameters
    ----------
    inputs : :class:`_DetectWorkerInputs`
        Shared-memory handle, trajectory metadata, RPO, and detection
        parameters.

    Returns
    -------
    rpo_index : int
        Index of the RPO supplied in ``inputs``; echoed back so the parent can
        attribute completion under :func:`multiprocessing.pool.Pool.imap_unordered`.
    events : list[ShadowingEvent]
        Events detected for the single RPO.
    """
    with _attach_trajectory(
        inputs.modes_shm_name,
        inputs.modes_shape,
        inputs.trajectory_dt,
        inputs.resolution,
    ) as trajectory:
        events = _extract_shadowing_events(
            _compute_distances_sq(trajectory, inputs.rpo, inputs.rpo_trajectory, inputs.chunk_size),
            inputs.rpo.index,
            inputs.rpo.time_steps,
            inputs.resolution,
            inputs.threshold,
            inputs.min_duration,
        )
    return inputs.rpo.index, events


def _min_distances_single_rpo(
    inputs: _MinDistanceWorkerInputs,
) -> tuple[int, NDArray[np.float64]]:
    """Compute per-timestep minimum :math:`L_2` distances for a single RPO,
    worker-side.

    Attaches a :class:`~ks_shadowing.core.trajectory.KSTrajectory` view over
    the shared-memory trajectory modes, streams squared distances, and reduces
    each chunk to a per-timestep minimum over all phases and shifts.

    Parameters
    ----------
    inputs : :class:`_MinDistanceWorkerInputs`
        Shared-memory handle, trajectory metadata, and RPO.

    Returns
    -------
    rpo_index : int
        Index of the RPO supplied in ``inputs``; echoed back so the parent can
        attribute completion under :func:`multiprocessing.pool.Pool.imap_unordered`.
    min_distances : NDArray[np.float64], shape (num_timesteps,)
        Minimum :math:`L_2` distance to this RPO at each trajectory timestep.
    """
    with _attach_trajectory(
        inputs.modes_shm_name,
        inputs.modes_shape,
        inputs.trajectory_dt,
        inputs.resolution,
    ) as trajectory:
        min_distances_sq = np.full(len(trajectory), np.inf, dtype=np.float64)
        for _, chunk_start, dist_sq in _compute_distances_sq(
            trajectory, inputs.rpo, inputs.rpo_trajectory, inputs.chunk_size
        ):
            chunk_end = chunk_start + dist_sq.shape[0]
            phase_min_sq = np.min(dist_sq, axis=1)
            np.minimum(
                min_distances_sq[chunk_start:chunk_end],
                phase_min_sq,
                out=min_distances_sq[chunk_start:chunk_end],
            )
    return inputs.rpo.index, np.sqrt(min_distances_sq)


def _compute_distances_sq(
    trajectory: KSTrajectory,
    rpo: RPO,
    rpo_trajectory: KSTrajectory,
    chunk_size: int,
) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
    """Yield squared :math:`L_2` distance arrays for each phase offset in the
    RPO's co-moving frame.

    Both trajectory and RPO are transformed to the RPO's co-moving frame where
    the RPO becomes truly periodic. The trajectory is processed in chunks via
    :meth:`~ks_shadowing.core.trajectory.KSTrajectory.chunks_fourier` to control
    peak memory usage. RPO phase offsets are produced by modular fancy indexing
    into the one-period co-moving RPO, avoiding any full-length tile.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory to compare against the RPO.
    rpo : :class:`~ks_shadowing.core.rpo.RPO`
        Source RPO metadata.
    rpo_trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Integrated RPO trajectory; its number of rows is the cycle length used
        as ``period`` for modular phase indexing.
    chunk_size : int
        Maximum number of trajectory timesteps per chunk.

    Yields
    ------
    phase : int
        RPO phase offset index in ``range(period)``.
    chunk_start : int
        First trajectory timestep index in this chunk.
    dist_sq : NDArray[np.float64], shape (chunk_len, resolution)
        Squared :math:`L_2` distances at each ``(timestep, shift)`` within the
        chunk, clamped at zero to absorb numerical noise.
    """
    period = rpo_trajectory.num_timesteps
    traj_comoving = trajectory.to_comoving(rpo.drift_rate)
    rpo_comoving_modes = rpo_trajectory.to_comoving(rpo.drift_rate).modes

    for chunk_start, chunk_modes in traj_comoving.chunks_fourier(chunk_size):
        chunk_len = chunk_modes.shape[0]
        arange_chunk = np.arange(chunk_len)
        for phase in range(period):
            indices = (chunk_start + phase + arange_chunk) % period
            rpo_slice = rpo_comoving_modes[indices]
            dist_sq = shift_distances_sq(chunk_modes, rpo_slice, rpo_trajectory.resolution)
            yield phase, chunk_start, np.maximum(dist_sq, 0.0)
