"""Persistent Homology Approach (PHA) shadowing detection.

Detects shadowing between a Kuramoto-Sivashinsky trajectory and a collection of
relative periodic orbits (RPOs) by comparing persistence diagrams. Each point
of the trajectory and each phase of each RPO is reduced to a zeroth persistence
diagram of its physical-space field; Wasserstein distances between these
diagrams populate a 2D ``(trajectory timestep, RPO phase)`` grid per RPO.
Event detection over this grid is delegated to
:mod:`~ks_shadowing.pha.pathfinding`.

The continuous spatial symmetry along the periodic domain is quotiented out by
the persistence representation, so the detection grid has no shift dimension.
Optimal shifts are reconstructed post-hoc in :mod:`~ks_shadowing.pha.shifts`.

A tunable time-delay embedding is applied to each distance matrix, summing
:math:`w` consecutive Wasserstein distances to increase the effective
dimensionality of the comparison. Wasserstein distances are computed via the
`Hera <https://github.com/anigmetov/hera>`_ C++ library through custom batched
bindings; this is the dominant cost. Trajectory persistence diagrams are
flattened once into :class:`~multiprocessing.shared_memory.SharedMemory` so
worker processes attach to the shared buffer instead of receiving a copy per
task.
"""

from collections.abc import Iterable, Iterator, Sequence
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
from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha.pathfinding import _extract_shadowing_events
from ks_shadowing.pha.persistence import _KSPersistenceTrajectory
from ks_shadowing.pha.shifts import _compute_event_shifts
from ks_shadowing.pha.wasserstein import _wasserstein_column


def detect(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    delay: int,
    threshold: float,
    min_duration: int = 1,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ShadowingEvent]:
    """Detect shadowing events between ``trajectory`` and ``rpos``.

    Computes zeroth persistence diagrams for every trajectory timestep and every
    RPO phase, then, for each RPO, builds the ``(num_timesteps, period)``
    Wasserstein distance matrix, applies a time-delay embedding of window
    ``delay``, and extracts shadowing events. Spatial shifts for closest
    physical-space shadowing are reconstructed post-hoc with
    :func:`~ks_shadowing.pha.shifts._compute_event_shifts`.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory to scan for shadowing events.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to shadow against. Each RPO is integrated at
        its own native timestep to preserve numerical accuracy.
    delay : int
        Time-delay embedding window size. Comparison with SSA is useful to tune
        this parameter.
    threshold : float
        Maximum delay-embedded Wasserstein distance for a grid entry to count
        as a close pass. Typically set by quantile with
        :func:`~ks_shadowing.pha.detection.compute_min_distances`. This flow
        is automated via :func:`~ks_shadowing.pha.detection.auto_detect`.
    min_duration : int, optional
        Minimum event duration in timesteps. Default is 1.
    show_progress : bool, optional
        Whether to display ``tqdm`` progress bars. Default is ``False``.
    n_jobs : int, optional
        Number of parallel workers for the Wasserstein sweep. ``-1`` uses all
        available CPUs. Default is 1.
    chunk_size : int, optional
        Maximum number of trajectory timesteps materialized in physical space
        at once when computing persistence diagrams. Default is
        :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    list[ShadowingEvent]
        Events sorted by ``(start_timestep, rpo_index)``.
    """
    trajectory_diagrams = _KSPersistenceTrajectory.from_trajectory(trajectory, chunk_size)
    rpo_diagram_pairs = _compute_rpo_diagram_pairs(rpos, trajectory.resolution, chunk_size)
    n_workers = _resolve_n_jobs(n_jobs)

    events = _detect_from_diagrams(
        trajectory_diagrams,
        rpo_diagram_pairs,
        delay,
        threshold,
        min_duration,
        show_progress,
        n_workers,
    )
    return _attach_shifts(events, trajectory, rpos)


def compute_min_distances(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    delay: int,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> NDArray[np.float64]:
    """Compute the minimum delay-embedded Wasserstein distance to any phase of
    any RPO at each trajectory timestep.

    Useful for shadowing threshold selection by quantile; see
    :func:`auto_detect`.

    Due to the time-delay embedding, the final ``delay - 1`` timesteps have
    insufficient future data and are returned as infinity.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory whose minimum distances are computed.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to shadow against. Each RPO is integrated at
        its own native timestep to preserve numerical accuracy.
    delay : int
        Time-delay embedding window size. Comparison with SSA is useful to tune
        this parameter.
    show_progress : bool, optional
        Whether to display ``tqdm`` progress bars. Default is ``False``.
    n_jobs : int, optional
        Number of parallel workers for the Wasserstein sweep. ``-1`` uses all
        available CPUs. Default is 1.
    chunk_size : int, optional
        Maximum number of trajectory timesteps materialized in physical space
        at once when computing persistence diagrams. Default is
        :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    NDArray[np.float64], shape (num_timesteps,)
        Minimum Wasserstein distance at each trajectory timestep.
    """
    trajectory_diagrams = _KSPersistenceTrajectory.from_trajectory(trajectory, chunk_size)
    rpo_diagram_pairs = _compute_rpo_diagram_pairs(rpos, trajectory.resolution, chunk_size)
    n_workers = _resolve_n_jobs(n_jobs)

    return _min_distances_from_diagrams(
        trajectory_diagrams, rpo_diagram_pairs, delay, show_progress, n_workers
    )


def auto_detect(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    delay: int,
    threshold_quantile: float = 0.4,
    min_duration: int = 1,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[ShadowingEvent], float]:
    """Detect shadowing events with a threshold chosen automatically from the
    distribution of per-timestep minimum Wasserstein distances.

    The threshold is set to the ``threshold_quantile`` quantile of per-timestep
    minimal delay-embedded Wasserstein distances. For example,
    ``threshold_quantile=0.4`` selects a threshold such that roughly 40% of
    trajectory timesteps have a minimum Wasserstein distance to some RPO phase
    below the threshold.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory to scan for shadowing events.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to shadow against. Each RPO is integrated at
        its own native timestep to preserve numerical accuracy.
    delay : int
        Time-delay embedding window size. Comparison with SSA is useful to tune
        this parameter.
    threshold_quantile : float, optional
        Quantile of per-timestep minimum distances used as the detection
        threshold. Default is 0.4.
    min_duration : int, optional
        Minimum event duration in timesteps. Default is 1.
    show_progress : bool, optional
        Whether to display ``tqdm`` progress bars. Default is ``False``.
    n_jobs : int, optional
        Number of parallel workers for the Wasserstein sweep. ``-1`` uses all
        available CPUs. Default is 1.
    chunk_size : int, optional
        Maximum number of trajectory timesteps materialized in physical space
        at once when computing persistence diagrams. Default is
        :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    events : list[ShadowingEvent]
        Detected events sorted by ``(start_timestep, rpo_index)``.
    threshold : float
        The automatically selected threshold.
    """
    trajectory_diagrams = _KSPersistenceTrajectory.from_trajectory(trajectory, chunk_size)
    rpo_diagram_pairs = _compute_rpo_diagram_pairs(rpos, trajectory.resolution, chunk_size)
    n_workers = _resolve_n_jobs(n_jobs)

    min_distances = _min_distances_from_diagrams(
        trajectory_diagrams, rpo_diagram_pairs, delay, show_progress, n_workers
    )
    finite_distances = min_distances[np.isfinite(min_distances)]
    threshold = float(np.quantile(finite_distances, threshold_quantile))

    events = _detect_from_diagrams(
        trajectory_diagrams,
        rpo_diagram_pairs,
        delay,
        threshold,
        min_duration,
        show_progress,
        n_workers,
    )
    events = _attach_shifts(events, trajectory, rpos)
    return events, threshold


def _compute_rpo_diagram_pairs(
    rpos: Sequence[RPO],
    resolution: int,
    chunk_size: int,
) -> list[tuple[RPO, _KSPersistenceTrajectory]]:
    """Integrate each RPO and compute persistence diagrams of each phase.

    Returned pairs are sorted by RPO period descending so that the
    longest-running RPOs are dispatched first.
    """
    diagram_pairs: list[tuple[RPO, _KSPersistenceTrajectory]] = []
    for rpo in rpos:
        rpo_trajectory = KSTrajectory.from_initial_state(
            rpo.modes, rpo.dt, rpo.time_steps, resolution
        )
        phase_diagrams = _KSPersistenceTrajectory.from_trajectory(rpo_trajectory, chunk_size)
        diagram_pairs.append((rpo, phase_diagrams))

    diagram_pairs.sort(key=lambda pair: pair[0].time_steps, reverse=True)
    return diagram_pairs


def _detect_from_diagrams(  # noqa: PLR0913
    trajectory_diagrams: _KSPersistenceTrajectory,
    rpo_diagram_pairs: list[tuple[RPO, _KSPersistenceTrajectory]],
    delay: int,
    threshold: float,
    min_duration: int,
    show_progress: bool,
    n_workers: int,
) -> list[ShadowingEvent]:
    """Run detection given a trajectory and RPO diagrams in persistence form."""
    events: list[ShadowingEvent] = []
    for rpo_index, distance_matrix in _stream_distance_matrices(
        trajectory_diagrams, rpo_diagram_pairs, delay, show_progress, n_workers
    ):
        events.extend(
            _extract_shadowing_events(distance_matrix, rpo_index, threshold, min_duration)
        )
    return events


def _min_distances_from_diagrams(
    trajectory_diagrams: _KSPersistenceTrajectory,
    rpo_diagram_pairs: list[tuple[RPO, _KSPersistenceTrajectory]],
    delay: int,
    show_progress: bool,
    n_workers: int,
) -> NDArray[np.float64]:
    """Compute per-timestep minimum distances given a trajectory and RPO
    diagrams in persistence form.

    Timesteps for which no RPO supplies a finite distance (always the final
    ``delay - 1`` timesteps due to the embedding) are returned as infinity.
    """
    num_timesteps = len(trajectory_diagrams)
    min_distances = np.full(num_timesteps, np.inf, dtype=np.float64)

    for _, distance_matrix in _stream_distance_matrices(
        trajectory_diagrams, rpo_diagram_pairs, delay, show_progress, n_workers
    ):
        embedded_length = distance_matrix.shape[0]
        rpo_column_min = distance_matrix.min(axis=1)
        min_distances[:embedded_length] = np.minimum(
            min_distances[:embedded_length], rpo_column_min
        )

    return min_distances


def _stream_distance_matrices(
    trajectory_diagrams: _KSPersistenceTrajectory,
    rpo_diagram_pairs: list[tuple[RPO, _KSPersistenceTrajectory]],
    delay: int,
    show_progress: bool,
    n_workers: int,
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Yield ``(rpo_index, delay-embedded distance matrix)`` for each RPO.

    Parameters
    ----------
    trajectory_diagrams : :class:`~ks_shadowing.pha.persistence._KSPersistenceTrajectory`
        Zeroth persistence diagrams of each trajectory timestep.
    rpo_diagram_pairs : list[tuple[RPO, _KSPersistenceTrajectory]]
        ``(rpo, phase_diagrams)`` pairs produced by
        :func:`_compute_rpo_diagram_pairs`.
    delay : int
        Time-delay embedding window size.
    show_progress : bool
        Whether to display ``tqdm`` progress bars for the outer RPO loop and
        the inner phase sweep.
    n_workers : int
        Number of worker processes.

    Yields
    ------
    rpo_index : int
        Index of the current RPO.
    distance_matrix : NDArray[np.float64], shape (num_timesteps - delay + 1, period)
        Delay-embedded Wasserstein distance matrix for the current RPO.
    """
    flat_diagrams, offsets = trajectory_diagrams._flatten()
    num_timesteps = len(trajectory_diagrams)

    # Per-RPO cost (one Wasserstein column per phase) scales linearly in
    # ``rpo.time_steps``, so weighting the outer bar by phase count gives a
    # percent-complete that tracks actual work rather than RPO count.
    if n_workers == 1:
        with tqdm(
            total=sum(rpo.time_steps for rpo, _ in rpo_diagram_pairs),
            desc="Detecting",
            unit="phase",
            disable=not show_progress,
        ) as outer_bar:
            for rpo, phase_diagrams in rpo_diagram_pairs:
                num_phases = len(phase_diagrams)
                wasserstein_matrix = np.empty((num_timesteps, num_phases), dtype=np.float64)

                column_inputs: Iterable[tuple[int, NDArray[np.float64]]] = enumerate(phase_diagrams)
                if show_progress:
                    column_inputs = tqdm(
                        column_inputs,
                        total=num_phases,
                        desc="  Phases",
                        leave=False,
                    )

                for phase_index, diagram in column_inputs:
                    wasserstein_matrix[:, phase_index] = _wasserstein_column(
                        flat_diagrams, offsets, diagram
                    )

                yield rpo.index, _apply_delay_embedding(wasserstein_matrix, delay)
                outer_bar.update(rpo.time_steps)
        return

    with (
        _shared_memory_view(flat_diagrams) as diagrams_shm,
        _shared_memory_view(offsets) as offsets_shm,
        _forkserver_pool(n_workers) as pool,
        tqdm(
            total=sum(rpo.time_steps for rpo, _ in rpo_diagram_pairs),
            desc="Detecting",
            unit="phase",
            disable=not show_progress,
        ) as outer_bar,
    ):
        for rpo, phase_diagrams in rpo_diagram_pairs:
            num_phases = len(phase_diagrams)
            wasserstein_matrix = np.empty((num_timesteps, num_phases), dtype=np.float64)

            column_inputs = [
                _WassersteinColumnInputs(
                    phase_index=phase_index,
                    diagrams_shm_name=diagrams_shm.name,
                    offsets_shm_name=offsets_shm.name,
                    num_timesteps=num_timesteps,
                    rpo_diagram=diagram,
                )
                for phase_index, diagram in enumerate(phase_diagrams)
            ]
            column_results: Iterable[tuple[int, NDArray[np.float64]]] = pool.imap_unordered(
                _compute_wasserstein_column, column_inputs
            )
            if show_progress:
                column_results = tqdm(
                    column_results, total=num_phases, desc="  Phases", leave=False
                )

            for phase_index, column in column_results:
                wasserstein_matrix[:, phase_index] = column

            yield rpo.index, _apply_delay_embedding(wasserstein_matrix, delay)
            outer_bar.update(rpo.time_steps)


class _WassersteinColumnInputs(NamedTuple):
    """Inputs to :func:`_compute_wasserstein_column` for one RPO phase.

    Attributes
    ----------
    phase_index : int
        RPO phase index; echoed back in the worker's return value so the
        dispatcher can reassemble the distance matrix from unordered results.
    diagrams_shm_name : str
        Name of the :class:`~multiprocessing.shared_memory.SharedMemory` block
        holding the flattened trajectory persistence diagrams.
    offsets_shm_name : str
        Name of the :class:`~multiprocessing.shared_memory.SharedMemory` block
        holding trajectory diagram offsets.
    num_timesteps : int
        Number of timesteps in the trajectory in shared memory.
    rpo_diagram : NDArray[np.float64]
        Persistence diagram of the RPO phase, against which each trajectory
        diagram is compared.
    """

    phase_index: int
    diagrams_shm_name: str
    offsets_shm_name: str
    num_timesteps: int
    rpo_diagram: NDArray[np.float64]


def _compute_wasserstein_column(
    inputs: _WassersteinColumnInputs,
) -> tuple[int, NDArray[np.float64]]:
    """Compute one column of a Wasserstein distance matrix, worker-side.

    Opens the two :class:`~multiprocessing.shared_memory.SharedMemory` blocks
    holding the flattened trajectory persistence diagrams, then delegates to
    :func:`~ks_shadowing.pha.wasserstein._wasserstein_column`.

    Parameters
    ----------
    inputs : :class:`_WassersteinColumnInputs`
        Phase index, shared-memory handles, and the RPO phase diagram.

    Returns
    -------
    phase_index : int
        The phase index supplied in ``inputs``.
    column : NDArray[np.float64], shape (num_timesteps,)
        Wasserstein distance from each trajectory diagram to
        ``inputs.rpo_diagram``.
    """
    diagrams_shm = SharedMemory(name=inputs.diagrams_shm_name)
    offsets_shm = SharedMemory(name=inputs.offsets_shm_name)
    try:
        offsets = np.ndarray((inputs.num_timesteps + 1,), dtype=np.int64, buffer=offsets_shm.buf)
        trajectory_diagrams = np.ndarray(
            (int(offsets[-1]), 2), dtype=np.float64, buffer=diagrams_shm.buf
        )
        column = _wasserstein_column(trajectory_diagrams, offsets, inputs.rpo_diagram)
    finally:
        diagrams_shm.close()
        offsets_shm.close()

    return inputs.phase_index, column


def _apply_delay_embedding(
    wasserstein_matrix: NDArray[np.float64],
    delay: int,
) -> NDArray[np.float64]:
    r"""Apply time-delay embedding to a Wasserstein distance matrix.

    Computes :math:`W^w(i, j) = \sum_{l=0}^{w-1} W(i+l, (j+l) \bmod J)` where
    :math:`w` is the delay window. This increases the effective dimensionality
    of the comparison by considering consecutive timesteps rather than single
    snapshots.

    Parameters
    ----------
    wasserstein_matrix : NDArray[np.float64], shape (I, J)
        Original Wasserstein distance matrix.
    delay : int
        Time-delay embedding window size (:math:`w`).

    Returns
    -------
    NDArray[np.float64], shape (I - delay + 1, J)
        Embedded distance matrix.

    Raises
    ------
    ValueError
        If ``delay < 1`` or ``delay`` exceeds the trajectory length.
    """
    trajectory_timesteps, rpo_timesteps = wasserstein_matrix.shape

    if delay < 1:
        raise ValueError(f"delay must be at least 1, got {delay}")
    if delay > trajectory_timesteps:
        raise ValueError(f"delay ({delay}) exceeds trajectory length ({trajectory_timesteps})")

    delayed_timesteps = trajectory_timesteps - delay + 1
    delayed = np.zeros((delayed_timesteps, rpo_timesteps), dtype=np.float64)

    for offset in range(delay):
        # At offset l: trajectory index is (i + l), RPO index is (j + l) % J
        col_indices = (np.arange(rpo_timesteps) + offset) % rpo_timesteps
        delayed += wasserstein_matrix[offset : offset + delayed_timesteps][:, col_indices]

    return delayed


def _attach_shifts(
    events: list[ShadowingEvent],
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
) -> list[ShadowingEvent]:
    """Reconstruct spatial shifts for each event and sort the result.

    PHA quotients out spatial shifts during detection, leaving events with
    zero-filled ``shifts`` arrays. For each RPO shadowed by at least one event,
    integrates the RPO once and transforms to its co-moving frame, then passes
    this co-moving trajectory to each event's shift reconstruction in
    :func:`~ks_shadowing.pha.shifts._compute_event_shifts`. Sorts the result
    by ``(start_timestep, rpo_index)``.
    """
    rpo_by_index = {rpo.index: rpo for rpo in rpos}

    events_by_rpo: dict[int, list[ShadowingEvent]] = {}
    for event in events:
        events_by_rpo.setdefault(event.rpo_index, []).append(event)

    events_with_shifts: list[ShadowingEvent] = []
    for rpo_index, rpo_events in events_by_rpo.items():
        rpo = rpo_by_index[rpo_index]
        rpo_trajectory = KSTrajectory.from_initial_state(
            rpo.modes, rpo.dt, rpo.time_steps, trajectory.resolution
        )
        drift_rate = rpo.spatial_shift / rpo.time_steps
        rpo_comoving = rpo_trajectory.to_comoving(drift_rate)

        for event in rpo_events:
            events_with_shifts.append(
                _compute_event_shifts(event, trajectory, drift_rate, rpo_comoving)
            )

    events_with_shifts.sort(key=lambda event: (event.start_timestep, event.rpo_index))
    return events_with_shifts
