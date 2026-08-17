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

Two orthogonal embedding stages thicken the comparison.
``max_derivative_order`` enriches snapshots spatially: persistence diagrams of
the spatial-derivative fields of orders ``0..max_derivative_order`` (where
order 0 is the field itself) are computed and the per-order Wasserstein
distances are averaged into the ``(timestep, phase)`` score. The optional
``rescale_orders`` flag divides each order's distances by a per-order median
scale before that mean. ``delay`` controls
temporal aggregation: ``delay`` consecutive Wasserstein distances along the
diagonal of the per-RPO matrix are averaged, attributed to the window's center
timestep. ``max_derivative_order`` defaults
to ``0`` and ``delay`` to ``1`` (no embedding); they compose freely.

Wasserstein distances are computed via the
`Hera <https://github.com/anigmetov/hera>`_ C++ library through custom batched
bindings; this is the dominant PHA cost. Trajectory persistence diagrams are
flattened once per order into
:class:`~multiprocessing.shared_memory.SharedMemory` so worker processes attach
to the shared buffers instead of receiving a copy per task.
"""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from multiprocessing.shared_memory import SharedMemory

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
from ks_shadowing.core.results import DetectionResult
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory, _resolve_rpo_downsample
from ks_shadowing.pha.pathfinding import _extract_shadowing_events
from ks_shadowing.pha.persistence import KSPersistenceTrajectory
from ks_shadowing.pha.shifts import _compute_event_shifts
from ks_shadowing.pha.wasserstein import _wasserstein_column

_SCALE_SAMPLE_TIMESTEPS = 64
_SCALE_SAMPLE_PHASES_PER_RPO = 8


def _compute_order_scales(
    trajectory_diagrams_per_order: list[KSPersistenceTrajectory],
    rpo_diagram_pairs: list[tuple[RPO, list[KSPersistenceTrajectory]]],
) -> NDArray[np.float64]:
    """Estimate one Wasserstein scale per derivative order from a subsample.

    For each derivative order, computes the median Wasserstein distance over a
    strided subsample of trajectory timesteps against a pooled subsample of RPO
    phases. The scales are global across RPOs: per-RPO scales would rescale each
    RPO's distances differently and distort the min over RPOs.

    Parameters
    ----------
    trajectory_diagrams_per_order : list[KSPersistenceTrajectory]
        Trajectory persistence diagrams, one sequence per derivative order.
    rpo_diagram_pairs : list[tuple[RPO, list[KSPersistenceTrajectory]]]
        ``(rpo, per_order_phase_diagrams)`` pairs from
        ``_compute_rpo_diagram_pairs``.

    Returns
    -------
    NDArray[np.float64], shape (num_orders,)
        Per-order median scale.

    Raises
    ------
    ValueError
        If ``rpo_diagram_pairs`` is empty, or if any order's median scale is
        non-positive or non-finite.
    """
    if not rpo_diagram_pairs:
        raise ValueError("rescale_orders requires at least one RPO to estimate order scales")

    num_orders = len(trajectory_diagrams_per_order)
    num_timesteps = len(trajectory_diagrams_per_order[0])

    timestep_sample = np.unique(
        np.linspace(0, num_timesteps - 1, min(_SCALE_SAMPLE_TIMESTEPS, num_timesteps)).astype(
            np.int64
        )
    )

    scales = np.empty(num_orders, dtype=np.float64)
    for order_index in range(num_orders):
        sampled_trajectory = KSPersistenceTrajectory(
            diagrams=[
                trajectory_diagrams_per_order[order_index].diagrams[t] for t in timestep_sample
            ],
            dt=trajectory_diagrams_per_order[order_index].dt,
        )
        flat_diagrams, offsets = sampled_trajectory._flatten()

        pooled_values: list[NDArray[np.float64]] = []
        for _, phase_diagrams_per_order in rpo_diagram_pairs:
            phase_diagrams = phase_diagrams_per_order[order_index]
            num_phases = len(phase_diagrams)
            phase_sample = np.unique(
                np.linspace(
                    0, num_phases - 1, min(_SCALE_SAMPLE_PHASES_PER_RPO, num_phases)
                ).astype(np.int64)
            )
            for phase_index in phase_sample:
                pooled_values.append(
                    _wasserstein_column(
                        flat_diagrams, offsets, phase_diagrams.diagrams[phase_index]
                    )
                )

        median = float(np.median(np.concatenate(pooled_values)))
        if not np.isfinite(median) or median <= 0.0:
            raise ValueError(
                f"order {order_index} produced a non-positive or non-finite median "
                f"scale ({median}); cannot rescale"
            )
        scales[order_index] = median

    return scales


def _prepare_diagrams_and_scales(  # noqa: PLR0913, PLR0917
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    delay: int,
    max_derivative_order: int,
    rescale_orders: bool,
    native: bool,
    chunk_size: int,
) -> tuple[
    list[KSPersistenceTrajectory],
    list[tuple[RPO, list[KSPersistenceTrajectory]]],
    NDArray[np.float64],
]:
    """Validate embedding parameters and build diagrams and per-order scales.

    Shared setup for :func:`detect`, :func:`compute_min_distances`, and
    :func:`auto_detect`.

    Raises
    ------
    ValueError
        If ``delay`` is less than 1, ``delay`` exceeds the trajectory length,
        ``max_derivative_order`` is negative, or ``trajectory.dt`` is not an
        integer multiple of some RPO's native timestep.
    """
    if delay < 1:
        raise ValueError(f"delay must be at least 1, got {delay}")
    if delay > len(trajectory):
        raise ValueError(f"delay ({delay}) exceeds trajectory length ({len(trajectory)})")
    if max_derivative_order < 0:
        raise ValueError(f"max_derivative_order must be non-negative, got {max_derivative_order}")

    trajectory_diagrams_per_order = [
        KSPersistenceTrajectory.from_trajectory(trajectory, chunk_size, order=order)
        for order in range(max_derivative_order + 1)
    ]
    rpo_diagram_pairs = _compute_rpo_diagram_pairs(
        rpos,
        trajectory.resolution,
        chunk_size,
        native,
        max_derivative_order,
        trajectory.dt,
    )

    if rescale_orders:
        order_scales = _compute_order_scales(trajectory_diagrams_per_order, rpo_diagram_pairs)
    else:
        order_scales = np.ones(max_derivative_order + 1, dtype=np.float64)

    return trajectory_diagrams_per_order, rpo_diagram_pairs, order_scales


def detect(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    threshold: float,
    *,
    delay: int = 1,
    max_derivative_order: int = 0,
    rescale_orders: bool = False,
    native: bool = False,
    min_duration: int = 1,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DetectionResult:
    """Detect shadowing events between ``trajectory`` and ``rpos``.

    Computes zeroth persistence diagrams for every trajectory timestep and every
    RPO phase, then, for each RPO, builds the ``(num_timesteps, period)``
    Wasserstein distance matrix, applies a time-delay embedding of window
    ``delay``, and extracts shadowing events. Spatial shifts for closest
    physical-space shadowing are reconstructed post-hoc with
    ``_compute_event_shifts``.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory to scan for shadowing events.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to shadow against. Each RPO's per-RPO
        trajectory is built at a downsample stride inferred from
        ``trajectory.dt`` and the RPO's own native timestep.
    threshold : float
        Grid entries with delay-embedded Wasserstein distance strictly below
        ``threshold`` count as close passes. Typically set by quantile with
        :func:`~ks_shadowing.pha.detection.compute_min_distances`. This flow
        is automated via :func:`~ks_shadowing.pha.detection.auto_detect`.
    delay : int, optional
        Time-delay embedding window size. Mean Wasserstein distance is taken
        across ``delay`` consecutive timesteps, attributed to the window's
        center timestep and phase (offset ``(delay - 1) // 2`` from the window
        start). ``1`` (default) applies no temporal embedding. Events cannot
        start before timestep ``(delay - 1) // 2`` nor extend past
        ``num_timesteps - delay // 2``.
    max_derivative_order : int, optional
        Highest spatial-derivative order to include in the persistence-diagram
        comparison. Persistence diagrams of orders ``0..max_derivative_order``
        are computed and the per-order Wasserstein distances are averaged into
        the ``(timestep, phase)`` score. ``0`` (default) uses only the field
        itself.
    rescale_orders : bool, optional
        When ``True``, divides each derivative order's Wasserstein matrix by a
        per-order median scale estimated from a deterministic subsample before
        averaging across orders; distances become dimensionless multiples of the
        per-order scale, so a manual ``threshold`` is not comparable with runs
        using the other setting. Default ``False`` (no rescaling).
    native : bool, optional
        If ``True``, build per-RPO trajectories by reordering native rows
        with the inferred-stride permutation; if ``False``, by slicing every
        stride-th native row. Default ``False``. See
        :meth:`~ks_shadowing.core.trajectory.KSTrajectory.from_rpo`.
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
    DetectionResult
        ``events`` sorted by ``(start_timestep, rpo_index)`` and ``threshold``
        echoing the input ``threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is negative, ``delay`` is less than 1, ``delay``
        exceeds the trajectory length, ``max_derivative_order`` is negative,
        or ``trajectory.dt`` is not an integer multiple of some RPO's native
        timestep.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    trajectory_diagrams_per_order, rpo_diagram_pairs, order_scales = _prepare_diagrams_and_scales(
        trajectory, rpos, delay, max_derivative_order, rescale_orders, native, chunk_size
    )
    n_workers = _resolve_n_jobs(n_jobs)

    events = _detect_from_diagrams(
        trajectory_diagrams_per_order,
        rpo_diagram_pairs,
        delay,
        threshold,
        min_duration,
        show_progress,
        n_workers,
        order_scales,
    )
    events = _attach_shifts(events, trajectory, rpos, native)
    return DetectionResult(
        events=events,
        threshold=threshold,
        order_scales=order_scales if rescale_orders else None,
    )


def compute_min_distances(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    *,
    delay: int = 1,
    max_derivative_order: int = 0,
    rescale_orders: bool = False,
    native: bool = False,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> NDArray[np.float64]:
    """Compute the minimum delay-embedded Wasserstein distance to any phase of
    any RPO at each trajectory timestep.

    Useful for shadowing threshold selection by quantile; see
    :func:`auto_detect`.

    Due to the time-delay embedding, the leading ``(delay - 1) // 2`` and
    trailing ``delay // 2`` timesteps have no centered window and are
    returned as infinity.

    Parameters
    ----------
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Trajectory whose minimum distances are computed.
    rpos : Sequence[:class:`~ks_shadowing.core.rpo.RPO`]
        Relative periodic orbits to shadow against. Each RPO's per-RPO
        trajectory is built at a downsample stride inferred from
        ``trajectory.dt`` and the RPO's own native timestep.
    delay : int, optional
        Time-delay embedding window size. Mean Wasserstein distance is
        taken across ``delay`` consecutive timesteps, attributed to the window's
        center timestep. ``1`` (default) applies no temporal embedding.
    max_derivative_order : int, optional
        Highest spatial-derivative order to include in the persistence-diagram
        comparison. Persistence diagrams of orders ``0..max_derivative_order``
        are computed and the per-order Wasserstein distances are averaged into
        the ``(timestep, phase)`` score. ``0`` (default) uses only the field
        itself.
    rescale_orders : bool, optional
        When ``True``, divides each derivative order's Wasserstein matrix by a
        per-order median scale estimated from a deterministic subsample before
        averaging across orders; distances become dimensionless multiples of the
        per-order scale, so a manual ``threshold`` is not comparable with runs
        using the other setting. Default ``False`` (no rescaling).
    native : bool, optional
        If ``True``, build per-RPO trajectories by reordering native rows
        with the inferred-stride permutation; if ``False``, by slicing every
        stride-th native row. Default ``False``. See
        :meth:`~ks_shadowing.core.trajectory.KSTrajectory.from_rpo`.
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

    Raises
    ------
    ValueError
        If ``delay`` is less than 1, ``delay`` exceeds the trajectory length,
        ``max_derivative_order`` is negative, or ``trajectory.dt`` is not an
        integer multiple of some RPO's native timestep.
    """
    trajectory_diagrams_per_order, rpo_diagram_pairs, order_scales = _prepare_diagrams_and_scales(
        trajectory, rpos, delay, max_derivative_order, rescale_orders, native, chunk_size
    )
    n_workers = _resolve_n_jobs(n_jobs)

    return _min_distances_from_diagrams(
        trajectory_diagrams_per_order,
        rpo_diagram_pairs,
        delay,
        show_progress,
        n_workers,
        order_scales,
    )


def auto_detect(  # noqa: PLR0913
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    threshold_quantile: float = 0.4,
    *,
    delay: int = 1,
    max_derivative_order: int = 0,
    rescale_orders: bool = False,
    native: bool = False,
    min_duration: int = 1,
    show_progress: bool = False,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DetectionResult:
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
        Relative periodic orbits to shadow against. Each RPO's per-RPO
        trajectory is built at a downsample stride inferred from
        ``trajectory.dt`` and the RPO's own native timestep.
    threshold_quantile : float, optional
        Quantile of per-timestep minimum distances used as the detection
        threshold. Default is 0.4.
    delay : int, optional
        Time-delay embedding window size. Mean Wasserstein distance is
        taken across ``delay`` consecutive timesteps, attributed to the window's
        center timestep. ``1`` (default) applies no temporal embedding.
    max_derivative_order : int, optional
        Highest spatial-derivative order to include in the persistence-diagram
        comparison. Persistence diagrams of orders ``0..max_derivative_order``
        are computed and the per-order Wasserstein distances are averaged into
        the ``(timestep, phase)`` score. ``0`` (default) uses only the field
        itself.
    rescale_orders : bool, optional
        When ``True``, divides each derivative order's Wasserstein matrix by a
        per-order median scale estimated from a deterministic subsample before
        averaging across orders; distances become dimensionless multiples of the
        per-order scale, so a manual ``threshold`` is not comparable with runs
        using the other setting. Default ``False`` (no rescaling).
    native : bool, optional
        If ``True``, build per-RPO trajectories by reordering native rows
        with the inferred-stride permutation; if ``False``, by slicing every
        stride-th native row. Default ``False``. See
        :meth:`~ks_shadowing.core.trajectory.KSTrajectory.from_rpo`.
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
    DetectionResult
        ``events`` sorted by ``(start_timestep, rpo_index)`` and ``threshold``
        set to the automatically selected quantile value.

    Raises
    ------
    ValueError
        If ``delay`` is less than 1, ``delay`` exceeds the trajectory length,
        ``max_derivative_order`` is negative, or ``trajectory.dt`` is not an
        integer multiple of some RPO's native timestep.
    """
    trajectory_diagrams_per_order, rpo_diagram_pairs, order_scales = _prepare_diagrams_and_scales(
        trajectory, rpos, delay, max_derivative_order, rescale_orders, native, chunk_size
    )
    n_workers = _resolve_n_jobs(n_jobs)

    min_distances = _min_distances_from_diagrams(
        trajectory_diagrams_per_order,
        rpo_diagram_pairs,
        delay,
        show_progress,
        n_workers,
        order_scales,
    )
    finite_distances = min_distances[np.isfinite(min_distances)]
    threshold = float(np.quantile(finite_distances, threshold_quantile))

    events = _detect_from_diagrams(
        trajectory_diagrams_per_order,
        rpo_diagram_pairs,
        delay,
        threshold,
        min_duration,
        show_progress,
        n_workers,
        order_scales,
    )
    events = _attach_shifts(events, trajectory, rpos, native)
    return DetectionResult(
        events=events,
        threshold=threshold,
        order_scales=order_scales if rescale_orders else None,
    )


def _compute_rpo_diagram_pairs(  # noqa: PLR0913, PLR0917
    rpos: Sequence[RPO],
    resolution: int,
    chunk_size: int,
    native: bool,
    max_derivative_order: int,
    trajectory_dt: float,
) -> list[tuple[RPO, list[KSPersistenceTrajectory]]]:
    """Build per-RPO trajectories and their per-order persistence diagrams.

    Derives each RPO's downsample stride from ``trajectory_dt`` before
    integrating it. Each pair carries ``max_derivative_order + 1`` diagram
    sequences, one per derivative order ``0..max_derivative_order``.
    Returned pairs are sorted by RPO ``time_steps`` descending so that the
    longest-running RPOs are dispatched first.
    """
    diagram_pairs: list[tuple[RPO, list[KSPersistenceTrajectory]]] = []
    for rpo in rpos:
        downsample = _resolve_rpo_downsample(trajectory_dt, rpo)
        rpo_trajectory = KSTrajectory.from_rpo(rpo, resolution, downsample, native)
        per_order = [
            KSPersistenceTrajectory.from_trajectory(rpo_trajectory, chunk_size, order=order)
            for order in range(max_derivative_order + 1)
        ]
        diagram_pairs.append((rpo, per_order))

    diagram_pairs.sort(key=lambda pair: pair[0].time_steps, reverse=True)
    return diagram_pairs


def _detect_from_diagrams(  # noqa: PLR0913, PLR0917
    trajectory_diagrams_per_order: list[KSPersistenceTrajectory],
    rpo_diagram_pairs: list[tuple[RPO, list[KSPersistenceTrajectory]]],
    delay: int,
    threshold: float,
    min_duration: int,
    show_progress: bool,
    n_workers: int,
    order_scales: NDArray[np.float64],
) -> list[ShadowingEvent]:
    """Run detection given trajectory and RPO diagrams in persistence form."""
    events: list[ShadowingEvent] = []
    for rpo_index, distance_matrix in _stream_distance_matrices(
        trajectory_diagrams_per_order,
        rpo_diagram_pairs,
        delay,
        show_progress,
        n_workers,
        order_scales,
    ):
        events.extend(
            _center_events(
                _extract_shadowing_events(distance_matrix, rpo_index, threshold, min_duration),
                delay,
                distance_matrix.shape[1],
            )
        )
    return events


def _min_distances_from_diagrams(  # noqa: PLR0913, PLR0917
    trajectory_diagrams_per_order: list[KSPersistenceTrajectory],
    rpo_diagram_pairs: list[tuple[RPO, list[KSPersistenceTrajectory]]],
    delay: int,
    show_progress: bool,
    n_workers: int,
    order_scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute per-timestep minimum distances given trajectory and RPO diagrams
    in persistence form.

    Timesteps for which no RPO supplies a finite distance (always the leading
    ``(delay - 1) // 2`` and trailing ``delay // 2`` timesteps due to the
    centered embedding) are returned as infinity.
    """
    num_timesteps = len(trajectory_diagrams_per_order[0])
    min_distances = np.full(num_timesteps, np.inf, dtype=np.float64)
    offset = (delay - 1) // 2

    for _, distance_matrix in _stream_distance_matrices(
        trajectory_diagrams_per_order,
        rpo_diagram_pairs,
        delay,
        show_progress,
        n_workers,
        order_scales,
    ):
        embedded_length = distance_matrix.shape[0]
        rpo_column_min = distance_matrix.min(axis=1)
        window = slice(offset, offset + embedded_length)
        min_distances[window] = np.minimum(min_distances[window], rpo_column_min)

    return min_distances


def _stream_distance_matrices(  # noqa: PLR0913, PLR0917
    trajectory_diagrams_per_order: list[KSPersistenceTrajectory],
    rpo_diagram_pairs: list[tuple[RPO, list[KSPersistenceTrajectory]]],
    delay: int,
    show_progress: bool,
    n_workers: int,
    order_scales: NDArray[np.float64],
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Yield ``(rpo_index, delay-embedded distance matrix)`` for each RPO.

    Computes one Wasserstein ``(T, P)`` matrix per derivative order, divides each
    by its per-order scale in ``order_scales``, averages them across orders, then
    applies the time-delay embedding (also a mean). The number of derivative
    orders is ``len(trajectory_diagrams_per_order)`` and must match each RPO's
    per-order phase-diagram list length and ``order_scales``.

    Parameters
    ----------
    trajectory_diagrams_per_order : list[KSPersistenceTrajectory]
        Trajectory persistence diagrams, one sequence per derivative order.
    rpo_diagram_pairs : list[tuple[RPO, list[KSPersistenceTrajectory]]]
        ``(rpo, per_order_phase_diagrams)`` pairs from
        ``_compute_rpo_diagram_pairs``.
    delay : int
        Time-delay embedding window size.
    show_progress : bool
        Whether to display ``tqdm`` progress bars.
    n_workers : int
        Number of worker processes.
    order_scales : NDArray[np.float64], shape (num_orders,)
        Per-order scale each order's Wasserstein column is divided by before the
        mean across orders. Pass ``np.ones(num_orders)`` for no rescaling.

    Yields
    ------
    rpo_index : int
        Index of the current RPO.
    distance_matrix : NDArray[np.float64], shape (num_timesteps - delay + 1, period)
        Embedded distance matrix for the current RPO.
    """
    num_orders = len(trajectory_diagrams_per_order)
    flat_offsets_per_order = [diagrams._flatten() for diagrams in trajectory_diagrams_per_order]
    num_timesteps = len(trajectory_diagrams_per_order[0])

    # Per-RPO cost scales as rpo.time_steps * num_orders (one Wasserstein
    # column per phase per order).
    total_cost = sum(rpo.time_steps for rpo, _ in rpo_diagram_pairs) * num_orders

    if n_workers == 1:
        with tqdm(
            total=total_cost,
            desc="Detecting",
            unit="step",
            disable=not show_progress,
        ) as outer_bar:
            for rpo, phase_diagrams_per_order in rpo_diagram_pairs:
                num_phases = len(phase_diagrams_per_order[0])
                wasserstein_matrix = np.zeros((num_timesteps, num_phases), dtype=np.float64)

                for order_index in range(num_orders):
                    flat_diagrams, offsets = flat_offsets_per_order[order_index]
                    phase_diagrams = phase_diagrams_per_order[order_index]

                    column_inputs: Iterable[tuple[int, NDArray[np.float64]]] = enumerate(
                        phase_diagrams
                    )
                    if show_progress:
                        column_inputs = tqdm(
                            column_inputs,
                            total=num_phases,
                            desc=f"  Order {order_index} phases",
                            leave=False,
                        )

                    for phase_index, diagram in column_inputs:
                        wasserstein_matrix[:, phase_index] += (
                            _wasserstein_column(flat_diagrams, offsets, diagram)
                            / order_scales[order_index]
                        )

                    outer_bar.update(rpo.time_steps)

                wasserstein_matrix /= num_orders
                yield rpo.index, _apply_delay_embedding(wasserstein_matrix, delay)
        return

    # Parallel branch: open one shared-memory pair per order; workers attach to
    # all of them and return the per-phase mean across orders as a single (T,)
    # column. Cross-process bandwidth stays linear in T.
    with (
        ExitStack() as stack,
        _forkserver_pool(n_workers) as pool,
        tqdm(
            total=total_cost,
            desc="Detecting",
            unit="step",
            disable=not show_progress,
        ) as outer_bar,
    ):
        shm_blocks = [
            (
                stack.enter_context(_shared_memory_view(flat)),
                stack.enter_context(_shared_memory_view(offsets)),
            )
            for flat, offsets in flat_offsets_per_order
        ]
        diagrams_shm_names = tuple(diagrams_shm.name for diagrams_shm, _ in shm_blocks)
        offsets_shm_names = tuple(offsets_shm.name for _, offsets_shm in shm_blocks)

        for rpo, phase_diagrams_per_order in rpo_diagram_pairs:
            num_phases = len(phase_diagrams_per_order[0])
            wasserstein_matrix = np.empty((num_timesteps, num_phases), dtype=np.float64)

            parallel_column_inputs = [
                _WassersteinColumnInputs(
                    phase_index=phase_index,
                    diagrams_shm_names=diagrams_shm_names,
                    offsets_shm_names=offsets_shm_names,
                    num_timesteps=num_timesteps,
                    rpo_diagrams=tuple(
                        phase_diagrams_per_order[order_index].diagrams[phase_index]
                        for order_index in range(num_orders)
                    ),
                    order_scales=tuple(float(scale) for scale in order_scales),
                )
                for phase_index in range(num_phases)
            ]
            column_results: Iterable[tuple[int, NDArray[np.float64]]] = pool.imap_unordered(
                _compute_wasserstein_column, parallel_column_inputs
            )
            if show_progress:
                column_results = tqdm(
                    column_results, total=num_phases, desc="  Phases", leave=False
                )

            for phase_index, column in column_results:
                wasserstein_matrix[:, phase_index] = column

            yield rpo.index, _apply_delay_embedding(wasserstein_matrix, delay)
            outer_bar.update(rpo.time_steps * num_orders)


@dataclass(frozen=True, slots=True)
class _WassersteinColumnInputs:
    """Inputs to ``_compute_wasserstein_column`` for one RPO phase.

    Attributes
    ----------
    phase_index : int
        RPO phase index; echoed back in the worker's return value so the
        dispatcher can reassemble the distance matrix from unordered results.
    diagrams_shm_names : tuple[str, ...]
        One shared-memory block name per derivative order, holding flattened
        trajectory persistence diagrams.
    offsets_shm_names : tuple[str, ...]
        One shared-memory block name per derivative order, holding trajectory
        diagram offsets. Same length as ``diagrams_shm_names``.
    num_timesteps : int
        Number of timesteps in the trajectory in shared memory.
    rpo_diagrams : tuple[NDArray[np.float64], ...]
        Persistence diagrams of this RPO phase, one per derivative order.
        Same length as ``diagrams_shm_names``.
    order_scales : tuple[float, ...]
        Per-order scale each order's Wasserstein column is divided by before the
        mean across orders. Same length as ``diagrams_shm_names``.
    """

    phase_index: int
    diagrams_shm_names: tuple[str, ...]
    offsets_shm_names: tuple[str, ...]
    num_timesteps: int
    rpo_diagrams: tuple[NDArray[np.float64], ...]
    order_scales: tuple[float, ...]


def _compute_wasserstein_column(
    inputs: _WassersteinColumnInputs,
) -> tuple[int, NDArray[np.float64]]:
    """Compute the per-phase mean Wasserstein column across derivative orders.

    Opens the per-order :class:`~multiprocessing.shared_memory.SharedMemory`
    blocks holding flattened trajectory persistence diagrams, computes one
    Wasserstein column per order via ``_wasserstein_column``, divides each by its
    per-order scale in ``inputs.order_scales``, and returns the mean across
    orders. Performing the reduction on the worker keeps cross-process bandwidth
    linear in ``num_timesteps``, not ``num_timesteps * len(diagrams_shm_names)``.
    """
    num_orders = len(inputs.diagrams_shm_names)
    column = np.zeros(inputs.num_timesteps, dtype=np.float64)

    diagrams_shms = [SharedMemory(name=name) for name in inputs.diagrams_shm_names]
    offsets_shms = [SharedMemory(name=name) for name in inputs.offsets_shm_names]
    try:
        for order_index in range(num_orders):
            offsets = np.ndarray(
                (inputs.num_timesteps + 1,),
                dtype=np.int64,
                buffer=offsets_shms[order_index].buf,
            )
            trajectory_diagrams = np.ndarray(
                (int(offsets[-1]), 2),
                dtype=np.float64,
                buffer=diagrams_shms[order_index].buf,
            )
            column += (
                _wasserstein_column(trajectory_diagrams, offsets, inputs.rpo_diagrams[order_index])
                / inputs.order_scales[order_index]
            )
    finally:
        for shm in diagrams_shms:
            shm.close()
        for shm in offsets_shms:
            shm.close()

    column /= num_orders
    return inputs.phase_index, column


def _apply_delay_embedding(
    wasserstein_matrix: NDArray[np.float64],
    delay: int,
) -> NDArray[np.float64]:
    r"""Apply time-delay embedding to a Wasserstein distance matrix.

    Computes :math:`W^w(i, j) = \frac{1}{w} \sum_{l=0}^{w-1}
    W(i+l, (j+l) \bmod J)` where :math:`w` is the delay window. This
    increases the effective dimensionality of the comparison by considering
    consecutive timesteps rather than single snapshots.

    Row ``i`` of the returned matrix aggregates input rows ``i`` through
    ``i + delay - 1``; callers attribute it to the window center
    ``i + (delay - 1) // 2``. For even ``delay`` the true center falls
    between samples and the floor convention leaves a residual attribution
    bias of at most half a timestep.

    Parameters
    ----------
    wasserstein_matrix : NDArray[np.float64], shape (I, J)
        Original Wasserstein distance matrix.
    delay : int
        Time-delay embedding window size (:math:`w`).

    Returns
    -------
    NDArray[np.float64], shape (I - delay + 1, J)
        Embedded distance matrix (mean over the window).

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
        column_indices = (np.arange(rpo_timesteps) + offset) % rpo_timesteps
        delayed += wasserstein_matrix[offset : offset + delayed_timesteps][:, column_indices]

    return delayed / delay


def _center_events(
    events: list[ShadowingEvent],
    delay: int,
    period: int,
) -> list[ShadowingEvent]:
    """Relabel events from embedded-matrix indices to window-center indices.

    Embedded cell ``(i, j)`` averages the raw pairs ``(i + l, (j + l) %
    period)`` for ``l`` in ``0..delay - 1``; the window's mean is attributed
    to its center pair, so ``start_timestep``/``end_timestep`` shift by
    ``(delay - 1) // 2`` and ``start_phase`` by the same offset modulo
    ``period``. Identity at ``delay = 1``.
    """
    offset = (delay - 1) // 2
    if offset == 0:
        return events
    return [
        replace(
            event,
            start_timestep=event.start_timestep + offset,
            end_timestep=event.end_timestep + offset,
            start_phase=(event.start_phase + offset) % period,
        )
        for event in events
    ]


def _attach_shifts(
    events: list[ShadowingEvent],
    trajectory: KSTrajectory,
    rpos: Sequence[RPO],
    native: bool,
) -> list[ShadowingEvent]:
    """Reconstruct spatial shifts for each event and sort the result.

    PHA quotients out spatial shifts during detection, leaving events with
    zero-filled ``shifts`` arrays. For each RPO shadowed by at least one event,
    builds an RPO trajectory at its inferred sampling and transforms it to
    its co-moving frame, then passes this co-moving trajectory to each event's
    shift reconstruction in ``_compute_event_shifts``. Sorts the result by
    ``(start_timestep, rpo_index)``.
    """
    rpo_by_index = {rpo.index: rpo for rpo in rpos}

    events_by_rpo: dict[int, list[ShadowingEvent]] = {}
    for event in events:
        events_by_rpo.setdefault(event.rpo_index, []).append(event)

    events_with_shifts: list[ShadowingEvent] = []
    for rpo_index, rpo_events in events_by_rpo.items():
        rpo = rpo_by_index[rpo_index]
        downsample = _resolve_rpo_downsample(trajectory.dt, rpo)
        rpo_trajectory = KSTrajectory.from_rpo(rpo, trajectory.resolution, downsample, native)
        rpo_comoving = rpo_trajectory.to_comoving(rpo.drift_rate)

        for event in rpo_events:
            events_with_shifts.append(
                _compute_event_shifts(event, trajectory, rpo.drift_rate, rpo_comoving)
            )

    events_with_shifts.sort(key=lambda event: (event.start_timestep, event.rpo_index))
    return events_with_shifts
