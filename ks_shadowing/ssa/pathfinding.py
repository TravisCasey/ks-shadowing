"""Shadowing event extraction via longest pathfinding in 3D.

Extracts shadowing events from streaming squared-distance data over the 3D
``(timestep, rpo_phase, shift)`` grid with a three-stage pipeline:

1. Collect "close passes": grid entries whose distance falls below a threshold.
2. Group close passes into 26-connected components, with wraparound in the
   ``rpo_phase`` and ``shift`` dimensions.
3. Find the longest valid path through each component.

A valid path satisfies temporal co-evolution (trajectory and RPO advance
together one row per timestep, so ``rpo_phase`` advances by exactly 1 modulo
the period at each step) and spatial continuity (``shift`` changes by at most
1 between steps, with wraparound modulo the spatial resolution).
"""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.unionfind import _find_components

# Structured dtype for close passes over the ``(timestep, rpo_phase, shift)`` grid.
_CLOSE_PASS_DTYPE = np.dtype(
    [
        ("timestep", np.int32),
        ("rpo_phase", np.int32),
        ("shift", np.int32),
        ("distance", np.float64),
    ]
)


def _extract_shadowing_events(  # noqa: PLR0913
    dist_sq_generator: Iterator[tuple[int, int, NDArray[np.float64]]],
    rpo_index: int,
    period: int,
    resolution: int,
    threshold: float,
    min_duration: int,
) -> list[ShadowingEvent]:
    """Extract shadowing events from streaming squared distances.

    Entry point for the 3D pathfinding pipeline. Collects close passes below
    ``threshold``, groups them into 26-connected components, and returns the
    longest valid path through each component as a
    :class:`~ks_shadowing.core.event.ShadowingEvent`, skipping components
    whose longest path is shorter than ``min_duration``.

    Parameters
    ----------
    dist_sq_generator : Iterator[tuple[int, int, NDArray[np.float64]]]
        Yields ``(phase, chunk_start, dist_sq)`` tuples where ``dist_sq`` has
        shape ``(chunk_len, resolution)`` containing squared :math:`L_2`
        distances.
    rpo_index : int
        Index of the RPO whose phases are being tested; stored on each
        returned event.
    period : int
        Number of rows in the RPO trajectory used during detection;
        ``rpo_phase`` wraps modulo ``period``.
    resolution : int
        Spatial resolution; shift dimension wraps modulo ``resolution``.
    threshold : float
        Grid entries with :math:`L_2` distance strictly below ``threshold``
        count as close passes.
    min_duration : int
        Minimum event duration in timesteps.

    Returns
    -------
    list[ShadowingEvent]
    """
    close_passes = _collect_close_passes(dist_sq_generator, period, threshold)
    if len(close_passes) == 0:
        return []

    components = _find_connected_components(close_passes, period, resolution)

    events: list[ShadowingEvent] = []
    for component in components:
        path, mean_distance, min_distance = _find_longest_path(component, period, resolution)
        if len(path) < min_duration:
            continue

        events.append(
            ShadowingEvent(
                rpo_index=rpo_index,
                start_timestep=int(path["timestep"][0]),
                end_timestep=int(path["timestep"][-1]) + 1,
                mean_distance=mean_distance,
                min_distance=min_distance,
                start_phase=int(path["rpo_phase"][0]),
                shifts=path["shift"],
            )
        )

    return events


def _collect_close_passes(
    dist_sq_generator: Iterator[tuple[int, int, NDArray[np.float64]]],
    period: int,
    threshold: float,
) -> NDArray:
    """Collect all entries below threshold from a squared-distance generator.

    Stores the absolute RPO phase ``rpo_phase = (chunk_start + phase +
    step_index) mod period`` in place of the raw phase offset emitted by the
    generator.

    Parameters
    ----------
    dist_sq_generator : Iterator[tuple[int, int, NDArray[np.float64]]]
        Yields ``(phase, chunk_start, dist_sq)`` tuples.
    period : int
        Number of rows in the RPO trajectory; ``rpo_phase`` wraps modulo
        ``period``.
    threshold : float
        :math:`L_2` distances strictly below ``threshold`` are collected as
        close passes.

    Returns
    -------
    NDArray
        Structured array with dtype ``_CLOSE_PASS_DTYPE``.
    """
    threshold_sq = threshold * threshold
    chunks: list[NDArray] = []

    for phase, chunk_start, dist_sq in dist_sq_generator:
        step_index, shift_index = (dist_sq < threshold_sq).nonzero()
        if len(step_index) == 0:
            continue

        chunk: NDArray = np.empty(len(step_index), dtype=_CLOSE_PASS_DTYPE)
        chunk["timestep"] = step_index + chunk_start
        chunk["rpo_phase"] = (chunk_start + phase + step_index) % period
        chunk["shift"] = shift_index
        chunk["distance"] = np.sqrt(dist_sq[step_index, shift_index])
        chunks.append(chunk)

    if not chunks:
        return np.array([], dtype=_CLOSE_PASS_DTYPE)
    return np.concatenate(chunks)


def _find_connected_components(  # noqa: PLR0912
    close_passes: NDArray,
    period: int,
    resolution: int,
) -> list[NDArray]:
    """Group close passes into 26-connected components.

    Two close passes are adjacent if they differ by at most 1 in each of
    ``timestep``, ``rpo_phase``, and ``shift``, with wraparound on
    ``rpo_phase`` and ``shift``. Sweep-line over timesteps maintains two
    ``(period, resolution)`` dense label slices; union-find is batched in C++.

    Parameters
    ----------
    close_passes : NDArray
        Structured array with dtype ``_CLOSE_PASS_DTYPE``.
    period : int
        Number of rows in the RPO trajectory; ``rpo_phase`` wraps modulo
        ``period``.
    resolution : int
        Spatial resolution; shift wraps modulo ``resolution``.

    Returns
    -------
    list[NDArray]
        One ``_CLOSE_PASS_DTYPE`` structured array per connected component.
    """
    if len(close_passes) == 0:
        return []

    sort_order = np.lexsort(
        (
            close_passes["shift"],
            close_passes["rpo_phase"],
            close_passes["timestep"],
        )
    )
    close_passes = close_passes[sort_order]
    pass_count = len(close_passes)

    timesteps = close_passes["timestep"]
    rpo_phases = close_passes["rpo_phase"]
    shifts = close_passes["shift"]

    labels_prev = np.full((period, resolution), -1, dtype=np.int32)
    labels_curr = np.full((period, resolution), -1, dtype=np.int32)

    # Previous-slice (dt=-1) neighbors: full 3x3 face -- all nine cells.
    prev_slice_offsets = [(dphase, ds) for dphase in (-1, 0, 1) for ds in (-1, 0, 1)]
    # Current-slice (dt=0) backward neighbors under (timestep, rpo_phase, shift)
    # ordering: cells with dphase=-1 (any ds) plus (dphase=0, ds=-1).
    curr_slice_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    current_timestep = -1
    edges_a: list[int] = []
    edges_b: list[int] = []

    # Each pass's edges depend on the label grids left behind by earlier
    # passes in sorted (timestep, rpo_phase, shift) order.
    for pass_index in range(pass_count):
        timestep = int(timesteps[pass_index])
        rpo_phase = int(rpo_phases[pass_index])
        shift = int(shifts[pass_index])

        if timestep != current_timestep:
            if timestep > current_timestep + 1:
                labels_prev.fill(-1)
                labels_curr.fill(-1)
            else:
                labels_prev, labels_curr = labels_curr, labels_prev
                labels_curr.fill(-1)
            current_timestep = timestep

        labels_curr[rpo_phase, shift] = pass_index

        if timestep > 0:
            for dphase, ds in prev_slice_offsets:
                neighbor = labels_prev[(rpo_phase + dphase) % period, (shift + ds) % resolution]
                if neighbor >= 0:
                    edges_a.append(pass_index)
                    edges_b.append(neighbor)

        for dphase, ds in curr_slice_offsets:
            neighbor = labels_curr[(rpo_phase + dphase) % period, (shift + ds) % resolution]
            if neighbor >= 0:
                edges_a.append(pass_index)
                edges_b.append(neighbor)

        # Wraparound in rpo_phase: when at the last phase row in the current
        # slice, phase 0 at the same timestep has already been processed.
        if rpo_phase == period - 1:
            for ds in (-1, 0, 1):
                neighbor = labels_curr[0, (shift + ds) % resolution]
                if neighbor >= 0:
                    edges_a.append(pass_index)
                    edges_b.append(neighbor)

        # Wraparound in shift: when at the last shift column for this
        # (timestep, rpo_phase), shift 0 has already been processed.
        if shift == resolution - 1:
            neighbor = labels_curr[rpo_phase, 0]
            if neighbor >= 0:
                edges_a.append(pass_index)
                edges_b.append(neighbor)

    component_labels = _find_components(
        pass_count,
        np.array(edges_a, dtype=np.int32),
        np.array(edges_b, dtype=np.int32),
    )

    group_order = np.argsort(component_labels)
    sorted_labels = component_labels[group_order]
    splits = np.where(np.diff(sorted_labels) != 0)[0] + 1
    return [close_passes[group] for group in np.split(group_order, splits)]


def _find_longest_path(
    passes: NDArray,
    period: int,
    resolution: int,
) -> tuple[NDArray, float, float]:
    """Find the longest valid path through a 3D connected component.

    A valid path satisfies temporal co-evolution: from each step to the next,
    ``timestep`` advances by exactly 1, ``rpo_phase`` advances by exactly 1
    modulo ``period``, and ``shift`` changes by at most 1 modulo
    ``resolution``. Ties in path length are broken by lowest mean distance.

    Parameters
    ----------
    passes : NDArray
        Non-empty structured array with dtype ``_CLOSE_PASS_DTYPE``.
    period : int
        Number of rows in the RPO trajectory; ``rpo_phase`` advances mod
        ``period``.
    resolution : int
        Spatial resolution; shift transitions wrap modulo ``resolution``.

    Returns
    -------
    path : NDArray
        Slice of ``passes`` with dtype ``_CLOSE_PASS_DTYPE`` forming the chosen
        path, ordered by timestep.
    mean_distance : float
        Mean ``distance`` along ``path``.
    min_distance : float
        Minimum ``distance`` along ``path``.
    """
    passes = passes[np.argsort(passes["timestep"])]
    pass_count = len(passes)

    cell_to_entries: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pass_index in range(pass_count):
        key = (int(passes["timestep"][pass_index]), int(passes["rpo_phase"][pass_index]))
        cell_to_entries.setdefault(key, []).append((int(passes["shift"][pass_index]), pass_index))

    path_length = np.ones(pass_count, dtype=np.int32)
    distance_sum = passes["distance"].astype(np.float64)
    min_distance = distance_sum.copy()
    predecessor = np.full(pass_count, -1, dtype=np.int32)

    # Each entry's best predecessor lookup needs path_length/distance_sum
    # already finalized for earlier timesteps.
    for pass_index in range(pass_count):
        pass_timestep = int(passes["timestep"][pass_index])
        pass_rpo_phase = int(passes["rpo_phase"][pass_index])
        pass_shift = int(passes["shift"][pass_index])

        prev_entries = cell_to_entries.get((pass_timestep - 1, (pass_rpo_phase - 1) % period))
        if prev_entries is None:
            continue

        best_predecessor: int | None = None
        best_length = 0
        best_mean = float("inf")

        for prev_shift, prev_index in prev_entries:
            shift_diff = (pass_shift - prev_shift) % resolution
            if shift_diff > 1 and shift_diff != resolution - 1:
                continue

            prev_length = int(path_length[prev_index])
            prev_mean = float(distance_sum[prev_index]) / prev_length
            if prev_length > best_length or (prev_length == best_length and prev_mean < best_mean):
                best_predecessor = prev_index
                best_length = prev_length
                best_mean = prev_mean

        if best_predecessor is None:
            continue

        path_length[pass_index] = path_length[best_predecessor] + 1
        distance_sum[pass_index] = distance_sum[best_predecessor] + passes["distance"][pass_index]
        min_distance[pass_index] = min(
            min_distance[best_predecessor], passes["distance"][pass_index]
        )
        predecessor[pass_index] = best_predecessor

    mean_distances = distance_sum / path_length
    best_end = int(np.lexsort((mean_distances, -path_length))[0])

    path_indices: list[int] = []
    cursor = best_end
    while cursor >= 0:
        path_indices.append(cursor)
        cursor = int(predecessor[cursor])
    path_indices.reverse()

    return (
        passes[path_indices],
        float(distance_sum[best_end]) / len(path_indices),
        float(min_distance[best_end]),
    )
