"""Shadowing event extraction via longest path computation in 2D distance
matrices.

Extracts shadowing events from Wasserstein distance matrices over the 2D
``(timestep, phase)`` grid with a three-stage pipeline:

1. Collect "close passes": grid entries whose distance falls below a threshold.
2. Group close passes into 8-connected components, with wraparound in the phase
   dimension.
3. Find the longest valid path through each component.

A valid path satisfies the co-evolution constraint: at each step the trajectory
timestep and RPO phase both advance by exactly 1 (phase modulo the RPO period).
No shift dimension is tracked; spatial symmetry is quotiented out upstream by
the persistence diagram representation and shifts are reconstructed post-hoc in
:mod:`~ks_shadowing.pha.shifts`.
"""

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.unionfind import _find_components

# Structured dtype for close passes over the ``(timestep, phase)`` grid.
_CLOSE_PASS_DTYPE = np.dtype(
    [
        ("timestep", np.int32),
        ("phase", np.int32),
        ("distance", np.float64),
    ]
)


def _extract_shadowing_events(
    distance_matrix: NDArray[np.float64],
    rpo_index: int,
    threshold: float,
    min_duration: int,
) -> list[ShadowingEvent]:
    """Extract shadowing events from a Wasserstein distance matrix.

    Entry point for the 2D pathfinding pipeline. Collects close passes below
    ``threshold``, groups them into 8-connected components, and returns the
    longest valid path through each component as a
    :class:`~ks_shadowing.core.event.ShadowingEvent`, skipping components
    whose longest path is shorter than ``min_duration``. Returned events
    carry zero-filled ``shifts``; PHA reconstructs spatial shifts post-hoc via
    ``_compute_event_shifts`` in ``pha/shifts.py``.

    Event indices are in embedded-matrix coordinates; the detection caller
    relabels them to window-center trajectory timesteps and phases via
    ``_center_events``.

    Parameters
    ----------
    distance_matrix : NDArray[np.float64], shape (num_timesteps, period)
        Wasserstein distance matrix following time-delay embedding.
    rpo_index : int
        Index of the RPO whose phases label the columns of ``distance_matrix``;
        stored on each returned event.
    threshold : float
        Grid entries with Wasserstein distance strictly below ``threshold``
        count as close passes.
    min_duration : int
        Minimum event duration in timesteps.

    Returns
    -------
    list[ShadowingEvent]
    """
    close_passes = _collect_close_passes(distance_matrix, threshold)
    if len(close_passes) == 0:
        return []

    _, period = distance_matrix.shape
    components = _find_connected_components(close_passes, period)

    events: list[ShadowingEvent] = []
    for component in components:
        path, mean_distance, min_distance = _find_longest_path(component, period)
        if len(path) < min_duration:
            continue

        start_timestep = int(path["timestep"][0])
        end_timestep = int(path["timestep"][-1]) + 1
        events.append(
            ShadowingEvent(
                rpo_index=rpo_index,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                mean_distance=mean_distance,
                min_distance=min_distance,
                start_phase=int(path["phase"][0]),
                shifts=np.zeros(end_timestep - start_timestep, dtype=np.int32),
            )
        )

    return events


def _collect_close_passes(
    distance_matrix: NDArray[np.float64],
    threshold: float,
) -> NDArray:
    """Collect all entries of ``distance_matrix`` below ``threshold``.

    Parameters
    ----------
    distance_matrix : NDArray[np.float64]
        Distance matrix to threshold.
    threshold : float
        Distances strictly below ``threshold`` are collected as close passes.

    Returns
    -------
    NDArray
        Structured array with dtype ``_CLOSE_PASS_DTYPE``.
    """
    timesteps, phases = np.asarray(distance_matrix < threshold).nonzero()

    passes = np.empty(len(timesteps), dtype=_CLOSE_PASS_DTYPE)
    passes["timestep"] = timesteps
    passes["phase"] = phases
    passes["distance"] = distance_matrix[timesteps, phases]

    return passes


def connected_components(
    timesteps: NDArray[np.int64],
    phases: NDArray[np.int64],
    period: int,
) -> NDArray[np.int32]:
    """Group close passes into 8-connected components over the phase cylinder.

    Two close passes are adjacent if their ``(timestep, phase)`` coordinates
    differ by at most 1 in each dimension, with wraparound in the phase
    dimension. This is the grouping stage of PHA detection, exposed so that
    callers can inspect it directly: a
    :class:`~ks_shadowing.core.event.ShadowingEvent` records only the longest
    valid path through its component, so the component it came from cannot be
    recovered from a detection result.

    The inputs are the coordinates of the entries of a distance matrix that
    fall below a detection threshold, in the form
    ``np.nonzero(distances < threshold)`` returns them.

    Parameters
    ----------
    timesteps : NDArray[np.int64], shape (num_passes,)
        Trajectory timestep index of each close pass.
    phases : NDArray[np.int64], shape (num_passes,)
        RPO phase index of each close pass, in ``[0, period)``.
    period : int
        RPO period; phase wraps modulo ``period``.

    Returns
    -------
    NDArray[np.int32], shape (num_passes,)
        Component label of each close pass, in the order the passes were
        given. Labels are contiguous in ``[0, num_components)`` but otherwise
        carry no meaning.

    Raises
    ------
    ValueError
        If ``timesteps`` and ``phases`` have different lengths, or any phase
        falls outside ``[0, period)``.
    """
    timesteps = np.asarray(timesteps)
    phases = np.asarray(phases)
    if len(timesteps) != len(phases):
        raise ValueError(
            f"timesteps and phases must have the same length, "
            f"got {len(timesteps)} and {len(phases)}"
        )
    if len(timesteps) == 0:
        return np.empty(0, dtype=np.int32)
    if phases.min() < 0 or phases.max() >= period:
        raise ValueError(
            f"phases must lie in [0, {period}), got range [{phases.min()}, {phases.max()}]"
        )

    # Union-find roots are pass indices, so they are neither contiguous nor
    # meaningful; densify them for callers who want to index by label.
    _, dense_labels = np.unique(_label_close_passes(timesteps, phases, period), return_inverse=True)
    return dense_labels.astype(np.int32)


def _find_connected_components(
    close_passes: NDArray,
    period: int,
) -> list[NDArray]:
    """Group close passes into 8-connected components.

    Two close passes are adjacent if they differ by at most 1 in each dimension
    with wraparound in the phase dimension.

    Parameters
    ----------
    close_passes : NDArray
        Structured array with dtype ``_CLOSE_PASS_DTYPE``.
    period : int
        RPO period; phase wraps modulo ``period``.

    Returns
    -------
    list[NDArray]
        One ``_CLOSE_PASS_DTYPE`` structured array per connected component.
    """
    if len(close_passes) == 0:
        return []

    # The labeller sorts internally and undoes it; this sort canonicalizes the
    # order the partition below sees, so tie-breaking between candidate paths
    # of equal length and equal mean distance in ``_find_longest_path`` depends
    # only on the cell set, never on caller order.
    sort_order = np.lexsort((close_passes["phase"], close_passes["timestep"]))
    close_passes = close_passes[sort_order]

    component_labels = _label_close_passes(close_passes["timestep"], close_passes["phase"], period)

    # Partition passes by component root.
    group_order = np.argsort(component_labels)
    sorted_labels = component_labels[group_order]
    splits = np.where(np.diff(sorted_labels) != 0)[0] + 1
    return [close_passes[group] for group in np.split(group_order, splits)]


def _label_close_passes(
    timesteps: NDArray,
    phases: NDArray,
    period: int,
) -> NDArray[np.int32]:
    """Assign each close pass the union-find root of its 8-connected component."""
    pass_count = len(timesteps)

    # Sort by ``(timestep, phase)`` so that every backward neighbor has been
    # processed when we visit a cell. ``np.lexsort`` uses the last key as
    # primary, and is the identity on input that is already in this order.
    sort_order = np.lexsort((phases, timesteps))
    timesteps = timesteps[sort_order]
    phases = phases[sort_order]

    # Two rolling rows suffice since passes are processed in timestep order and
    # only the previous row can contain backward neighbors. ``-1`` is an empty
    # cell.
    prev_row = np.full(period, -1, dtype=np.int32)
    curr_row = np.full(period, -1, dtype=np.int32)
    current_timestep = -1

    # Previously-processed neighbors given ``(timestep, phase)`` ordering:
    #   left, upper-left, up, upper-right.
    backward_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    edges_a: list[int] = []
    edges_b: list[int] = []

    # Each pass's edges depend on the row buffers left behind by earlier
    # passes in sorted (timestep, phase) order.
    for pass_index in range(pass_count):
        timestep = int(timesteps[pass_index])
        phase = int(phases[pass_index])

        if timestep != current_timestep:
            if timestep == current_timestep + 1:
                prev_row, curr_row = curr_row, prev_row
                curr_row.fill(-1)
            else:
                # First row, or a gap of >= 2 timesteps: no row t-1 to inherit.
                prev_row.fill(-1)
                curr_row.fill(-1)
            current_timestep = timestep

        curr_row[phase] = pass_index

        for dtimestep, dphase in backward_neighbors:
            row = prev_row if dtimestep == -1 else curr_row
            neighbor = row[(phase + dphase) % period]
            if neighbor >= 0:
                edges_a.append(pass_index)
                edges_b.append(neighbor)

        # Phase wraparound: when at the last phase column, phase 0 in the
        # same row was previously processed.
        if phase == period - 1:
            neighbor = curr_row[0]
            if neighbor >= 0:
                edges_a.append(pass_index)
                edges_b.append(neighbor)

    component_labels = _find_components(
        pass_count,
        np.array(edges_a, dtype=np.int32),
        np.array(edges_b, dtype=np.int32),
    )

    # Labels are indexed by sorted position; restore the caller's order.
    labels = np.empty_like(component_labels)
    labels[sort_order] = component_labels
    return labels


def _find_longest_path(
    passes: NDArray,
    period: int,
) -> tuple[NDArray, float, float]:
    """Find the longest valid path through a 2D connected component.

    A valid path satisfies the co-evolution constraint: from each step to the
    next, ``timestep`` advances by exactly 1 and ``phase`` advances by exactly 1
    modulo ``period``. Ties in path length are broken by lowest mean
    distance.

    Parameters
    ----------
    passes : NDArray
        Non-empty structured array with dtype ``_CLOSE_PASS_DTYPE``.
    period : int
        RPO period; phase transitions wrap modulo ``period``.

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
    # Sort so each successor is processed after its predecessor.
    passes = passes[np.argsort(passes["timestep"])]
    pass_count = len(passes)

    # Lookup from ``(timestep, phase)`` to index within the sorted array.
    cell_to_index: dict[tuple[int, int], int] = {
        (int(passes["timestep"][pass_index]), int(passes["phase"][pass_index])): pass_index
        for pass_index in range(pass_count)
    }

    path_length = np.ones(pass_count, dtype=np.int32)
    distance_sum = passes["distance"].astype(np.float64)
    min_distance = distance_sum.copy()
    predecessor = np.full(pass_count, -1, dtype=np.int32)

    # Each entry's predecessor lookup needs path_length/distance_sum already
    # finalized for earlier timesteps.
    for pass_index in range(pass_count):
        prev_key = (
            int(passes["timestep"][pass_index]) - 1,
            (int(passes["phase"][pass_index]) - 1) % period,
        )
        previous = cell_to_index.get(prev_key)
        if previous is None:
            continue

        path_length[pass_index] = path_length[previous] + 1
        distance_sum[pass_index] = distance_sum[previous] + passes["distance"][pass_index]
        min_distance[pass_index] = min(min_distance[previous], passes["distance"][pass_index])
        predecessor[pass_index] = previous

    # Pick the endpoint maximizing length, breaking ties by mean distance.
    # ``np.lexsort`` treats its last key as primary; negating ``path_length``
    # turns maximization into ascending sort.
    mean_distances = distance_sum / path_length
    best_end = int(np.lexsort((mean_distances, -path_length))[0])

    # Walk predecessor links back to the start of the path.
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
