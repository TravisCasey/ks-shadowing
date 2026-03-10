"""Shadowing event characterization via longest pathfinding algorithm (3D version).

This module implements the graph-based algorithm for extracting shadowing events
from distance data in 3D space (timestep x phase x shift):

1. Collect "close passes" - points where trajectory is within threshold of an RPO
2. Group close passes into connected components (26-connectivity in 3D grid)
3. Find the longest valid path through each component

A valid path must satisfy temporal co-evolution (trajectory and RPO timesteps
advance together) and spatial continuity (shift changes by at most 1, with
wraparound).
"""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.util import _UnionFind
from ks_shadowing.ssa.rpo import _RPOStateSpace

# Structured dtype for close passes in 3D (timestep, phase, shift).
_CLOSE_PASS_DTYPE_3D = np.dtype(
    [
        ("timestep", np.int32),
        ("phase", np.int32),
        ("shift", np.int32),
        ("distance", np.float64),
    ]
)


class _ComponentPathFinder3D:
    """Finds the longest valid shadowing path through a connected component.

    A valid path satisfies these constraints at each step:
    - Timestep advances by exactly 1
    - Phase offset remains constant (trajectory and RPO advance together)
    - Spatial shift changes by at most 1 (with wraparound)
    """

    def __init__(self, passes: NDArray, period: int, resolution: int):
        """Initialize with structured array of close passes."""
        self.period = period
        self.resolution = resolution

        # Sort by timestep
        sort_indices = np.argsort(passes["timestep"])
        self.passes = passes[sort_indices]
        self.pass_count = len(self.passes)

        # Build lookup: (timestep, phase) -> [(shift, pass_index), ...],
        # where pass_index indexes into self.passes
        self.lookup: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for pass_index in range(self.pass_count):
            key = (int(self.passes["timestep"][pass_index]), int(self.passes["phase"][pass_index]))
            if key not in self.lookup:
                self.lookup[key] = []
            self.lookup[key].append((int(self.passes["shift"][pass_index]), pass_index))

    def find_longest_path(self) -> tuple[NDArray, float, float] | None:
        """Find the longest valid path, breaking ties by lowest mean distance.

        Returns `(path, mean_distance, min_distance)` or None if the component
        is empty.
        """
        if self.pass_count == 0:
            return None

        path_length = np.ones(self.pass_count, dtype=np.int32)
        dist_sum = self.passes["distance"].astype(np.float64)
        min_dist = dist_sum.copy()
        predecessor = np.full(self.pass_count, -1, dtype=np.int32)

        # Process in timestep order, extending paths from valid predecessors
        for pass_index in range(self.pass_count):
            pass_timestep = int(self.passes["timestep"][pass_index])
            pass_phase = int(self.passes["phase"][pass_index])
            pass_shift = int(self.passes["shift"][pass_index])

            best = self._find_best_predecessor(
                pass_timestep, pass_phase, pass_shift, path_length, dist_sum
            )
            if best is None:
                continue

            path_length[pass_index] = path_length[best] + 1
            dist_sum[pass_index] = dist_sum[best] + self.passes["distance"][pass_index]
            min_dist[pass_index] = min(min_dist[best], self.passes["distance"][pass_index])
            predecessor[pass_index] = best

        # Find best endpoint (longest path, then lowest mean distance)
        best_endpoint = self._find_best_endpoint(path_length, dist_sum)
        path = self._reconstruct_path(predecessor, best_endpoint)
        mean_distance = float(dist_sum[best_endpoint] / len(path))
        return path, mean_distance, float(min_dist[best_endpoint])

    def _find_best_predecessor(
        self,
        pass_timestep: int,
        pass_phase: int,
        pass_shift: int,
        path_length: NDArray[np.int32],
        dist_sum: NDArray[np.float64],
    ) -> int | None:
        """Find the best predecessor for a close pass, or ``None`` if none valid.

        Phase must remain constant along a path (trajectory and RPO advance
        together), while shift can vary by at most 1 with wraparound.
        """
        # Phase must be constant along path - only look at same phase
        prev_key = (pass_timestep - 1, pass_phase)
        if prev_key not in self.lookup:
            return None

        best_index: int | None = None
        best_length = 0
        best_mean = float("inf")

        for prev_shift, prev_index in self.lookup[prev_key]:
            if not self._is_valid_shift_transition(prev_shift, pass_shift):
                continue

            prev_len = path_length[prev_index]
            prev_mean = dist_sum[prev_index] / prev_len

            if prev_len > best_length or (prev_len == best_length and prev_mean < best_mean):
                best_index = prev_index
                best_length = prev_len
                best_mean = prev_mean

        return best_index

    def _find_best_endpoint(
        self,
        path_length: NDArray[np.int32],
        dist_sum: NDArray[np.float64],
    ) -> int:
        """Find the index with longest path, breaking ties by lowest mean distance."""
        assert self.pass_count > 0
        best_index = 0
        best_length = 0
        best_mean = float("inf")

        for pass_index in range(self.pass_count):
            pass_length = path_length[pass_index]
            pass_mean = dist_sum[pass_index] / pass_length

            if pass_length > best_length or (pass_length == best_length and pass_mean < best_mean):
                best_index = pass_index
                best_length = pass_length
                best_mean = pass_mean

        return best_index

    def _reconstruct_path(self, predecessor: NDArray[np.int32], end: int) -> NDArray:
        """Reconstruct path by following predecessor links."""
        path_indices = []
        current_index = end

        while current_index >= 0:
            path_indices.append(current_index)
            current_index = predecessor[current_index]
        path_indices.reverse()

        return self.passes[path_indices]

    def _is_valid_shift_transition(self, shift_from: int, shift_to: int) -> bool:
        """Check if shift transition is valid (differs by at most 1, with wraparound)."""
        diff = (shift_to - shift_from) % self.resolution
        return diff <= 1 or diff == self.resolution - 1


def _collect_close_passes_3d(
    dist_sq_generator: Iterator[tuple[int, NDArray[np.float64]]],
    threshold: float,
) -> NDArray:
    """Collect all entries below threshold from a squared distance generator.

    The generator yields `(phase, dist_sq)` tuples where `dist_sq` has shape
    `(num_timesteps, resolution)` containing squared distances.

    Returns a structured array with dtype `_CLOSE_PASS_DTYPE_3D`.
    """
    threshold_sq = threshold * threshold
    chunks: list[NDArray] = []

    for phase, dist_sq in dist_sq_generator:
        # Find entries below threshold (compare squared values)
        step_index, shift_index = np.asarray(dist_sq < threshold_sq).nonzero()
        step_count = len(step_index)
        if step_count == 0:
            continue

        chunk: NDArray = np.empty(step_count, dtype=_CLOSE_PASS_DTYPE_3D)
        chunk["timestep"] = step_index
        chunk["phase"] = phase
        chunk["shift"] = shift_index
        chunk["distance"] = np.sqrt(dist_sq[step_index, shift_index])
        chunks.append(chunk)

    if not chunks:
        return np.array([], dtype=_CLOSE_PASS_DTYPE_3D)
    return np.concatenate(chunks)


def _find_connected_components_3d(  # noqa: PLR0912
    close_passes: NDArray,
    period: int,
    resolution: int,
) -> list[NDArray]:
    """Group close passes into connected components using 26-connectivity.

    Two points are adjacent if they differ by at most 1 in each dimension,
    with wraparound in phase and shift dimensions.

    Takes and returns structured arrays with dtype `_CLOSE_PASS_DTYPE_3D`.
    """
    if len(close_passes) == 0:
        return []

    # Sort by (timestep, phase, shift) for sweep-line processing
    # np.lexsort sorts by the last key first
    sort_order = np.lexsort(
        (
            close_passes["shift"],
            close_passes["phase"],
            close_passes["timestep"],
        )
    )
    close_passes = close_passes[sort_order]

    pass_count = len(close_passes)
    uf = _UnionFind(pass_count)

    # Two-slice dense label arrays: -1 = no pass, >=0 = pass index
    labels_prev = np.full((period, resolution), -1, dtype=np.int32)
    labels_curr = np.full((period, resolution), -1, dtype=np.int32)

    timesteps = close_passes["timestep"]
    phases = close_passes["phase"]
    shifts = close_passes["shift"]

    # Backward neighbors in previous slice (dt=-1): all 9 neighbors
    prev_slice_offsets = [(dp, ds) for dp in (-1, 0, 1) for ds in (-1, 0, 1)]

    # Backward neighbors in current slice (dt=0): 4 neighbors in row-major order
    # left, upper-left, up, upper-right
    curr_slice_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    current_timestep = -1

    for pass_index in range(pass_count):
        t = int(timesteps[pass_index])
        p = int(phases[pass_index])
        s = int(shifts[pass_index])

        # Handle timestep transitions
        if t != current_timestep:
            if t > current_timestep + 1:
                # Skipped timesteps - clear both slices
                labels_prev.fill(-1)
                labels_curr.fill(-1)
            else:
                # Normal advance - swap slices
                labels_prev, labels_curr = labels_curr, labels_prev
                labels_curr.fill(-1)
            current_timestep = t

        labels_curr[p, s] = pass_index

        # Check previous slice (dt=-1)
        if t > 0:
            for dp, ds in prev_slice_offsets:
                neighbor_label = labels_prev[(p + dp) % period, (s + ds) % resolution]
                if neighbor_label >= 0:
                    uf.union(pass_index, neighbor_label)

        # Check current slice backward neighbors (dt=0)
        for dp, ds in curr_slice_offsets:
            neighbor_label = labels_curr[(p + dp) % period, (s + ds) % resolution]
            if neighbor_label >= 0:
                uf.union(pass_index, neighbor_label)

        # Handle phase wraparound in current slice: when at last phase row,
        # check phase 0 which has already been processed
        if p == period - 1:
            for ds in (-1, 0, 1):
                neighbor_label = labels_curr[0, (s + ds) % resolution]
                if neighbor_label >= 0:
                    uf.union(pass_index, neighbor_label)

        # Handle shift wraparound in current slice: when at last shift column,
        # check shift 0 which has already been processed
        if s == resolution - 1:
            neighbor_label = labels_curr[p, 0]
            if neighbor_label >= 0:
                uf.union(pass_index, neighbor_label)

    # Group by component root
    component_indices: dict[int, list[int]] = {}
    for pass_index in range(pass_count):
        root = uf.find(pass_index)
        if root not in component_indices:
            component_indices[root] = []
        component_indices[root].append(pass_index)

    return [close_passes[indices] for indices in component_indices.values()]


def _extract_shadowing_events_3d(
    dist_sq_generator: Iterator[tuple[int, NDArray[np.float64]]],
    rpo_data: _RPOStateSpace,
    threshold: float,
    min_duration: int = 1,
) -> list[ShadowingEvent]:
    r"""Extract shadowing events from squared distances using connected components.

    Main entry point for the 3D pathfinding algorithm: collects close passes
    below ``threshold``, groups them into 26-connected components, and finds
    the longest valid path through each. Returns one
    :class:`~ks_shadowing.core.event.ShadowingEvent` per component (if longer
    than ``min_duration``).

    Parameters
    ----------
    dist_sq_generator : Iterator[tuple[int, NDArray[np.float64]]]
        Yields ``(phase, dist_sq)`` tuples where ``dist_sq`` has shape
        ``(num_timesteps, resolution)`` containing squared :math:`L_2`
        distances.
    rpo_data : _RPOStateSpace
        Precomputed RPO trajectory in state space.
    threshold : float
        Maximum :math:`L_2` distance for close passes.
    min_duration : int, optional
        Minimum event duration in timesteps. Default is 1.

    Returns
    -------
    list[ShadowingEvent]
        Events sorted by ``start_timestep``.
    """
    close_passes = _collect_close_passes_3d(dist_sq_generator, threshold)
    if len(close_passes) == 0:
        return []

    period = rpo_data.time_steps
    resolution = rpo_data.resolution
    components = _find_connected_components_3d(close_passes, period, resolution)
    events: list[ShadowingEvent] = []

    for component in components:
        finder = _ComponentPathFinder3D(component, period, resolution)
        result = finder.find_longest_path()

        if result is None or len(result[0]) < min_duration:
            continue
        path, mean_distance, min_distance = result

        events.append(
            ShadowingEvent(
                rpo_index=rpo_data.index,
                start_timestep=int(path["timestep"][0]),
                end_timestep=int(path["timestep"][-1]) + 1,
                mean_distance=mean_distance,
                min_distance=min_distance,
                start_phase=int(path["phase"][0]),
                shifts=path["shift"],
            )
        )

    events.sort(key=lambda e: e.start_timestep)
    return events
