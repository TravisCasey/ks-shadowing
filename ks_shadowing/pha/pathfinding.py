"""Shadowing event characterization via longest pathfinding algorithm in 2D.

This module implements the graph-based algorithm for extracting shadowing events
from Wasserstein distance data in 2D space ``(timestep, phase)``:

1. Collect "close passes" - points where trajectory is within threshold of RPO
2. Group close passes into connected components (8-connectivity in 2D grid)
3. Find the longest valid path through each component

A valid path must satisfy temporal co-evolution: both trajectory timestep and
RPO phase advance by 1 at each step.
"""

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.util import _UnionFind
from ks_shadowing.pha.persistence import _RPOPersistence

# Structured dtype for close passes in 2D (timestep, phase).
# No shift dimension since PHA quotients out spatial symmetry.
_CLOSE_PASS_DTYPE_2D = np.dtype(
    [
        ("timestep", np.int32),
        ("phase", np.int32),
        ("distance", np.float64),
    ]
)


class _ComponentPathFinder2D:
    """Finds the longest valid shadowing path through a 2D connected component.

    A valid path satisfies these constraints at each step:
    - Timestep advances by exactly 1
    - Phase advances by exactly 1 (mod period)

    Unlike the 3D version, there is no shift dimension to track. The path
    represents temporal co-evolution of trajectory and RPO.
    """

    def __init__(self, passes: NDArray, period: int):
        """Initialize with structured array of close passes."""
        self.period = period

        # Sort by timestep
        sort_indices = np.argsort(passes["timestep"])
        self.passes = passes[sort_indices]
        self.pass_count = len(self.passes)

        # Build lookup: (timestep, phase) -> index
        # In 2D, each (timestep, phase) pair should be unique
        self.lookup: dict[tuple[int, int], int] = {}
        for pass_index in range(self.pass_count):
            key = (int(self.passes["timestep"][pass_index]), int(self.passes["phase"][pass_index]))
            self.lookup[key] = pass_index

    def find_longest_path(self) -> tuple[NDArray, float, float] | None:
        """Find the longest valid path, breaking ties by lowest mean distance.

        Returns ``(path, mean_distance, min_distance)`` or ``None`` if the
        component is empty.
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

            best = self._find_best_predecessor(pass_timestep, pass_phase, path_length, dist_sum)
            if best is None:
                continue

            path_length[pass_index] = path_length[best] + 1
            dist_sum[pass_index] = dist_sum[best] + self.passes["distance"][pass_index]
            min_dist[pass_index] = min(min_dist[best], self.passes["distance"][pass_index])
            predecessor[pass_index] = best

        # Find best endpoint (longest path, then lowest mean distance)
        best_end = self._find_best_endpoint(path_length, dist_sum)
        path = self._reconstruct_path(predecessor, best_end)
        mean_distance = float(dist_sum[best_end] / len(path))
        return path, mean_distance, float(min_dist[best_end])

    def _find_best_predecessor(
        self,
        pass_timestep: int,
        pass_phase: int,
        path_length: NDArray[np.int32],
        dist_sum: NDArray[np.float64],
    ) -> int | None:
        """Find the best predecessor for a close pass, or ``None`` if none valid.

        In 2D, the predecessor must have timestep-1 and phase-1 (mod period).
        Both trajectory and RPO advance together.
        """
        prev_timestep = pass_timestep - 1
        prev_phase = (pass_phase - 1) % self.period
        prev_key = (prev_timestep, prev_phase)

        if prev_key not in self.lookup:
            return None

        return self.lookup[prev_key]

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
            length = path_length[pass_index]
            mean = dist_sum[pass_index] / length

            if length > best_length or (length == best_length and mean < best_mean):
                best_index = pass_index
                best_length = length
                best_mean = mean

        return best_index

    def _reconstruct_path(self, predecessor: NDArray[np.int32], end: int) -> NDArray:
        """Reconstruct path by following predecessor links."""
        indices = []
        current_index = end
        while current_index >= 0:
            indices.append(current_index)
            current_index = predecessor[current_index]
        indices.reverse()
        return self.passes[indices]


def _collect_close_passes_2d(
    distance_matrix: NDArray[np.float64],
    threshold: float,
) -> NDArray:
    """Collect all entries below threshold from a 2D distance matrix.

    Parameters
    ----------
    distance_matrix : NDArray[np.float64], shape (num_timesteps, period)
        Distance matrix with entries to threshold.
    threshold : float
        Maximum distance for close passes.

    Returns
    -------
    NDArray
        Structured array with dtype ``_CLOSE_PASS_DTYPE_2D``.
    """
    timestep_indices, phase_indices = np.asarray(distance_matrix < threshold).nonzero()
    count = len(timestep_indices)

    if count == 0:
        return np.array([], dtype=_CLOSE_PASS_DTYPE_2D)

    passes = np.empty(count, dtype=_CLOSE_PASS_DTYPE_2D)
    passes["timestep"] = timestep_indices
    passes["phase"] = phase_indices
    passes["distance"] = distance_matrix[timestep_indices, phase_indices]

    return passes


def _find_connected_components_2d(
    close_passes: NDArray,
    period: int,
    num_timesteps: int,
) -> list[NDArray]:
    """Group close passes into connected components using 8-connectivity.

    Two points are adjacent if they differ by at most 1 in each dimension,
    with wraparound in the phase dimension. Uses a dense label array and
    single-pass sweep for efficiency.

    Parameters
    ----------
    close_passes : NDArray
        Structured array with dtype ``_CLOSE_PASS_DTYPE_2D``.
    period : int
        RPO period for phase wraparound.
    num_timesteps : int
        Number of timesteps in the distance matrix.

    Returns
    -------
    list[NDArray]
        One structured array per connected component.
    """
    if len(close_passes) == 0:
        return []

    # Sort by (timestep, phase) to enable single-pass sweep algorithm.
    # np.lexsort sorts by last key first.
    sort_order = np.lexsort((close_passes["phase"], close_passes["timestep"]))
    close_passes = close_passes[sort_order]

    pass_count = len(close_passes)
    uf = _UnionFind(pass_count)

    # Dense label array: -1 = not a close pass, >=0 = pass index
    labels = np.full((num_timesteps, period), -1, dtype=np.int32)

    timesteps = close_passes["timestep"]
    phases = close_passes["phase"]

    # Single-pass sweep: assign labels and check only backward neighbors.
    # Since close_passes are sorted by (timestep, phase), we only need to
    # check neighbors that have already been processed: left, upper-left,
    # up, and upper-right.
    backward_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for pass_index in range(pass_count):
        t, p = int(timesteps[pass_index]), int(phases[pass_index])
        labels[t, p] = pass_index

        for dt, dp in backward_neighbors:
            nt = t + dt
            if nt < 0:
                continue
            neighbor_phase = (p + dp) % period
            neighbor_label = labels[nt, neighbor_phase]
            if neighbor_label >= 0:
                uf.union(pass_index, neighbor_label)

        # Handle phase wraparound: when at the last phase column, check if
        # phase 0 in the same row was already processed (it was, since we
        # process in row-major order).
        if p == period - 1:
            neighbor_label = labels[t, 0]
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


def _extract_shadowing_events_2d(
    distance_matrix: NDArray[np.float64],
    rpo_data: _RPOPersistence,
    threshold: float,
    min_duration: int,
) -> list[ShadowingEvent]:
    r"""Extract shadowing events from Wasserstein distances using connected components.

    Main entry point for the 2D pathfinding algorithm: collects close passes
    below ``threshold``, groups them into 8-connected components, and finds
    the longest valid path through each. Returns one
    :class:`~ks_shadowing.core.event.ShadowingEvent` per component (if longer
    than ``min_duration``).

    Parameters
    ----------
    distance_matrix : NDArray[np.float64], shape (num_timesteps, period)
        Distance matrix (typically after time-delay embedding).
    rpo_data : _RPOPersistence
        Precomputed RPO persistence data.
    threshold : float
        Maximum Wasserstein distance for close passes.
    min_duration : int
        Minimum event duration in timesteps.

    Returns
    -------
    list[ShadowingEvent]
        Events sorted by ``start_timestep``.
    """
    close_passes = _collect_close_passes_2d(distance_matrix, threshold)
    if len(close_passes) == 0:
        return []

    num_timesteps, period = distance_matrix.shape
    components = _find_connected_components_2d(close_passes, period, num_timesteps)
    events: list[ShadowingEvent] = []

    for component in components:
        finder = _ComponentPathFinder2D(component, period)
        result = finder.find_longest_path()

        if result is None or len(result[0]) < min_duration:
            continue

        path, mean_distance, min_distance = result

        start_timestep = int(path["timestep"][0])
        end_timestep = int(path["timestep"][-1]) + 1
        duration = end_timestep - start_timestep

        # PHA doesn't track shifts during pathfinding - fill with zeros.
        # The detector computes shifts post-hoc before returning events.
        shifts = np.zeros(duration, dtype=np.int32)

        events.append(
            ShadowingEvent(
                rpo_index=rpo_data.index,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                mean_distance=mean_distance,
                min_distance=min_distance,
                start_phase=int(path["phase"][0]),
                shifts=shifts,
            )
        )

    events.sort(key=lambda e: e.start_timestep)
    return events
