"""Shadowing event characterization via longest pathfinding algorithm (2D version).

This module implements the graph-based algorithm for extracting shadowing events
from Wasserstein distance data in 2D space (timestep x phase):

1. Collect "close passes" - points where trajectory is within threshold of an RPO
2. Group close passes into connected components (8-connectivity in 2D grid)
3. Find the longest valid path through each component

A valid path must satisfy temporal co-evolution: both trajectory timestep and
RPO phase advance by 1 at each step.
"""

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.util import UnionFind
from ks_shadowing.pha.persistence import RPOPersistence

# Structured dtype for close passes in 2D (timestep, phase).
# No shift dimension since PHA quotients out spatial symmetry.
CLOSE_PASS_DTYPE_2D = np.dtype(
    [
        ("timestep", np.int32),
        ("phase", np.int32),
        ("distance", np.float64),
    ]
)


class ComponentPathFinder2D:
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
            p_timestep = int(self.passes["timestep"][pass_index])
            p_phase = int(self.passes["phase"][pass_index])

            best = self._find_best_predecessor(p_timestep, p_phase, path_length, dist_sum)
            if best >= 0:
                path_length[pass_index] = path_length[best] + 1
                dist_sum[pass_index] = dist_sum[best] + self.passes["distance"][pass_index]
                min_dist[pass_index] = min(min_dist[best], self.passes["distance"][pass_index])
                predecessor[pass_index] = best

        # Find best endpoint (longest path, then lowest mean distance)
        best_end = self._find_best_endpoint(path_length, dist_sum)
        if best_end < 0:
            return None

        path = self._reconstruct_path(predecessor, best_end)
        mean_distance = float(dist_sum[best_end] / len(path))
        return path, mean_distance, float(min_dist[best_end])

    def _find_best_predecessor(
        self,
        p_timestep: int,
        p_phase: int,
        path_length: NDArray[np.int32],
        dist_sum: NDArray[np.float64],
    ) -> int:
        """Find the best predecessor for a close pass, or -1 if none valid.

        In 2D, the predecessor must have timestep-1 and phase-1 (mod period).
        Both trajectory and RPO advance together.
        """
        prev_timestep = p_timestep - 1
        prev_phase = (p_phase - 1) % self.period
        prev_key = (prev_timestep, prev_phase)

        if prev_key not in self.lookup:
            return -1

        return self.lookup[prev_key]

    def _find_best_endpoint(
        self,
        path_length: NDArray[np.int32],
        dist_sum: NDArray[np.float64],
    ) -> int:
        """Find the index with longest path, breaking ties by lowest mean distance."""
        best_index = -1
        best_length = 0
        best_mean = float("inf")

        for i in range(self.pass_count):
            length = path_length[i]
            mean = dist_sum[i] / length

            if length > best_length or (length == best_length and mean < best_mean):
                best_index = i
                best_length = length
                best_mean = mean

        return best_index

    def _reconstruct_path(self, predecessor: NDArray[np.int32], end: int) -> NDArray:
        """Reconstruct path by following predecessor links."""
        indices = []
        i = end
        while i >= 0:
            indices.append(i)
            i = predecessor[i]
        indices.reverse()
        return self.passes[indices]


def collect_close_passes_2d(
    distance_matrix: NDArray[np.float64],
    threshold: float,
) -> NDArray:
    """Collect all entries below threshold from a 2D distance matrix.

    Args:
        distance_matrix: Wasserstein distance matrix of shape `(num_timesteps, period)`.
        threshold: Maximum distance for close passes.

    Returns:
        Structured array with dtype `CLOSE_PASS_DTYPE_2D`.
    """
    timestep_indices, phase_indices = np.where(distance_matrix < threshold)
    count = len(timestep_indices)

    if count == 0:
        return np.array([], dtype=CLOSE_PASS_DTYPE_2D)

    passes = np.empty(count, dtype=CLOSE_PASS_DTYPE_2D)
    passes["timestep"] = timestep_indices
    passes["phase"] = phase_indices
    passes["distance"] = distance_matrix[timestep_indices, phase_indices]

    return passes


def find_connected_components_2d(
    close_passes: NDArray,
    period: int,
) -> list[NDArray]:
    """Group close passes into connected components using 8-connectivity.

    Two points are adjacent if they differ by at most 1 in each dimension,
    with wraparound in the phase dimension.

    Args:
        close_passes: Structured array with dtype `CLOSE_PASS_DTYPE_2D`.
        period: RPO period for phase wraparound.

    Returns:
        List of structured arrays, one per connected component.
    """
    if len(close_passes) == 0:
        return []

    pass_count = len(close_passes)
    uf = UnionFind(pass_count)

    # Extract arrays for convenient access
    timesteps = close_passes["timestep"]
    phases = close_passes["phase"]

    # Build coordinate -> index mapping for sparse neighbor lookup
    coord_to_index: dict[tuple[int, int], int] = {}
    for pass_index in range(pass_count):
        coord_to_index[(int(timesteps[pass_index]), int(phases[pass_index]))] = pass_index

    # Union adjacent points (8-connectivity in 2D)
    for pass_index in range(pass_count):
        t, p = int(timesteps[pass_index]), int(phases[pass_index])
        for dt in (-1, 0, 1):
            for dp in (-1, 0, 1):
                if dt == 0 and dp == 0:
                    continue
                neighbor = (t + dt, (p + dp) % period)
                if neighbor in coord_to_index:
                    uf.union(pass_index, coord_to_index[neighbor])

    # Group by component root
    component_indices: dict[int, list[int]] = {}
    for pass_index in range(pass_count):
        root = uf.find(pass_index)
        if root not in component_indices:
            component_indices[root] = []
        component_indices[root].append(pass_index)

    return [close_passes[indices] for indices in component_indices.values()]


def extract_shadowing_events_2d(
    distance_matrix: NDArray[np.float64],
    rpo_data: RPOPersistence,
    threshold: float,
    min_duration: int,
    delay: int,
) -> list[ShadowingEvent]:
    """Extract shadowing events from Wasserstein distances using connected components.

    This is the main entry point for the 2D pathfinding algorithm. It:
    1. Collects all close passes (points below threshold)
    2. Groups them into connected components
    3. Finds the longest valid path through each component
    4. Returns one event per component (if longer than `min_duration`)

    Args:
        distance_matrix: Wasserstein distance matrix with time-delay embedding applied,
            shape `(num_timesteps - delay + 1, period)`.
        rpo_data: Precomputed RPO persistence data.
        threshold: Maximum Wasserstein distance for close passes.
        min_duration: Minimum event duration in timesteps.
        delay: Time-delay embedding window size (for offset correction).

    Returns:
        List of shadowing events sorted by start timestep.
    """
    close_passes = collect_close_passes_2d(distance_matrix, threshold)
    if len(close_passes) == 0:
        return []

    period = rpo_data.time_steps
    components = find_connected_components_2d(close_passes, period)
    events: list[ShadowingEvent] = []

    for component in components:
        finder = ComponentPathFinder2D(component, period)
        result = finder.find_longest_path()

        if result is None or len(result[0]) < min_duration:
            continue

        path, mean_distance, min_distance = result

        # The timestep in the path is relative to the delay-embedded matrix.
        # The actual trajectory timestep is the same (delay embedding doesn't shift start).
        start_timestep = int(path["timestep"][0])
        end_timestep = int(path["timestep"][-1]) + 1
        duration = end_timestep - start_timestep

        # PHA doesn't track shifts - fill with zeros
        # Shifts can be computed afterward using compute_event_shifts
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
