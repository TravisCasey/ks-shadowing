"""SSA pathfinding algorithm for shadowing event detection.

This module implements the streaming dynamic programming algorithm for detecting
shadowing events in the 3D distance space
`(trajectory_time, rpo_phase, spatial_shift)`.

The SSA algorithm enforces spatial continuity: consecutive spatial shifts can
only differ by at most 1 position (shift-1, shift, or shift+1).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.detection import ShadowingEvent


@dataclass
class SSAPathBuffer:
    """Tracks active shadowing paths for one timestep in SSA detection.

    Used by the streaming DP algorithm to maintain state for paths that are
    currently being extended. Two buffers are alternated to handle temporal
    dependencies between timesteps.

    Arrays have shape `(rpo_period, spatial_resolution)` for a single RPO,
    reflecting the 3D structure of SSA's distance space where spatial shift
    continuity is enforced.
    """

    path_length: NDArray[np.int32]
    distance_sum: NDArray[np.float64]
    min_distance: NDArray[np.float64]
    start_time: NDArray[np.int32]

    @classmethod
    def empty(cls, period: int, resolution: int) -> Self:
        """Create a buffer with no active paths."""
        shape: tuple[int, int] = (period, resolution)
        return cls(
            path_length=np.zeros(shape, dtype=np.int32),
            distance_sum=np.zeros(shape, dtype=np.float64),
            min_distance=np.full(shape, np.inf, dtype=np.float64),
            start_time=np.zeros(shape, dtype=np.int32),
        )

    def reset(self) -> None:
        """Clear all paths for reuse."""
        self.path_length.fill(0)
        self.distance_sum.fill(0.0)
        self.min_distance.fill(np.inf)
        self.start_time.fill(0)

    def to_event(self, rpo_index: int, phase: int, shift: int) -> ShadowingEvent:
        """Create a ShadowingEvent from an active path."""
        length = int(self.path_length[phase, shift])
        start = int(self.start_time[phase, shift])
        return ShadowingEvent(
            rpo_index=rpo_index,
            start_time=start,
            end_time=start + length,
            mean_distance=float(self.distance_sum[phase, shift] / length),
            min_distance=float(self.min_distance[phase, shift]),
        )


def compute_best_predecessors(
    prev_buf: SSAPathBuffer,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """Compute best predecessor values for each `(phase, shift)` entry.

    For each cell, finds the predecessor with the longest path among the three
    valid predecessors allowed by SSA's spatial continuity constraint:
        - `(phase-1, shift-1)`,
        - `(phase-1, shift)`,
        - `(phase-1, shift+1)`.

    Returns arrays for:
        - best `path_length`,
        - best `distance_sum`,
        - best `min_distance`,
        - best `start_time`.
    """
    # Roll phase dimension to get predecessor phase (phase-1) % period
    pred_len = np.roll(prev_buf.path_length, 1, axis=0)

    # Stack three shift-rolled versions and find best
    len_stack = np.stack(
        [np.roll(pred_len, 1, axis=1), pred_len, np.roll(pred_len, -1, axis=1)], axis=0
    )
    best_index = np.argmax(len_stack, axis=0)
    best_length = np.take_along_axis(len_stack, best_index[np.newaxis, :, :], axis=0)[0]

    # Apply same rolls to other arrays and select by best_index
    pred_dist = np.roll(prev_buf.distance_sum, 1, axis=0)
    dist_stack = np.stack(
        [
            np.roll(pred_dist, 1, axis=1),
            pred_dist,
            np.roll(pred_dist, -1, axis=1),
        ],
        axis=0,
    )
    best_dist_sum = np.take_along_axis(dist_stack, best_index[None, :, :], axis=0)[0]

    pred_min = np.roll(prev_buf.min_distance, 1, axis=0)
    min_stack = np.stack(
        [
            np.roll(pred_min, 1, axis=1),
            pred_min,
            np.roll(pred_min, -1, axis=1),
        ],
        axis=0,
    )
    best_min_dist = np.take_along_axis(min_stack, best_index[None, :, :], axis=0)[0]

    pred_start = np.roll(prev_buf.start_time, 1, axis=0)
    start_stack = np.stack(
        [
            np.roll(pred_start, 1, axis=1),
            pred_start,
            np.roll(pred_start, -1, axis=1),
        ],
        axis=0,
    )
    best_start = np.take_along_axis(start_stack, best_index[None, :, :], axis=0)[0]

    return best_length, best_dist_sum, best_min_dist, best_start


def update_paths_for_timestep(
    prev_buf: SSAPathBuffer,
    curr_buf: SSAPathBuffer,
    distances: NDArray[np.float64],
    threshold: float,
    current_time: int,
) -> None:
    """Update `curr_buf` by extending or starting paths based on distances.

    For each `(phase, shift)` where distance is below threshold, either extends
    an existing path from a valid predecessor (respecting SSA's shift continuity
    constraint) or starts a new path.
    """
    curr_buf.reset()

    best_length, best_dist_sum, best_min_dist, best_start = compute_best_predecessors(prev_buf)

    below: NDArray[np.bool_] = distances < threshold
    has_pred: NDArray[np.bool_] = best_length > 0
    extend_mask: NDArray[np.bool_] = below & has_pred
    start_mask: NDArray[np.bool_] = below & ~has_pred

    # Extend existing paths
    curr_buf.path_length[:] = np.where(extend_mask, best_length + 1, 0)
    curr_buf.distance_sum[:] = np.where(extend_mask, best_dist_sum + distances, 0.0)
    curr_buf.min_distance[:] = np.where(extend_mask, np.minimum(best_min_dist, distances), np.inf)
    curr_buf.start_time[:] = np.where(extend_mask, best_start, 0)

    # Start new paths
    curr_buf.path_length[:] = np.where(start_mask, 1, curr_buf.path_length)
    curr_buf.distance_sum[:] = np.where(start_mask, distances, curr_buf.distance_sum)
    curr_buf.min_distance[:] = np.where(start_mask, distances, curr_buf.min_distance)
    curr_buf.start_time[:] = np.where(start_mask, current_time, curr_buf.start_time)


def collect_terminated_events(
    prev_buf: SSAPathBuffer,
    curr_buf: SSAPathBuffer,
    rpo_index: int,
    min_duration: int,
) -> list[ShadowingEvent]:
    """Collect events for paths in `prev_buf` that were not continued in `curr_buf`.

    A path is continued if any valid successor entry `(phase+1, shift +/- 0/1)`
    has a path with length equal to the previous length plus one and the same
    start time. This respects SSA's spatial continuity constraint.
    """
    candidates: NDArray[np.bool_] = prev_buf.path_length >= min_duration
    if not np.any(candidates):
        return []

    # Check if each candidate was continued by any valid successor
    succ_len: NDArray[np.int32] = np.roll(curr_buf.path_length, -1, axis=0)
    succ_start: NDArray[np.int32] = np.roll(curr_buf.start_time, -1, axis=0)

    continued: NDArray[np.bool_] = np.zeros_like(candidates)
    for delta in (-1, 0, 1):
        succ_len_shifted: NDArray[np.int32] = np.roll(succ_len, -delta, axis=1)
        succ_start_shifted: NDArray[np.int32] = np.roll(succ_start, -delta, axis=1)
        match: NDArray[np.bool_] = (succ_len_shifted == prev_buf.path_length + 1) & (
            succ_start_shifted == prev_buf.start_time
        )
        continued |= match

    terminated: NDArray[np.bool_] = candidates & ~continued

    events: list[ShadowingEvent] = []
    for phase, shift in np.argwhere(terminated):
        events.append(prev_buf.to_event(rpo_index, phase, shift))
    return events


def collect_active_events(
    buf: SSAPathBuffer,
    rpo_index: int,
    min_duration: int,
) -> list[ShadowingEvent]:
    """Collect events for all paths still active in the buffer."""
    events: list[ShadowingEvent] = []
    for phase, shift in np.argwhere(buf.path_length >= min_duration):
        events.append(buf.to_event(rpo_index, phase, shift))
    return events


def extract_shadowing_events(
    distance_generator: Iterator[NDArray[np.float64]],
    rpo_index: int,
    period: int,
    threshold: float,
    min_duration: int = 1,
) -> list[ShadowingEvent]:
    """Extract shadowing events for a single RPO using SSA's vectorized DP.

    Finds longest paths through distance matrix entries below `threshold` where
    RPO phase advances by 1 and spatial shift changes by at most 1 (SSA's
    spatial continuity constraint).

    Args:
        distance_generator: Yields `(period, spatial_resolution)` distance
            arrays, one per trajectory timestep.
        rpo_index: Index of this RPO.
        period: Number of phases in this RPO.
        threshold: Distance threshold for shadowing detection.
        min_duration: Minimum path length to report as an event.

    Returns:
        List of detected shadowing events, sorted by start time.
    """
    # Use first yielded array to get shape
    first_distances: NDArray[np.float64] | None = next(distance_generator, None)
    if first_distances is None:
        return []

    resolution: int = first_distances.shape[1]

    prev_buf = SSAPathBuffer.empty(period, resolution)
    curr_buf = SSAPathBuffer.empty(period, resolution)
    events: list[ShadowingEvent] = []

    # Process first timestep
    update_paths_for_timestep(prev_buf, curr_buf, first_distances, threshold, 0)
    prev_buf, curr_buf = curr_buf, prev_buf

    # Process remaining timesteps
    for current_time, distances in enumerate(distance_generator, start=1):
        update_paths_for_timestep(prev_buf, curr_buf, distances, threshold, current_time)
        events.extend(collect_terminated_events(prev_buf, curr_buf, rpo_index, min_duration))
        prev_buf, curr_buf = curr_buf, prev_buf

    events.extend(collect_active_events(prev_buf, rpo_index, min_duration))
    events.sort(key=lambda e: e.start_time)
    return events
