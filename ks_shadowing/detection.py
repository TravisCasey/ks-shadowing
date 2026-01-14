"""Shadowing event detection utilities.

This module provides generic event detection infrastructure that can be reused
by both the State Space Approach (SSA) and Persistent Homology Approach (PHA).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ShadowingEvent:
    """A detected shadowing episode between a trajectory and an RPO.

    A shadowing event represents a contiguous time interval where a chaotic
    trajectory closely follows an RPO, with the RPO phase advancing by 1 at
    each trajectory timestep (with wraparound at the period boundary).

    Note that the start time is inclusive (the first timestep at which the
    trajectory shadows the RPO) but the end time is exclusive (the first
    timestep at which the trajectory no longer shadows the RPO).
    """

    rpo_index: int
    start_time: int
    end_time: int
    mean_distance: float
    min_distance: float

    @property
    def duration(self) -> int:
        """Number of timesteps in the shadowing event."""
        return self.end_time - self.start_time


@dataclass
class PathBuffer:
    """Tracks active shadowing paths for one timestep.

    Used by the streaming DP algorithm to maintain state for paths that are
    currently being extended. Two buffers are alternated to correctly handle the
    temporal dependency between timesteps.

    In other words, updating an entry of the buffer depends on other entries of
    this buffer; it needs these values to be from the current timestep. However,
    these entries are also update this timestep. We use a second buffer to write
    current timestep values to and preserve previous timestep values in the
    other.
    """

    rpo_count: int
    max_period: int
    path_length: NDArray[np.int32]
    distance_sum: NDArray[np.float64]
    min_distance: NDArray[np.float64]
    start_time: NDArray[np.int32]

    @classmethod
    def empty(cls, rpo_count: int, max_period: int) -> Self:
        """Create a buffer with no active paths."""
        return cls(
            rpo_count=rpo_count,
            max_period=max_period,
            path_length=np.zeros((rpo_count, max_period), dtype=np.int32),
            distance_sum=np.zeros((rpo_count, max_period), dtype=np.float64),
            min_distance=np.full((rpo_count, max_period), np.inf, dtype=np.float64),
            start_time=np.zeros((rpo_count, max_period), dtype=np.int32),
        )

    def reset(self) -> None:
        """Clear all paths for reuse as the buffer to be written to."""
        self.path_length.fill(0)
        self.distance_sum.fill(0.0)
        self.min_distance.fill(np.inf)
        self.start_time.fill(0)

    def start_path(self, rpo: int, phase: int, distance: float, time: int) -> None:
        """Begin a new path at the given position."""
        self.path_length[rpo, phase] = 1
        self.distance_sum[rpo, phase] = distance
        self.min_distance[rpo, phase] = distance
        self.start_time[rpo, phase] = time

    def extend_from(
        self, prev: Self, rpo: int, phase: int, pred_phase: int, distance: float
    ) -> None:
        """Extend a path from the previous buffer into this buffer."""
        self.path_length[rpo, phase] = prev.path_length[rpo, pred_phase] + 1
        self.distance_sum[rpo, phase] = prev.distance_sum[rpo, pred_phase] + distance
        self.min_distance[rpo, phase] = min(prev.min_distance[rpo, pred_phase], distance)
        self.start_time[rpo, phase] = prev.start_time[rpo, pred_phase]

    def to_event(self, rpo: int, phase: int) -> ShadowingEvent:
        """Create a ShadowingEvent from an active path."""
        length = int(self.path_length[rpo, phase])
        start = int(self.start_time[rpo, phase])
        return ShadowingEvent(
            rpo_index=rpo,
            start_time=start,
            end_time=start + length,
            mean_distance=float(self.distance_sum[rpo, phase] / length),
            min_distance=float(self.min_distance[rpo, phase]),
        )

    def is_active(self, rpo: int, phase: int) -> bool:
        """Check if there is an active path at this position."""
        return self.path_length[rpo, phase] > 0

    def is_continued_by(self, successor: Self, rpo: int, phase: int, period: int) -> bool:
        """Check if the path at (rpo, phase) was continued in the successor buffer.

        A path is continued if the successor buffer has an active path at the
        next phase (with wraparound) that has length exactly one greater and
        identical start time.
        """
        succ_phase = (phase + 1) % period
        return (
            successor.path_length[rpo, succ_phase] == self.path_length[rpo, phase] + 1
            and successor.start_time[rpo, succ_phase] == self.start_time[rpo, phase]
        )


def update_paths_for_timestep(  # noqa: PLR0913
    prev_buf: PathBuffer,
    curr_buf: PathBuffer,
    distances: NDArray[np.float64],
    rpo_periods: list[int],
    threshold: float,
    current_time: int,
) -> None:
    """Update curr_buf by extending or starting paths based on distances.

    For each (rpo, phase) where the distance is below threshold, either extends
    an existing path from the predecessor phase or starts a new path.
    """
    for rpo_idx, period in enumerate(rpo_periods):
        for phase in range(period):
            if distances[rpo_idx, phase] < threshold:
                pred_phase = (phase - 1) % period
                dist = float(distances[rpo_idx, phase])

                if prev_buf.is_active(rpo_idx, pred_phase):
                    curr_buf.extend_from(prev_buf, rpo_idx, phase, pred_phase, dist)
                else:
                    curr_buf.start_path(rpo_idx, phase, dist, current_time)


def collect_terminated_events(
    prev_buf: PathBuffer,
    curr_buf: PathBuffer,
    rpo_periods: list[int],
    min_duration: int,
) -> list[ShadowingEvent]:
    """Collect events for paths in prev_buf that were not continued in curr_buf."""
    events = []
    for rpo_idx, period in enumerate(rpo_periods):
        for phase in range(period):
            if prev_buf.path_length[rpo_idx, phase] < min_duration:
                continue
            if not prev_buf.is_continued_by(curr_buf, rpo_idx, phase, period):
                events.append(prev_buf.to_event(rpo_idx, phase))
    return events


def collect_active_events(
    buf: PathBuffer,
    rpo_periods: list[int],
    min_duration: int,
) -> list[ShadowingEvent]:
    """Collect events for all paths still active in the buffer."""
    events = []
    for rpo_idx, period in enumerate(rpo_periods):
        for phase in range(period):
            if buf.path_length[rpo_idx, phase] >= min_duration:
                events.append(buf.to_event(rpo_idx, phase))
    return events


def extract_shadowing_events(
    distance_generator: Iterator[NDArray[np.float64]],
    rpo_periods: list[int],
    threshold: float,
    min_duration: int = 1,
) -> list[ShadowingEvent]:
    """Extract shadowing events from streaming distance computation.

    Uses dynamic programming to find contiguous paths through the distance array
    where all distances are below `threshold` and RPO phase advances
    by one each trajectory timestep (with wraparound at period boundary).

    The algorithm processes distances one trajectory timestep at a time,
    maintaining `PathBuffer` objects that track path lengths, accumulated
    distances, and start times. Events are emitted when paths terminate
    (distance exceeds threshold) or when the trajectory ends.

    Args:
        distance_generator: Yields `(rpo_count, max_period)` distance arrays,
            one per trajectory timestep. Invalid phases (beyond an RPO's
            period) should contain infinity.
        rpo_periods: List of period lengths for each RPO.
        threshold: Distance threshold for shadowing detection.
        min_duration: Minimum path length to report as an event.

    Returns:
        List of detected shadowing events, sorted by start time.
    """
    rpo_count = len(rpo_periods)
    if rpo_count == 0:
        return []

    max_period = max(rpo_periods)
    prev_buf = PathBuffer.empty(rpo_count, max_period)
    curr_buf = PathBuffer.empty(rpo_count, max_period)
    events: list[ShadowingEvent] = []

    for current_time, distances in enumerate(distance_generator):
        curr_buf.reset()
        update_paths_for_timestep(
            prev_buf, curr_buf, distances, rpo_periods, threshold, current_time
        )
        events.extend(collect_terminated_events(prev_buf, curr_buf, rpo_periods, min_duration))
        prev_buf, curr_buf = curr_buf, prev_buf

    events.extend(collect_active_events(prev_buf, rpo_periods, min_duration))
    events.sort(key=lambda e: (e.start_time, e.rpo_index))
    return events
