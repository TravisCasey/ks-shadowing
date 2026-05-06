"""Shift reconstruction for PHA shadowing events.

PHA detection quotients out spatial shifts via persistence diagrams, so the
:class:`ks_shadowing.core.event.ShadowingEvent` instances initially have
zero-filled ``shifts`` arrays. This module computes shifts post-hoc by
minimizing :math:`L_2` distances in the co-moving physical frame, subject to the
constraint that the relative shift at each step changes by at most 1.
"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq


def _compute_event_shifts(
    event: ShadowingEvent,
    trajectory: KSTrajectory,
    drift_rate: float,
    rpo_comoving: KSTrajectory,
) -> ShadowingEvent:
    """Compute spatial shifts for a PHA shadowing event.

    Returns a new event with the ``shifts`` field populated. The shifts
    minimize :math:`L_2` distance in the co-moving frame, subject to the shift
    at each step changing by at most ``(-1, 0, +1)``.

    Parameters
    ----------
    event : :class:`~ks_shadowing.core.event.ShadowingEvent`
        The shadowing event to compute shifts for.
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Full trajectory in spectral form.
    drift_rate : float
        Spatial drift per timestep, ``rpo.spatial_shift / rpo.time_steps``.
    rpo_comoving : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        The RPO integrated for one period and transformed to its co-moving
        frame. Shared across all events against the same RPO.

    Returns
    -------
    :class:`~ks_shadowing.core.event.ShadowingEvent`
        New event with computed shifts.
    """
    period = len(rpo_comoving)
    resolution = trajectory.resolution
    duration = event.end_timestep - event.start_timestep

    traj_slice = trajectory[event.start_timestep : event.end_timestep]
    traj_comoving = traj_slice.to_comoving(drift_rate, start_timestep=event.start_timestep)

    phase_indices = (event.start_phase + np.arange(duration)) % period
    rpo_slice_modes = rpo_comoving.modes[phase_indices]

    dist_sq = shift_distances_sq(traj_comoving.modes, rpo_slice_modes, resolution)
    distances = np.sqrt(np.maximum(dist_sq, 0.0))

    shifts = _find_optimal_shifts(distances, resolution)
    return replace(event, shifts=shifts.astype(np.int32))


def _find_optimal_shifts(
    distances: NDArray[np.float64],
    resolution: int,
) -> NDArray[np.int32]:
    """Find optimal shift sequence.

    Minimizes total distance subject to the constraint that each consecutive
    shift differs by at most 1 (with wraparound).

    Parameters
    ----------
    distances : NDArray[np.float64], shape (duration, resolution)
        Distance to each shift at each timestep.
    resolution : int
        Spatial resolution (number of shift positions).

    Returns
    -------
    NDArray[np.int32], shape (duration,)
        Optimal shift at each timestep.
    """
    duration = distances.shape[0]

    if duration == 0:
        return np.array([], dtype=np.int32)

    # total_dist[t, s] = minimum total distance to reach timestep t with shift s
    total_dist = np.empty((duration, resolution), dtype=np.float64)
    predecessor = np.empty((duration, resolution), dtype=np.int32)
    total_dist[0] = distances[0]
    predecessor[0] = -1

    shift_indices = np.arange(resolution, dtype=np.int32)
    # Rows correspond to predecessor offsets (s-1, s, s+1): for target s,
    # candidates[0, s] = prev[(s - 1) % R], candidates[1, s] = prev[s], etc.
    candidates = np.empty((3, resolution), dtype=np.float64)
    for t in range(1, duration):
        prev = total_dist[t - 1]
        candidates[0] = np.roll(prev, 1)
        candidates[1] = prev
        candidates[2] = np.roll(prev, -1)
        best_offset = candidates.argmin(axis=0)
        total_dist[t] = candidates[best_offset, shift_indices] + distances[t]
        predecessor[t] = (shift_indices + (best_offset - 1)) % resolution

    # Backtrack from best ending shift
    shifts = np.empty(duration, dtype=np.int32)
    shifts[-1] = int(np.argmin(total_dist[-1]))
    for t in range(duration - 2, -1, -1):
        shifts[t] = predecessor[t + 1, shifts[t + 1]]

    return shifts
