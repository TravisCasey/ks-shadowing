r"""Shift reconstruction for PHA shadowing events.

PHA detection quotients out spatial shifts via persistence diagrams, so events
have zero-filled ``shifts`` arrays. This module computes shifts post-hoc using
:math:`L_2` distances in the co-moving frame, subject to the constraint that
each step changes by at most 1.
"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq


def _compute_event_shifts(
    event: ShadowingEvent,
    trajectory: KSTrajectory,
    rpo: RPO,
) -> ShadowingEvent:
    r"""Compute spatial shifts for a PHA shadowing event.

    Returns a new event with the ``shifts`` field populated. The shifts
    minimize :math:`L_2` distance in the co-moving frame, subject to each step
    changing by at most ``(-1, 0, +1)``.

    Parameters
    ----------
    event : :class:`~ks_shadowing.core.event.ShadowingEvent`
        The shadowing event to compute shifts for.
    trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
        Full trajectory in spectral form.
    rpo : :class:`~ks_shadowing.core.rpo.RPO`
        The RPO that was shadowed.

    Returns
    -------
    :class:`~ks_shadowing.core.event.ShadowingEvent`
        New event with computed shifts.
    """
    period = rpo.time_steps
    resolution = trajectory.resolution

    # Integrate RPO for one period
    rpo_dt = rpo.period / period
    rpo_trajectory = KSTrajectory.from_initial_state(
        rpo.fourier_coeffs, rpo_dt, period + 1, resolution
    )[:-1]

    # Extract the relevant slice of the trajectory
    duration = event.end_timestep - event.start_timestep
    traj_slice = trajectory[event.start_timestep : event.end_timestep]

    # Compute drift rate and transform to co-moving frame
    drift_rate = rpo.spatial_shift / period
    traj_comoving = traj_slice.to_comoving(drift_rate, start_step=event.start_timestep)
    rpo_comoving = rpo_trajectory.to_comoving(drift_rate)

    # Tile RPO to cover the event duration with the correct phase alignment
    rpo_tiled = rpo_comoving.tile(duration + period)

    # Compute distances using shift_distances_sq
    rpo_slice_modes = rpo_tiled.modes[event.start_phase : event.start_phase + duration]
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
    total_dist = np.full((duration, resolution), np.inf, dtype=np.float64)
    predecessor = np.full((duration, resolution), -1, dtype=np.int32)

    # Initialize first timestep - any shift is valid
    total_dist[0, :] = distances[0, :]

    # Populate DP tables
    for t in range(1, duration):
        for s in range(resolution):
            # Valid predecessors: s-1, s, s+1 (with wraparound)
            for ds in (-1, 0, 1):
                prev_s = (s + ds) % resolution
                candidate_dist = total_dist[t - 1, prev_s] + distances[t, s]
                if candidate_dist < total_dist[t, s]:
                    total_dist[t, s] = candidate_dist
                    predecessor[t, s] = prev_s

    # Find best ending shift
    best_end_shift = int(np.argmin(total_dist[duration - 1, :]))

    # Backtrack to reconstruct path
    shifts = np.empty(duration, dtype=np.int32)
    shifts[duration - 1] = best_end_shift
    for t in range(duration - 2, -1, -1):
        shifts[t] = predecessor[t + 1, shifts[t + 1]]

    return shifts
