"""Shift reconstruction for PHA shadowing events.

PHA detection quotients out spatial shifts via persistence diagrams, so events
have zero-filled `shifts` arrays. This module computes shifts post-hoc using
L2 distances in the co-moving frame, subject to the constraint that each step
changes by at most 1.
"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy import fft

from ks_shadowing.core import DOMAIN_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import (
    _tile_periodic,
    interleaved_to_physical,
    to_comoving_frame,
)


def compute_event_shifts(
    event: ShadowingEvent,
    trajectory_fourier: NDArray[np.floating],
    rpo: RPO,
    resolution: int,
) -> ShadowingEvent:
    """Compute spatial shifts for a PHA shadowing event.

    Returns a new event with the `shifts` field populated. The shifts minimize
    L2 distance in the co-moving frame, subject to each step changing by at
    most (-1, 0, or +1).

    Args:
        event: The shadowing event to compute shifts for.
        trajectory_fourier: Full trajectory in Fourier space, shape `(num_timesteps, 30)`.
        rpo: The RPO that was shadowed.
        resolution: Spatial resolution for physical space computation.

    Returns:
        New ShadowingEvent with computed shifts.
    """
    period = rpo.time_steps

    # Integrate RPO for one period and convert to physical space
    rpo_dt = rpo.period / period
    rpo_fourier = ksint(rpo.fourier_coeffs, rpo_dt, period)[:-1]
    rpo_physical = interleaved_to_physical(rpo_fourier, resolution)

    # Extract the relevant slice of the trajectory
    start = event.start_timestep
    end = event.end_timestep
    duration = end - start

    # Convert trajectory slice to physical space
    trajectory_physical = interleaved_to_physical(trajectory_fourier[start:end], resolution)

    # Compute drift rate and transform to co-moving frame
    drift_per_step = (rpo.spatial_shift / DOMAIN_SIZE) * resolution / period
    trajectory_comoving = to_comoving_frame(trajectory_physical, drift_per_step)
    rpo_comoving = to_comoving_frame(rpo_physical, drift_per_step)

    # Tile RPO to cover the event duration with the correct phase alignment
    rpo_tiled = _tile_periodic(rpo_comoving, duration + period)

    # Compute distances for each timestep to all shifts
    distances = compute_distance_matrix(trajectory_comoving, rpo_tiled, event.start_phase)
    shifts = find_optimal_shifts(distances, resolution)

    return replace(event, shifts=shifts.astype(np.int32))


def compute_distance_matrix(
    trajectory_comoving: NDArray[np.float64],
    rpo_tiled: NDArray[np.float64],
    start_phase: int,
) -> NDArray[np.float64]:
    """Compute L2 distance matrix between trajectory and RPO for all shifts.

    Uses FFT-based cross-correlation for efficient computation of distances
    to all spatial shifts simultaneously.
    """
    duration = trajectory_comoving.shape[0]
    resolution = trajectory_comoving.shape[1]

    # Extract the RPO slice aligned with the event phase
    rpo_slice = rpo_tiled[start_phase : start_phase + duration]

    # Compute FFTs for cross-correlation
    traj_fft = fft.rfft(trajectory_comoving, axis=-1)
    rpo_fft = fft.rfft(rpo_slice, axis=-1)

    # Norms squared
    norm_traj_sq = np.sum(trajectory_comoving**2, axis=-1)
    norm_rpo_sq = np.sum(rpo_slice**2, axis=-1)

    # Cross-correlation via FFT: <u, roll(v, -s)> for all s
    cross_corr = fft.irfft(np.conj(traj_fft) * rpo_fft, resolution, axis=-1)

    # L2 distance squared: ||u - roll(v, -s)||^2 = ||u||^2 + ||v||^2 - 2<u, roll(v, -s)>
    dist_sq = norm_traj_sq[:, np.newaxis] + norm_rpo_sq[:, np.newaxis] - 2 * cross_corr
    dist_sq = np.maximum(dist_sq, 0.0)  # Numerical safety

    return np.sqrt(dist_sq)


def find_optimal_shifts(
    distances: NDArray[np.float64],
    resolution: int,
) -> NDArray[np.int32]:
    """Find optimal shift sequence using dynamic programming.

    Minimizes total distance subject to the constraint that each consecutive
    shift differs by at most 1 (with wraparound).
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
