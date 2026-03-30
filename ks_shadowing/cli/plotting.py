"""Plotting helpers for shadowing visualization CLI."""

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core import DOMAIN_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


def _align_rpo_to_window(
    rpo: RPO,
    event: ShadowingEvent,
    window_start: int,
    window_end: int,
    resolution: int,
) -> NDArray[np.float64]:
    r"""Reconstruct the RPO in the lab frame, spatially aligned to the trajectory.

    For each timestep in the plot window, computes the RPO phase and applies the
    spatial shift that best aligns it with the trajectory. The shift combines two
    contributions: the RPO's spatial drift (its velocity in grid cells per timestep)
    and the co-moving frame deviation recorded in ``event.shifts``.

    The RPO field is doubled along the spatial axis so that wraparound extraction
    (slicing across the periodic boundary) can be done with simple indexing.

    Parameters
    ----------
    rpo : RPO
        The RPO to align.
    event : ShadowingEvent
        Event containing ``start_phase`` and ``shifts`` for alignment.
    window_start : int
        First trajectory timestep of the plot window (inclusive).
    window_end : int
        Last trajectory timestep of the plot window (exclusive).
    resolution : int
        Number of spatial grid points.

    Returns
    -------
    NDArray[np.float64], shape (window_end - window_start, resolution)
        RPO field in physical space, spatially shifted to align with the
        trajectory at each timestep.
    """
    rpo_dt = rpo.period / rpo.time_steps
    rpo_trajectory = KSTrajectory.from_initial_state(
        rpo.fourier_coeffs, rpo_dt, rpo.time_steps + 1, resolution
    )[:-1]
    rpo_physical = rpo_trajectory.to_physical()

    period = rpo_physical.shape[0]
    # Double the spatial axis for wraparound extraction
    rpo_doubled = np.tile(rpo_physical, (1, 2))

    # RPO spatial velocity in grid cells per timestep
    drift_per_step = (rpo.spatial_shift / DOMAIN_SIZE) * resolution / period
    mean_shift = int(np.round(np.mean(event.shifts)))

    timesteps = np.arange(window_start, window_end)
    phases = (event.start_phase + timesteps - event.start_timestep) % period

    # Lab-frame shift: undo co-moving transform and apply event deviation.
    lab_shift = mean_shift - np.round(drift_per_step * (timesteps - phases)).astype(np.int64)
    extraction_offsets = lab_shift % resolution

    spatial_indices = extraction_offsets[:, np.newaxis] + np.arange(resolution)
    return rpo_doubled[phases[:, np.newaxis], spatial_indices]
