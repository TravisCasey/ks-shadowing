"""Plotting helpers for shadowing visualization CLI."""

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core import DOMAIN_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


def _align_rpo_to_window(  # noqa: PLR0913
    rpo: RPO,
    event: ShadowingEvent,
    window_start: int,
    window_end: int,
    resolution: int,
    downsample: int = 1,
) -> NDArray[np.float64]:
    r"""Reconstruct the RPO in the lab frame, spatially aligned to the trajectory.

    For each chaotic-trajectory timestep in the plot window, computes the
    matching RPO native phase and applies the spatial shift that best aligns
    it with the trajectory. The shift combines two contributions: the RPO's
    cumulative spatial drift (one ``spatial_shift`` per full RPO native
    period traversed) and the co-moving frame deviation recorded in
    ``event.shifts``.

    The RPO field is doubled along the spatial axis so that wraparound
    extraction (slicing across the periodic boundary) can be done with simple
    indexing.

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
    downsample : int, optional
        Sampling stride used to build the per-RPO trajectory during
        detection. Each chaotic-trajectory timestep advances the RPO by
        ``downsample`` native phases. Default 1.

    Returns
    -------
    NDArray[np.float64], shape (window_end - window_start, resolution)
        RPO field in physical space, spatially shifted to align with the
        trajectory at each timestep.
    """
    rpo_trajectory = KSTrajectory.from_initial_state(rpo.modes, rpo.dt, rpo.time_steps, resolution)
    rpo_physical = rpo_trajectory.to_physical()

    rpo_doubled = np.tile(rpo_physical, (1, 2))
    drift_per_native_step = rpo.spatial_shift * resolution / DOMAIN_SIZE / rpo.time_steps
    mean_shift = int(np.round(np.mean(event.shifts)))

    # Number of native RPO phases since event.start_timestep. event.start_phase
    # is an index into the per-RPO trajectory used during detection; the
    # corresponding native phase is event.start_phase * downsample (mod
    # rpo.time_steps), and each chaotic-trajectory timestep advances the RPO by
    # downsample more native phases.
    timesteps = np.arange(window_start, window_end)
    unrolled = (event.start_phase + timesteps - event.start_timestep) * downsample
    native_phases = unrolled % rpo.time_steps

    # Lab-frame shift to align the RPO at native_phase with the trajectory at
    # chaotic row i: undo the comoving correction (drift_per_native_step pixels
    # of drift per native step) accumulated from native phase 0 to chaotic
    # native time downsample*i, and apply the event's mean comoving deviation.
    drift_pixels = np.round(
        drift_per_native_step * (downsample * timesteps - native_phases)
    ).astype(np.int64)
    lab_shift = mean_shift - drift_pixels
    extraction_offsets = lab_shift % resolution

    spatial_indices = extraction_offsets[:, np.newaxis] + np.arange(resolution)
    return rpo_doubled[native_phases[:, np.newaxis], spatial_indices]
