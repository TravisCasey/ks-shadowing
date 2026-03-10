"""Shadowing event representation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ShadowingEvent:
    """A detected shadowing episode between a trajectory and an
    :class:`~ks_shadowing.core.rpo.RPO`.

    Represents a contiguous range of timesteps where a trajectory closely
    follows an RPO in the orbit's co-moving reference frame. The
    ``start_timestep`` is inclusive and ``end_timestep`` is exclusive.

    In the co-moving frame, the RPO is truly periodic (no spatial drift).
    The ``shifts`` array contains the spatial shift deviation from perfect
    alignment at each timestep. The ``start_phase`` is the RPO phase index
    at the start of the event; at trajectory timestep ``start_timestep + i``,
    the RPO phase is ``(start_phase + i) % period``.

    Attributes
    ----------
    rpo_index : int
        Index of the shadowed RPO in the data file it was loaded from.
    start_timestep : int
        First trajectory timestep of the event (inclusive).
    end_timestep : int
        Trajectory timestep after the last included timestep (exclusive).
    mean_distance : float
        Mean distance between trajectory and RPO over the event.
    min_distance : float
        Minimum distance between trajectory and RPO over the event.
    start_phase : int
        RPO phase index at ``start_timestep``. Phase at timestep ``t`` is
        ``(start_phase + t - start_timestep) % period``.
    shifts : NDArray[np.int32], shape (end_timestep - start_timestep,)
        Spatial shift deviations in the co-moving frame, one per timestep.
    """

    rpo_index: int
    start_timestep: int
    end_timestep: int
    mean_distance: float
    min_distance: float
    start_phase: int
    shifts: NDArray[np.int32]
