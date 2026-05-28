"""Public helpers for shadowing analysis and visualization scripts.

These helpers are general-purpose: any analysis or figure built on
shadowing results may reuse them. Figure-specific matplotlib code lives
in the gallery scripts that own each figure (see ``examples/``).
"""

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core import DOMAIN_SIZE
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


def events_to_union_mask(events: list[ShadowingEvent], num_timesteps: int) -> NDArray[np.bool_]:
    """Boolean mask of timesteps covered by at least one event.

    Each event covers ``[start_timestep, end_timestep)``. Overlapping events
    union naturally. The result is a flat per-timestep indicator independent of
    which RPO each event matched.

    Parameters
    ----------
    events : list[ShadowingEvent]
        Events to union.
    num_timesteps : int
        Length of the trajectory the events were detected against. The
        returned mask has this length.

    Returns
    -------
    NDArray[np.bool_], shape (num_timesteps,)
        ``True`` at timesteps covered by at least one event, ``False``
        elsewhere.
    """
    mask = np.zeros(num_timesteps, dtype=np.bool_)
    for event in events:
        mask[event.start_timestep : event.end_timestep] = True
    return mask


def select_event_by_rank(
    events: list[ShadowingEvent],
    rank: int = 0,
    key: str = "mean_distance",
) -> ShadowingEvent:
    """Pick the rank-th event after sorting by a numeric field, ascending.

    With the default ``key="mean_distance"``, ``rank=0`` returns the
    best-matching event (smallest mean distance).

    Parameters
    ----------
    events : list[ShadowingEvent]
        Events to choose from. Must be non-empty.
    rank : int, optional
        Zero-based rank into the sorted list. Default 0.
    key : str, optional
        Name of a numeric :class:`~ks_shadowing.core.event.ShadowingEvent`
        attribute to sort by. Default ``"mean_distance"``.

    Returns
    -------
    ShadowingEvent
        The rank-th event after sorting.

    Raises
    ------
    IndexError
        If ``rank`` is out of range for ``events``.
    """
    if not events:
        raise IndexError("cannot select from empty event list")
    sorted_events = sorted(events, key=lambda event: getattr(event, key))
    if rank < 0 or rank >= len(sorted_events):
        raise IndexError(f"rank {rank} is out of range for {len(sorted_events)} events")
    return sorted_events[rank]


def assert_same_trajectory(traj_a: KSTrajectory, traj_b: KSTrajectory) -> None:
    """Raise ``ValueError`` if two trajectories are not bit-identical.

    Compares length, resolution, and modes element-wise. Used when two
    result files are combined (e.g., matched-event analyses) to verify
    they were detected against the same trajectory.

    Parameters
    ----------
    traj_a, traj_b : KSTrajectory
        Trajectories to compare.

    Raises
    ------
    ValueError
        If the trajectories differ in length, resolution, or modes
        content.
    """
    if traj_a.num_timesteps != traj_b.num_timesteps:
        raise ValueError(
            f"Trajectories have different lengths: {traj_a.num_timesteps} vs {traj_b.num_timesteps}"
        )
    if traj_a.resolution != traj_b.resolution:
        raise ValueError(
            f"Trajectories have different resolutions: {traj_a.resolution} vs {traj_b.resolution}"
        )
    if traj_a.dt != traj_b.dt:
        raise ValueError(f"Trajectories have different dt: {traj_a.dt} vs {traj_b.dt}")
    if not np.array_equal(traj_a.modes, traj_b.modes):
        raise ValueError("Trajectories' modes arrays differ.")


def align_rpo_to_window(
    rpo: RPO,
    event: ShadowingEvent,
    window_start: int,
    window_end: int,
    trajectory: KSTrajectory,
) -> NDArray[np.float64]:
    r"""Reconstruct an RPO in the lab frame, aligned to a trajectory window.

    Integrates the RPO at its native ``dt`` for ``rpo.time_steps`` steps, then
    for each row ``t`` in ``[window_start, window_end)`` returns the native RPO
    field at the matching native phase, spatially shifted so that it aligns with
    the trajectory at row ``t``.

    The helper is agnostic to the detection-time ``downsample`` and ``native``
    flags. It derives the integer number of native RPO phases per
    saved-trajectory row from ``round(trajectory.dt / rpo.dt)``, and the per-row
    co-moving spatial shift from ``event.shifts`` (clamped to nearest-end
    outside ``[event.start_timestep, event.end_timestep)``).

    Parameters
    ----------
    rpo : RPO
        The RPO that ``event`` matched.
    event : ShadowingEvent
        Event whose ``start_phase`` and ``shifts`` drive the alignment.
    window_start : int
        First trajectory timestep to render (inclusive).
    window_end : int
        Trajectory timestep after the last rendered row (exclusive).
    trajectory : KSTrajectory
        Trajectory whose ``dt`` sets the time-per-row scale; resolution is used
        as the spatial grid for the output.

    Returns
    -------
    NDArray[np.float64], shape (window_end - window_start, resolution)
        RPO field in physical space, per-row spatially shifted so that
        row ``i`` aligns with trajectory row ``window_start + i``.
    """
    resolution = trajectory.resolution
    rpo_physical = KSTrajectory.from_initial_state(
        rpo.modes, rpo.dt, rpo.time_steps, resolution
    ).to_physical()

    drift_pixels_per_time = rpo.drift_rate * resolution / DOMAIN_SIZE
    spatial_shift_pixels = rpo.spatial_shift * resolution / DOMAIN_SIZE
    drift_offset = (
        drift_pixels_per_time * (event.start_timestep - event.start_phase) * trajectory.dt
    )

    window_rows = np.arange(window_start, window_end)
    unrolled_native = (event.start_phase + window_rows - event.start_timestep) * round(
        trajectory.dt / rpo.dt
    )
    native_phase = unrolled_native % rpo.time_steps
    wraps = unrolled_native // rpo.time_steps
    window_shifts = event.shifts[
        np.clip(window_rows - event.start_timestep, 0, len(event.shifts) - 1)
    ]

    extraction = window_shifts - drift_offset - wraps * spatial_shift_pixels
    extraction_offsets = np.round(extraction).astype(np.int64) % resolution

    window_len = window_end - window_start
    aligned = np.empty((window_len, resolution), dtype=np.float64)
    for i in range(window_len):
        aligned[i] = np.roll(rpo_physical[native_phase[i]], -extraction_offsets[i])

    return aligned
