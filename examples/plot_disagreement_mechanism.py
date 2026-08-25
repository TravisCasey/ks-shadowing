"""
Anatomy of SSA/PHA disagreement
================================

The :ref:`agreement example <sphx_glr_auto_examples_plot_coverage_vs_embedding.py>`
scores how closely the two detection methods agree; this figure shows the state
of the system where they do not, one panel per direction of disagreement. Each
panel overlays the trajectory and aligned-RPO fields at the timestep where the
missing method's distance, normalized by its own threshold, peaks: panel (a)
inside an SSA event not detected by PHA, panel (b) inside a PHA event not
detected by SSA. Each example is chosen as the mismatch whose peak normalized
distance is largest amongst all gaps of its kind.

The two panels fail in opposite ways, and the split is what sublevel-set
persistence reads: extremum values, never extremum positions. In (a) the extrema
lie in the right places with drifting values. The states nearly coincide, so the
:math:`L_2` distance stays below the SSA threshold; but the persistence pairs
track the extreme values, so the Wasserstein distance :math:`d_{W^2}` exceeds
the PHA threshold and PHA declines the event that SSA detects. In (b) the
extrema hold the right values under slight spatial shift. The persistence
diagrams nearly coincide, so PHA detects shadowing; but the shifted peaks leave
a pointwise deviation of broad support, so the :math:`L_2` distance exceeds the
SSA threshold.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import (
    DOMAIN_SIZE,
    KSTrajectory,
    ShadowingEvent,
    align_rpo_to_window,
    assert_same_trajectory,
    events_to_union_mask,
    load_results,
    load_rpos,
    shift_distances_sq,
)
from ks_shadowing.pha import KSPersistenceTrajectory, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
SSA_PATH = REPO_ROOT / "examples" / "data" / "ssa_r2048.h5"
PHA_PATH = REPO_ROOT / "examples" / "data" / "pha_r2048_d1_o0.h5"
# Gaps shorter than this are ignored.
MIN_GAP_TIMESTEPS = 3
# Timesteps of context shown on each side of the gap.
CONTEXT_TIMESTEPS = 15
# Black marks the trajectory and red the RPO, following the
# persistence-diagram example.
TRAJECTORY_COLOR = "black"
RPO_COLOR = "#EE6677"

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load both result files, which must share one trajectory, and build each RPO's
# one-period trajectory, its co-moving modes, and its per-phase persistence
# diagrams.
ssa_metadata, trajectory, ssa_events = load_results(SSA_PATH)
pha_metadata, pha_trajectory, pha_events = load_results(PHA_PATH)
assert_same_trajectory(trajectory, pha_trajectory)
resolution = trajectory.resolution
num_timesteps = trajectory.num_timesteps
rpos = load_rpos(REPO_ROOT / ssa_metadata.rpo_file)

rpo_trajectories = [
    KSTrajectory.from_rpo(rpo, resolution, ssa_metadata.downsample, ssa_metadata.native)
    for rpo in rpos
]
rpo_comoving_modes = [
    rpo_trajectory.to_comoving(rpo.drift_rate).modes
    for rpo, rpo_trajectory in zip(rpos, rpo_trajectories, strict=True)
]
rpo_persistence = [
    KSPersistenceTrajectory.from_trajectory(rpo_trajectory) for rpo_trajectory in rpo_trajectories
]


# %%
# Gaps. An event is compared against the other method's per-RPO coverage;
# maximal uncovered runs that touch neither end of the event's window are gaps
# the other method could have covered but did not.
def _per_rpo_masks(events: list[ShadowingEvent]) -> dict[int, NDArray[np.bool_]]:
    """Union coverage mask per RPO index."""
    events_by_rpo: dict[int, list[ShadowingEvent]] = defaultdict(list)
    for event in events:
        events_by_rpo[event.rpo_index].append(event)
    return {
        rpo_index: events_to_union_mask(rpo_events, num_timesteps)
        for rpo_index, rpo_events in events_by_rpo.items()
    }


def _interior_gaps(host: ShadowingEvent, other_mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal uncovered runs strictly inside the host event's window."""
    covered = other_mask[host.start_timestep : host.end_timestep]
    edges = np.diff(np.concatenate(([True], covered, [True])).astype(np.int8))
    starts = np.flatnonzero(edges == -1)
    ends = np.flatnonzero(edges == 1)
    keep = (starts > 0) & (ends < covered.size) & (ends - starts >= MIN_GAP_TIMESTEPS)
    return [
        (host.start_timestep + int(start), host.start_timestep + int(end))
        for start, end in zip(starts[keep], ends[keep], strict=True)
    ]


# %%
# Distances along the event's phase track. Within an event the RPO phase
# advances by one per trajectory timestep modulo the RPO trajectory's period.
def _track_distances(
    host: ShadowingEvent, window_start: int, window_end: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], KSPersistenceTrajectory, NDArray[np.int64]]:
    """Threshold-normalized SSA and PHA distances along ``host``'s phase track.

    Returns the two normalized distance arrays over ``[window_start,
    window_end)``, the window's persistence diagrams, and the RPO phase at
    each window row.
    """
    rpo = rpos[host.rpo_index]
    period = rpo_trajectories[host.rpo_index].num_timesteps
    rows = np.arange(window_start, window_end)
    phases = (host.start_phase + rows - host.start_timestep) % period

    window = trajectory[window_start:window_end]
    window_comoving = window.to_comoving(rpo.drift_rate, start_time=window_start * trajectory.dt)
    distances_sq = shift_distances_sq(
        window_comoving.modes, rpo_comoving_modes[host.rpo_index][phases], resolution
    )
    # The stored SSA threshold is an L2 distance; detection compares squared
    # distances against its square, so take the root before normalizing.
    ssa_normalized = np.sqrt(np.maximum(distances_sq, 0.0).min(axis=1)) / ssa_metadata.threshold

    window_persistence = KSPersistenceTrajectory.from_trajectory(window)
    orbit_persistence = rpo_persistence[host.rpo_index]
    # One Wasserstein value per (window row, matching phase) pair.
    wasserstein = np.array(
        [
            wasserstein_matrix(
                window_persistence[index : index + 1],
                orbit_persistence[phase : phase + 1],
            )[0, 0]
            for index, phase in enumerate(phases)
        ]
    )
    pha_normalized = wasserstein / pha_metadata.threshold
    return ssa_normalized, pha_normalized, window_persistence, phases


# %%
# Select each column's gap: the one whose missing-method normalized distance
# peaks highest over the gap itself.
columns: list[dict[str, Any]] = []
for name, host_events, other_events, missing in (
    (r"$\mathtt{SSA}$ detects shadowing; $\mathtt{PHA}$ does not", ssa_events, pha_events, "pha"),
    (r"$\mathtt{PHA}$ detects shadowing; $\mathtt{SSA}$ does not", pha_events, ssa_events, "ssa"),
):
    other_masks = _per_rpo_masks(other_events)
    candidates = [
        (host, gap)
        for host in host_events
        if host.rpo_index in other_masks
        for gap in _interior_gaps(host, other_masks[host.rpo_index])
    ]

    best_peak = -np.inf
    for host, (gap_start, gap_end) in candidates:
        ssa_normalized, pha_normalized, _, _ = _track_distances(host, gap_start, gap_end)
        peak = float((pha_normalized if missing == "pha" else ssa_normalized).max())
        if peak > best_peak:
            best_peak = peak
            best_host, best_gap = host, (gap_start, gap_end)

    gap_start, gap_end = best_gap
    window_start = max(best_host.start_timestep, gap_start - CONTEXT_TIMESTEPS)
    window_end = min(best_host.end_timestep, gap_end + CONTEXT_TIMESTEPS)
    ssa_normalized, pha_normalized, _, _ = _track_distances(best_host, window_start, window_end)
    missing_normalized = pha_normalized if missing == "pha" else ssa_normalized
    gap_slice = slice(gap_start - window_start, gap_end - window_start)
    peak_index = gap_slice.start + int(np.argmax(missing_normalized[gap_slice]))
    peak_timestep = window_start + peak_index

    columns.append(
        {
            "name": name,
            "trajectory_field": trajectory[peak_timestep : peak_timestep + 1].to_physical()[0],
            "rpo_field": align_rpo_to_window(
                rpos[best_host.rpo_index], best_host, peak_timestep, peak_timestep + 1, trajectory
            )[0],
        }
    )


# %%
# Render. Each panel overlays the two fields at its column's peak timestep;
# the shaded band is their pointwise deviation.
figure, axes = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True, sharey=True)
space = np.linspace(0.0, DOMAIN_SIZE, resolution, endpoint=False)

for ax, tag, column in zip(axes, ("(a)", "(b)"), columns, strict=True):
    ax.plot(space, column["trajectory_field"], color=TRAJECTORY_COLOR, label="Trajectory")
    ax.plot(space, column["rpo_field"], color=RPO_COLOR, linewidth=0.9, label="RPO")
    ax.fill_between(
        space,
        column["trajectory_field"],
        column["rpo_field"],
        color=RPO_COLOR,
        alpha=0.3,
        linewidth=0,
    )
    ax.set_title(tag, loc="left")
    ax.set_title(column["name"])
    ax.set_ylabel("$u$")

handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="outside upper right", ncols=2)
axes[-1].set_xlabel("$x$")
plt.show()
