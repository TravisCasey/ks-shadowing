"""
Anatomy of SSA/PHA disagreement
================================

The :ref:`agreement example <sphx_glr_auto_examples_plot_coverage_vs_embedding.py>`
scores how closely the two detection methods agree; this figure investigates
where they do not. Each column examines one gap: a stretch of timesteps inside
one method's event that the other method, matched against the same RPO, leaves
uncovered. The left column depcits an SSA event with a PHA gap, the right column
a PHA event with an SSA gap.

The top row follows each method's distance along the event's phase track,
normalized by that method's own detection threshold; the gray band marks the
gap. The middle row overlays the trajectory and aligned-RPO fields at the
timestep where the missing method's normalized distance peaks (the dotted line
in the top row).

The bottom row dissects each column's failing metric. Panel (e) overlays the two
fields' persistence diagrams at every timestep of the left column's gap, opacity
rising from the gap's first timestep to its last as in the
:ref:`persistence-diagram example <sphx_glr_auto_examples_plot_shadowing_diagrams.py>`.
Panel (f) accumulates the squared deviation between the two fields across the
domain at each column's peak timestep, in units of the squared SSA threshold,
so each curve ends at the square of its column's SSA trace value.

The columns fail in opposite ways, and the split is what sublevel-set
persistence reads: extremum values, never extremum positions. In the left
column the extrema sit in the right places with drifting values. The
persistence pairs pull away from their RPO counterparts through the gap (e), so
the Wasserstein distance exceeds the PHA threshold, yet the deviation has narrow
support: its accumulated square rises in steps at the mismatched extrema and
ends below the SSA threshold. In the right column the extrema hold the right
values in slightly shifted places.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
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
# Black marks the trajectory and red the RPO in the field and diagram rows.
# Marker areas and the opacity ramp follow the persistence-diagram example:
# trajectory circles draw beneath the smaller RPO triangles, so a coincident
# pair reads as a triangle inside a ring, and opacity rises from ALPHA_FLOOR at
# the gap's first timestep to 1.0 at its last.
TRAJECTORY_COLOR = "black"
RPO_COLOR = "#EE6677"
TRAJECTORY_SIZE = 34.0
RPO_SIZE = 11.0
ALPHA_FLOOR = 0.25
SSA_COLOR = "black"
PHA_COLOR = "0.55"
AXIS_PAD = 0.05

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
                KSPersistenceTrajectory([window_persistence.diagrams[index]], trajectory.dt),
                KSPersistenceTrajectory([orbit_persistence.diagrams[phase]], trajectory.dt),
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
    ("SSA event, PHA gap", ssa_events, pha_events, "pha"),
    ("PHA event, SSA gap", pha_events, ssa_events, "ssa"),
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
    ssa_normalized, pha_normalized, window_persistence, phases = _track_distances(
        best_host, window_start, window_end
    )
    missing_normalized = pha_normalized if missing == "pha" else ssa_normalized
    gap_slice = slice(gap_start - window_start, gap_end - window_start)
    peak_index = gap_slice.start + int(np.argmax(missing_normalized[gap_slice]))
    peak_timestep = window_start + peak_index

    columns.append(
        {
            "name": name,
            "rpo_index": best_host.rpo_index,
            "gap": (gap_start, gap_end),
            "window_start": window_start,
            "ssa_normalized": ssa_normalized,
            "pha_normalized": pha_normalized,
            "peak_timestep": peak_timestep,
            "trajectory_field": trajectory[peak_timestep : peak_timestep + 1].to_physical()[0],
            "rpo_field": align_rpo_to_window(
                rpos[best_host.rpo_index], best_host, peak_timestep, peak_timestep + 1, trajectory
            )[0],
            "trajectory_diagrams": window_persistence.diagrams[gap_slice],
            "rpo_diagrams": [
                rpo_persistence[best_host.rpo_index].diagrams[phase] for phase in phases[gap_slice]
            ],
        }
    )


# %%
# Legend proxies for the diagram row. Per-point opacity is baked into the face
# colors, so an automatic legend would draw its entries at the first point's
# opacity.
def _proxy(color: str, marker: str, size: float, alpha: float = 1.0) -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        color=color,
        alpha=alpha,
        markersize=float(np.sqrt(size)),
    )


# %%
# Render. The top two rows share their encoding across columns; the bottom row
# is asymmetric by design, giving each column's failing metric its own
# decomposition. The dotted line in the trace panel marks the peak timestep
# the field panel shows.
figure, axes = plt.subplots(3, 2, figsize=(7.0, 5.8), height_ratios=(1.0, 1.0, 1.35))
space = np.linspace(0.0, DOMAIN_SIZE, resolution, endpoint=False)

for column_index, column in enumerate(columns):
    gap_start, gap_end = column["gap"]
    times = (
        np.arange(column["window_start"], column["window_start"] + len(column["ssa_normalized"]))
        * trajectory.dt
    )
    peak_time = column["peak_timestep"] * trajectory.dt

    ax = axes[0, column_index]
    # Samples are points, so the band edges sit half a sample outside the
    # gap's timesteps: every uncovered sample falls strictly inside the band
    # and the covered samples flanking the gap strictly outside.
    ax.axvspan(
        (gap_start - 0.5) * trajectory.dt, (gap_end - 0.5) * trajectory.dt, color="0.92", zorder=0
    )
    ax.axhline(1.0, color="0.7", linewidth=0.6, linestyle="--")
    ax.axvline(peak_time, color="0.3", linewidth=0.6, linestyle=":")
    ax.plot(times, column["ssa_normalized"], color=SSA_COLOR, label="SSA")
    ax.plot(times, column["pha_normalized"], color=PHA_COLOR, label="PHA")
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_title(("(a)", "(b)")[column_index], loc="left")
    ax.set_title(column["name"])
    ax.set_xlabel("Time")

    ax = axes[1, column_index]
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
    ax.set_title(("(c)", "(d)")[column_index], loc="left")
    ax.set_title(f"RPO {column['rpo_index']}, timestep {column['peak_timestep']}")
    ax.set_xlabel("$x$")

axes[0, 0].set_ylabel("Distance / threshold")
axes[0, 0].legend()
axes[1, 0].set_ylabel("$u$")
axes[1, 0].legend()

# %%
# Panel (e): persistence diagrams at every timestep of the left column's gap.
ax = axes[2, 0]
left_column = columns[0]
all_points = np.vstack(left_column["trajectory_diagrams"] + left_column["rpo_diagrams"])
low = all_points.min()
high = all_points.max()
pad = AXIS_PAD * (high - low)
diagonal = (low - pad, high + pad)
ax.plot(diagonal, diagonal, color="0.7", linewidth=0.6, zorder=1)
gap_length = len(left_column["trajectory_diagrams"])
step_alphas = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * np.arange(gap_length) / max(gap_length - 1, 1)
for diagrams, color, marker, size, zorder in (
    (left_column["trajectory_diagrams"], TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE, 2),
    (left_column["rpo_diagrams"], RPO_COLOR, "^", RPO_SIZE, 3),
):
    points = np.vstack(diagrams)
    counts = np.array([diagram.shape[0] for diagram in diagrams], dtype=np.int64)
    # The opacity varies per point, so it goes into the face colors rather
    # than through ``alpha``, which takes one value for the collection.
    face_colors = np.tile(to_rgba(color), (points.shape[0], 1))
    face_colors[:, 3] = np.repeat(step_alphas, counts)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=size,
        marker=marker,
        c=face_colors,
        linewidths=0.0,
        zorder=zorder,
    )
ax.set_xlim(diagonal)
ax.set_ylim(diagonal)

ax.set_aspect("equal")
ax.set_anchor("W")
ax.set_title("(e)", loc="left")
ax.set_xlabel("Birth")
ax.set_ylabel("Death")
# The legend sits in the empty space to the panel's right.
ax.legend(
    [
        _proxy(TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE),
        _proxy(RPO_COLOR, "^", RPO_SIZE),
        tuple(
            _proxy(TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE, alpha)
            for alpha in (ALPHA_FLOOR, 0.5 * (1.0 + ALPHA_FLOOR), 1.0)
        ),
    ],
    ["Trajectory", "RPO", "Gap start to end"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)

# %%
# Panel (f): running sum of the squared field deviation across the domain at
# each column's peak timestep.
ax = axes[2, 1]
ax.axhline(1.0, color="0.7", linewidth=0.6, linestyle="--")
for column, color, label in zip(
    columns, ("0.55", "black"), ("PHA gap (c)", "SSA gap (d)"), strict=True
):
    cumulative = (
        np.cumsum((column["trajectory_field"] - column["rpo_field"]) ** 2)
        / ssa_metadata.threshold**2
    )
    ax.plot(space, cumulative, color=color, label=label)
ax.set_title("(f)", loc="left")
ax.set_xlabel("$x$")
ax.set_ylabel(r"Cumulative $(\Delta u)^2$ / SSA threshold$^2$")
ax.legend()
plt.show()
