"""
Event extraction: components and candidate paths
================================================

How events are detected from a distance matrix, in two panels. Every entry below
the detection threshold is a close pass; close passes are grouped into
8-connected components; and the longest valid path through each component
becomes at most one shadowing event.

A path is valid when the trajectory timestep and the RPO phase each advance by
exactly one per step, modulo the RPO period. Every candidate path is therefore a
unit-slope diagonal, and the candidates within one component run parallel to
each other.

Panel (a) draws only the close passes. Most components do not admit a path
long enough to be considered an event.

Panel (b) magnifies the boxed stretch of one component that does yield an
event, drawn as a graph on the grid of (timestep, phase) cells. Each node is a
close pass, and each edge joins two close passes one timestep and one phase
apart, which is exactly the step a valid path may take. The component holds
many parallel candidate paths; the detector records only the longest as the
shadowing event.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch

from ks_shadowing import KSTrajectory, load_results, load_rpos
from ks_shadowing.pha import KSPersistenceTrajectory, connected_components, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
# m = 1: both PHA embedding methods reduce to the identity, so the
# matrix recomputed below is exactly the one detection thresholded. At any other
# setting the detector averages over orders or along the delay diagonal, and a
# bare Wasserstein matrix would not be comparable with the recorded threshold.
RESULT_PATH = REPO_ROOT / "examples" / "data" / "pha_r2048_d1_o0.h5"
# The window spans exactly one orbital period, so panel (a) is square: a full
# turn of the phase axis against an equal span of trajectory.
CONTEXT_BEFORE = 0.4
CONTEXT_AFTER = 0.6
ZOOM_TIMESTEPS = 12
# Which of this orbit's events to center the window on, by the trajectory
# timestep it starts at.
EVENT_START_TIMESTEP = 45748

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# Tol bright red for every close pass, in both panels.
COMPONENT_COLOR = "#EE6677"
# Marker area in points squared and edge width in points for panel (b)'s graph.
NODE_SIZE = 8.0
EDGE_WIDTH = 0.9
LATTICE_COLOR = "0.9"

# %%
# Load the fixture and take one event recorded against the longest orbit.
metadata, trajectory, events = load_results(RESULT_PATH)
rpos = load_rpos(REPO_ROOT / metadata.rpo_file)
rpo_trajectories = [
    KSTrajectory.from_rpo(rpo, trajectory.resolution, metadata.downsample, metadata.native)
    for rpo in rpos
]
rpo_index = max(range(len(rpos)), key=lambda index: rpo_trajectories[index].num_timesteps)
rpo_trajectory = rpo_trajectories[rpo_index]
period = rpo_trajectory.num_timesteps
event = next(
    (
        candidate
        for candidate in events
        if candidate.rpo_index == rpo_index and candidate.start_timestep == EVENT_START_TIMESTEP
    ),
    None,
)
if event is None:
    raise ValueError(
        f"no event against RPO {rpo_index} starts at timestep {EVENT_START_TIMESTEP}; "
        "update EVENT_START_TIMESTEP if the fixtures were regenerated"
    )

# The event sits inside a one-period window, off center by the context split.
duration = event.end_timestep - event.start_timestep
leftover = period - duration
window_start = event.start_timestep - round(
    leftover * CONTEXT_BEFORE / (CONTEXT_BEFORE + CONTEXT_AFTER)
)
window_end = window_start + period

# %%
# The Wasserstein matrix over the window. Persistence diagrams are computed on
# the lab-frame fields exactly as detection does; the distance is
# translation-invariant, so no co-moving transform is needed.
window_diagrams = KSPersistenceTrajectory.from_trajectory(trajectory[window_start:window_end])
rpo_diagrams = KSPersistenceTrajectory.from_trajectory(rpo_trajectory)
distances = wasserstein_matrix(window_diagrams, rpo_diagrams)

# %%
# Close passes, then the same 8-connected grouping detection uses, wraparound in
# the phase dimension included.
close_timesteps, close_phases = np.nonzero(distances < metadata.threshold)
component_labels = connected_components(close_timesteps, close_phases, period)

# ``np.nonzero`` returns close passes in ``(timestep, phase)`` order, so this
# key is sorted and a pass can be looked up by its coordinates.
pass_keys = close_timesteps * period + close_phases

# %%
# The detected event as a path on the same grid. The phase advances one step per
# timestep from ``start_phase``, so its cells follow from its endpoints alone.
# Everything stays in window rows; the event-relative axis is applied once, at
# plot time.
steps = np.arange(duration)
event_timesteps = event.start_timestep + steps - window_start
event_phases = (event.start_phase + steps) % period
event_key = int(event_timesteps[0] * period + event_phases[0])
event_pass = int(np.searchsorted(pass_keys, event_key))
if event_pass == pass_keys.size or pass_keys[event_pass] != event_key:
    raise ValueError(
        "the event's first cell is not a close pass of the recomputed matrix; "
        "the result fixture and this build disagree"
    )

in_component = component_labels == component_labels[event_pass]
on_event_path = np.isin(pass_keys, event_timesteps * period + event_phases)

# %%
# Panel (b) plots phase unwrapped along the event path: the signed offset to the
# path, added back to the path's own running phase.
path_phases = event.start_phase + close_timesteps - event_timesteps[0]
phase_offsets = (close_phases - path_phases + period // 2) % period - period // 2
unwrapped_phases = path_phases + phase_offsets

# %%
# Place the zoom on the stretch of the event where the most component cells sit
# off the event path, which is where the competition is.
tallies = np.bincount(close_timesteps[in_component & ~on_event_path], minlength=distances.shape[0])
cumulative = np.concatenate([[0], np.cumsum(tallies)])
starts = np.arange(event_timesteps[0], event_timesteps[-1] - ZOOM_TIMESTEPS + 2)
zoom_start = int(starts[np.argmax(cumulative[starts + ZOOM_TIMESTEPS] - cumulative[starts])])
zoom_end = zoom_start + ZOOM_TIMESTEPS

in_zoom = in_component & (close_timesteps >= zoom_start) & (close_timesteps < zoom_end)
phase_low = int(unwrapped_phases[in_zoom].min()) - 1
phase_high = int(unwrapped_phases[in_zoom].max()) + 1

# %%
# Panel (b)'s graph is this component only, so a neighboring component straying
# into the box is not drawn as though it belonged. An edge joins two nodes one
# timestep and one phase apart. Edges are gathered one timestep past each side
# of the zoom, so a path that continues beyond the box runs out to its border.
in_margin = in_component & (close_timesteps >= zoom_start - 1) & (close_timesteps <= zoom_end)
margin_nodes = set(
    zip(close_timesteps[in_margin].tolist(), unwrapped_phases[in_margin].tolist(), strict=True)
)
edge_starts = {node for node in margin_nodes if (node[0] + 1, node[1] + 1) in margin_nodes}

# Each maximal diagonal run of edges is one polyline, so its joints stay clean.
diagonal_runs = []
for timestep, phase in sorted(edge_starts):
    if (timestep - 1, phase - 1) in edge_starts:
        continue
    length = 1
    while (timestep + length, phase + length) in edge_starts:
        length += 1
    diagonal_runs.append((timestep, phase, length))

# %%
# Render. Panel (b) magnifies the boxed stretch of panel (a) beside it,
# connected by two indicator lines.
figure, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), width_ratios=(1.5, 1.0))
origin = window_start - event.start_timestep

# Close passes are drawn as grid cells rather than as markers: at this scale a
# component is a one-cell-wide diagonal, and cells join into a ribbon where
# markers only speckle.
close_grid = np.full(distances.shape, np.nan)
close_grid[close_timesteps, close_phases] = 1.0
axes[0].pcolormesh(
    np.arange(distances.shape[0]) + origin,
    np.arange(period),
    np.ma.masked_invalid(close_grid).T,
    shading="nearest",
    cmap=ListedColormap([COMPONENT_COLOR]),
    rasterized=True,
)
axes[0].set_xlim(origin - 0.5, origin + period - 0.5)
axes[0].set_ylim(-0.5, period - 0.5)
axes[0].set_box_aspect(1.0)
axes[0].add_patch(
    plt.Rectangle(
        (zoom_start + origin - 0.5, phase_low - 0.5),
        ZOOM_TIMESTEPS,
        phase_high - phase_low + 1,
        fill=False,
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )
)

# A faint lattice through the node positions shows that each node is one
# (timestep, phase) cell.
for timestep in range(zoom_start, zoom_end):
    axes[1].axvline(timestep + origin, color=LATTICE_COLOR, linewidth=0.4, zorder=0)
for phase in range(phase_low, phase_high + 1):
    axes[1].axhline(phase, color=LATTICE_COLOR, linewidth=0.4, zorder=0)
for timestep, phase, length in diagonal_runs:
    axes[1].plot(
        [timestep + origin, timestep + length + origin],
        [phase, phase + length],
        color=COMPONENT_COLOR,
        linewidth=EDGE_WIDTH,
        solid_capstyle="round",
        zorder=2,
    )
axes[1].scatter(
    close_timesteps[in_zoom] + origin,
    unwrapped_phases[in_zoom],
    s=NODE_SIZE,
    color=COMPONENT_COLOR,
    linewidths=0.0,
    zorder=3,
)
axes[1].set_xlim(zoom_start + origin - 0.5, zoom_end + origin - 0.5)
axes[1].set_ylim(phase_low - 0.5, phase_high + 0.5)
# Equal aspect keeps every edge at its true unit slope.
axes[1].set_aspect("equal")
# The magnified panel is an inset: the box and indicator lines locate it, so
# it carries no tick numbers of its own.
axes[1].set_xticks([])
axes[1].set_yticks([])

# Indicator lines from the boxed stretch to the magnified panel.
for box_phase, axes_fraction in ((phase_low - 0.5, 0.0), (phase_high + 0.5, 1.0)):
    figure.add_artist(
        ConnectionPatch(
            xyA=(zoom_end + origin - 0.5, box_phase),
            coordsA=axes[0].transData,
            xyB=(0.0, axes_fraction),
            coordsB=axes[1].transAxes,
            color="0.5",
            linewidth=0.6,
        )
    )

# Tags only: the caption carries the panel descriptions at this size, and one
# figure-level x-label serves both panels.
for ax, tag in ((axes[0], "(a)"), (axes[1], "(b)")):
    ax.set_title(tag, loc="left")
axes[0].set_ylabel("RPO phase")
figure.supxlabel("Timestep relative to event start", fontsize=8)

plt.show()
