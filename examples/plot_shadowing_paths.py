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

Only the close passes are drawn in panel (a); each is colored by the component
it belongs to: red where the component's longest path is long enough to be
recorded as an event, blue where it is not. Most components do not admit a
sufficiently long path to be considered an event.

Panel (b) magnifies the boxed stretch of the one component that does yield an
event. Its pale cells are that component's close passes, the same marks panel
(a) draws in red. The black diagonal is the event the detector recorded there,
and the paler diagonals are the other valid paths through the same component.
The other paths were rejected as only the longest is selected as the shadowing
event.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import ConnectionPatch

from ks_shadowing import KSTrajectory, load_results, load_rpos
from ks_shadowing.pha import KSPersistenceTrajectory, connected_components, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
# w = 1, lambda = 1: both PHA embedding stages reduce to the identity, so the
# matrix recomputed below is exactly the one detection thresholded. At any other
# setting the detector averages over orders or along the delay diagonal, and a
# bare Wasserstein matrix would not be comparable with the recorded threshold.
RESULT_PATH = REPO_ROOT / "examples" / "data" / "pha_r2048_d1_o0.h5"
# The window spans exactly one orbital period, so panel (a) is square: a full
# turn of the phase axis against an equal span of trajectory.
CONTEXT_BEFORE = 0.4
CONTEXT_AFTER = 0.6
ZOOM_TIMESTEPS = 30
# Shortest candidate path drawn in panel (b). Every close pass is a one-cell
# path on its own, so drawing those would mark the whole component as
# candidates; two cells is the shortest run that is a diagonal rather than a
# point.
MIN_CANDIDATE_PATH = 2
# Which of this orbit's events to center the window on, by the trajectory
# timestep it starts at.
EVENT_START_TIMESTEP = 45748

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# Tol bright red and blue, splitting close passes by whether their component
# produces an event. Within panel (b) the winner is black and the candidates it
# beat are a muted shade of the component's own red.
EVENT_COMPONENT_COLOR = "#EE6677"
DISCARDED_COMPONENT_COLOR = "#4477AA"
EVENT_COLOR = "black"
CANDIDATE_COLOR = "#995555"
CELL_ALPHA = 0.18

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
# Maximal candidate paths. A valid path holds ``(phase - timestep) % period``
# constant and its timesteps consecutive, so the maximal paths are exactly the
# contiguous timestep runs within each diagonal offset class.
offsets = (close_phases - close_timesteps) % period
sort_order = np.lexsort((close_timesteps, offsets))
breaks = np.ones(close_timesteps.size, dtype=bool)
breaks[1:] = (np.diff(offsets[sort_order]) != 0) | (np.diff(close_timesteps[sort_order]) != 1)
path_ids = np.empty(close_timesteps.size, dtype=np.int64)
path_ids[sort_order] = np.cumsum(breaks) - 1
path_lengths = np.bincount(path_ids)
path_components = component_labels[sort_order][breaks]

# A component is recorded as an event when its longest path clears the run's
# minimum duration, which is what splits panel (a) into red and blue.
longest_path = np.zeros(component_labels.max() + 1, dtype=np.int64)
np.maximum.at(longest_path, path_components, path_lengths)
produces_event = longest_path[component_labels] >= metadata.min_duration

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
event_path_id = int(path_ids[event_pass])

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

# The pale field in panel (b) is this component only, so a neighboring component
# straying into the box is not tinted as though it belonged.
zoom_cells = np.zeros((ZOOM_TIMESTEPS, phase_high - phase_low + 1), dtype=bool)
zoom_cells[close_timesteps[in_zoom] - zoom_start, unwrapped_phases[in_zoom] - phase_low] = True

# %%
# Render. Panel (b) magnifies the boxed stretch of panel (a) beside it,
# connected by two indicator lines.
figure, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), width_ratios=(1.5, 1.0))
origin = window_start - event.start_timestep

# Close passes are drawn as grid cells rather than as markers: at this scale a
# component is a one-cell-wide diagonal, and cells join into a ribbon where
# markers only speckle.
close_grid = np.full(distances.shape, np.nan)
close_grid[close_timesteps, close_phases] = produces_event
axes[0].pcolormesh(
    np.arange(distances.shape[0]) + origin,
    np.arange(period),
    np.ma.masked_invalid(close_grid).T,
    shading="nearest",
    cmap=ListedColormap([DISCARDED_COMPONENT_COLOR, EVENT_COMPONENT_COLOR]),
    vmin=0,
    vmax=1,
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

axes[1].pcolormesh(
    np.arange(zoom_start, zoom_end) + origin,
    np.arange(phase_low, phase_high + 1),
    zoom_cells.T,
    shading="nearest",
    cmap=ListedColormap(["white", to_rgba(EVENT_COMPONENT_COLOR, CELL_ALPHA)]),
    vmin=0,
    vmax=1,
)

for path_id in np.flatnonzero(path_lengths >= MIN_CANDIDATE_PATH):
    if path_components[path_id] != component_labels[event_pass] or path_id == event_path_id:
        continue
    members = np.flatnonzero(path_ids == path_id)
    member_timesteps = close_timesteps[members]
    visible = members[(member_timesteps >= zoom_start) & (member_timesteps < zoom_end)]
    if visible.size == 0:
        continue
    axes[1].plot(
        close_timesteps[visible] + origin,
        unwrapped_phases[visible],
        color=CANDIDATE_COLOR,
        linewidth=1.0,
        solid_capstyle="round",
        zorder=3,
    )

visible_event = (event_timesteps >= zoom_start) & (event_timesteps < zoom_end)
axes[1].plot(
    event_timesteps[visible_event] + origin,
    event.start_phase + steps[visible_event],
    color=EVENT_COLOR,
    linewidth=1.6,
    solid_capstyle="round",
    zorder=4,
)
axes[1].set_xlim(zoom_start + origin - 0.5, zoom_end + origin - 0.5)
axes[1].set_ylim(phase_low - 0.5, phase_high + 0.5)
axes[1].set_box_aspect(1.0)
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
