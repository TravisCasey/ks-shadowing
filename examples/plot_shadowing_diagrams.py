"""
Shadowing event in persistence-diagram space
============================================

Panel (a) holds every persistence pair of every timestep in one shadowing
event, on the standard birth-death axes: the chaotic trajectory in black
circles, the shadowed RPO in red triangles. Opacity rises through the window,
so the earliest pairs are faint and the latest are solid.

Panel (b) holds an equal-length window drawn from the trajectory's longest
stretch with no SSA-detected event, compared against the same RPO on the same
axes. The orbit is placed there at the phase offset minimizing its mean
Wasserstein distance to the window, so the panel shows the closest alignment
available rather than an arbitrary one.

In (a) the red tracks the black point for point along a shared path through
diagram space. In (b) the two clouds pull apart: each traces its own path, and
hardly any trajectory pair has an orbit pair within a marker width of it, even
at that best-aligned phase.

The event shown in (a) is the one that shadows its RPO for the most orbital
periods, the same event the
:ref:`lab-frame example <sphx_glr_auto_examples_plot_shadowing_event.py>` and
the
:ref:`distance-matrix example <sphx_glr_auto_examples_plot_shadowing_matrices.py>`
select.

Each diagram also has two essential classes that never die: the connected
component born at the field's minimum and the loop born when the maximum closes
the periodic domain. They are drawn on the dashed line marked infinity above the
finite pairs, at their births. The essential :math:`H_0` class keeps its series'
marker; the essential :math:`H_1` class is drawn as a plus in the series' color.

No spatial alignment is applied in either panel. The sublevel-set persistence
diagram of a periodic field is invariant to spatial translation, so neither the
RPO's drift nor the event's per-timestep spatial shift moves any point in this
figure. That invariance is what leaves the distance matrix of the companion
example without a shift axis. The
:ref:`event-extraction example <sphx_glr_auto_examples_plot_shadowing_paths.py>`
picks the story up one step later, where Wasserstein distances like these are
thresholded and searched for the paths that become events.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ks_shadowing import KSTrajectory, events_to_union_mask, load_results, load_rpos
from ks_shadowing.pha import KSPersistenceTrajectory, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
RESULT_PATH = REPO_ROOT / "examples" / "data" / "ssa_r2048.h5"
# Opacity at each window's first timestep; it ramps to 1.0 at the last.
ALPHA_FLOOR = 0.25
# Black marks the trajectory and red the RPO. Both are fields rather than
# detection methods, so black here does not carry the gallery's SSA meaning.
TRAJECTORY_COLOR = "black"
RPO_COLOR = "#EE6677"
# Marker areas in points squared. The trajectory circles are drawn beneath the
# smaller RPO triangles, so a coincident pair reads as a red triangle inside a
# black ring rather than one series erasing the other.
TRAJECTORY_SIZE = 34.0
RPO_SIZE = 11.0
H1_MARKER = "P"
AXIS_PAD = 0.05
INFINITY_GAP = 0.15

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load the fixture and pick the event covering the most RPO periods.
metadata, trajectory, events = load_results(RESULT_PATH)
rpo_trajectories = [
    KSTrajectory.from_rpo(rpo, trajectory.resolution, metadata.downsample, metadata.native)
    for rpo in load_rpos(REPO_ROOT / metadata.rpo_file)
]
event = max(
    events,
    key=lambda candidate: (
        (candidate.end_timestep - candidate.start_timestep)
        / rpo_trajectories[candidate.rpo_index].num_timesteps
    ),
)
rpo_trajectory = rpo_trajectories[event.rpo_index]
period = rpo_trajectory.num_timesteps
duration = event.end_timestep - event.start_timestep
num_window_timesteps = min(duration, period)

# %%
# Diagrams of the RPO over its full period, and of the trajectory over the
# event. The RPO's are computed once and gathered by phase in each panel.
rpo_persistence = KSPersistenceTrajectory.from_trajectory(rpo_trajectory)
event_persistence = KSPersistenceTrajectory.from_trajectory(
    trajectory[event.start_timestep : event.start_timestep + num_window_timesteps]
)

# %%
# The control window: an equal-length window centered in the trajectory's
# longest run of timesteps no SSA event covers, so panel (b) compares against
# a trajectory that is shadowing nothing rather than one shadowing some other
# orbit.
covered = events_to_union_mask(events, trajectory.num_timesteps)
boundaries = np.diff(np.concatenate(([0], (~covered).astype(np.int64), [0])))
run_starts = np.flatnonzero(boundaries == 1)
run_ends = np.flatnonzero(boundaries == -1)
longest_run = int(np.argmax(run_ends - run_starts))
control_start = int(
    run_starts[longest_run]
    + (run_ends[longest_run] - run_starts[longest_run] - num_window_timesteps) // 2
)
control_persistence = KSPersistenceTrajectory.from_trajectory(
    trajectory[control_start : control_start + num_window_timesteps]
)

# %%
# Place the RPO in the control window at the phase offset minimizing the mean
# Wasserstein distance along the matrix diagonal.
control_distances = wasserstein_matrix(control_persistence, rpo_persistence)
window_steps = np.arange(num_window_timesteps)
offset_diagonals = control_distances[
    window_steps, (np.arange(period)[:, None] + window_steps) % period
]
control_phase = int(np.argmin(offset_diagonals.mean(axis=1)))

# %%
# Stack each panel's diagrams into point clouds, carrying each timestep's
# opacity onto its own pairs. The pair count varies per timestep.
step_alphas = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * np.arange(num_window_timesteps) / max(
    num_window_timesteps - 1, 1
)


def _cloud(
    diagrams: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    counts = np.array([diagram.shape[0] for diagram in diagrams], dtype=np.int64)
    return np.vstack(diagrams), np.repeat(step_alphas, counts)


def _essential_births(
    births: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Two essential classes per timestep, both at infinite death; only the
    # births are positioned, on the shared infinity line drawn at render time.
    # Column 0 is the H0 birth (minimum), column 1 the H1 birth (maximum).
    return births[:, 0], births[:, 1], step_alphas


panels = []
for tag, name, window_persistence, phase in (
    ("(a)", "During the event", event_persistence, event.start_phase),
    ("(b)", "Event-free window", control_persistence, control_phase),
):
    phases = (phase + window_steps) % period
    rpo_window = rpo_persistence[phases]
    panels.append(
        (
            tag,
            name,
            (
                (*_cloud(window_persistence.diagrams), TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE, 2),
                (*_cloud(rpo_window.diagrams), RPO_COLOR, "^", RPO_SIZE, 3),
            ),
            (
                (
                    *_essential_births(window_persistence.essential_births),
                    TRAJECTORY_COLOR,
                    "o",
                    TRAJECTORY_SIZE,
                    2,
                ),
                (*_essential_births(rpo_window.essential_births), RPO_COLOR, "^", RPO_SIZE, 3),
            ),
        )
    )


# %%
# Legend proxies. Per-point opacity is baked into the face colors, so an
# automatic legend would draw its entries at the first point's opacity.
def _proxy(color: str, marker: str, size: float) -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        color=color,
        markersize=float(np.sqrt(size)),
    )


# %%
# Render.
figure, axes = plt.subplots(2, 1, figsize=(3.4, 4.2), sharex=True, sharey=True)

all_points = np.vstack([points for _, _, series, _ in panels for points, *_ in series])
birth_low, death_low = all_points.min(axis=0)
birth_high, death_high = all_points.max(axis=0)
all_births = np.concatenate(
    [
        all_points[:, 0],
        *(minima for _, _, _, series in panels for minima, *_ in series),
        *(maxima for _, _, _, series in panels for _, maxima, *_ in series),
    ]
)
pad = AXIS_PAD * max(birth_high - birth_low, death_high - death_low)
# The infinity line sits above every birth, so the essential classes stay
# above the diagonal.
infinity = max(death_high, float(all_births.max())) + INFINITY_GAP * (death_high - death_low)


def _scatter(  # noqa: PLR0913, PLR0917
    ax: plt.Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alphas: NDArray[np.float64],
    color: str,
    marker: str,
    size: float,
    zorder: int,
) -> None:
    colors = np.tile(to_rgba(color), (alphas.size, 1))
    colors[:, 3] = alphas
    ax.scatter(x, y, s=size, marker=marker, c=colors, linewidths=0.0, zorder=zorder)


for ax, (tag, name, finite_series, essential_series) in zip(axes, panels, strict=True):
    ax.axline((0.0, 0.0), slope=1.0, color="0.7", linewidth=0.6, zorder=1)
    ax.axhline(infinity, color="0.7", linewidth=0.6, linestyle="--", zorder=1)
    for points, alphas, color, marker, size, zorder in finite_series:
        _scatter(ax, points[:, 0], points[:, 1], alphas, color, marker, size, zorder)
    for minima, maxima, alphas, color, marker, size, zorder in essential_series:
        line = np.full(minima.shape, infinity)
        _scatter(ax, minima, line, alphas, color, marker, size, zorder)
        _scatter(ax, maxima, line, alphas, color, H1_MARKER, size, zorder)
    ax.set_aspect("equal")
    ax.set_title(tag, loc="left")
    ax.set_title(name)
    ax.set_ylabel("Death")

axes[0].set_xlim(all_births.min() - pad, all_births.max() + pad)
axes[0].set_ylim(death_low - pad, infinity + pad)

finite_ticks = [
    tick for tick in axes[0].get_yticks() if death_low - pad <= tick <= death_high + pad
]
axes[0].set_yticks([*finite_ticks, infinity])
axes[0].set_yticklabels([*(f"{tick:g}" for tick in finite_ticks), r"$\infty$"])
axes[-1].set_xlabel("Birth")

# Axis-break glyphs mark the compressed gap between the finite scale and the
# infinity line.
break_center = 0.5 * (death_high + pad + infinity)
break_delta = 0.06 * (infinity - death_high)
for ax in axes:
    ax.plot(
        [0.0, 0.0],
        [break_center - break_delta, break_center + break_delta],
        transform=ax.get_yaxis_transform(),
        linestyle="none",
        marker=[(-1.0, -0.5), (1.0, 0.5)],
        markersize=5.0,
        markeredgewidth=0.8,
        markerfacecolor="none",
        markeredgecolor="black",
        clip_on=False,
        zorder=5,
    )

axes[0].legend(
    [
        _proxy(TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE),
        _proxy(RPO_COLOR, "^", RPO_SIZE),
        (
            _proxy(TRAJECTORY_COLOR, H1_MARKER, TRAJECTORY_SIZE),
            _proxy(RPO_COLOR, H1_MARKER, RPO_SIZE),
        ),
    ],
    [
        "Chaotic trajectory",
        f"RPO {event.rpo_index + 1}",
        "$H_1$ class",
    ],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
    loc="lower right",
    fontsize=6,
    handlelength=1.5,
    handletextpad=0.4,
    borderpad=0.25,
    labelspacing=0.25,
)

plt.show()
