"""
Shadowing event in persistence-diagram space
============================================

The figure holds every :math:`H_0` persistence pair of every timestep in one
shadowing event, on the standard birth-death axes: the chaotic trajectory in
black circles, the shadowed RPO in red triangles. Opacity rises through the
window, so the earliest pairs are faint and the latest are solid. The red
tracks the black point for point along a shared path through diagram space.

The event shown is the one that shadows its RPO for the most orbital periods,
the same event the
:ref:`lab-frame example <sphx_glr_auto_examples_plot_shadowing_event.py>` and
the
:ref:`distance-matrix example <sphx_glr_auto_examples_plot_shadowing_matrices.py>`
select.

The :math:`H_0` diagram also has one essential class that never dies: the
connected component born at the field's minimum. It is drawn on the dashed line
marked infinity above the finite pairs, at its birth. The diagram's
:math:`H_1` class, the loop born when the maximum closes the periodic domain,
enters the Wasserstein distance detection uses but is not drawn here.

No spatial alignment is applied. The sublevel-set persistence
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
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ks_shadowing import KSTrajectory, load_results, load_rpos
from ks_shadowing.pha import KSPersistenceTrajectory

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
# event. The RPO's are gathered by phase along the event.
rpo_persistence = KSPersistenceTrajectory.from_trajectory(rpo_trajectory)
event_persistence = KSPersistenceTrajectory.from_trajectory(
    trajectory[event.start_timestep : event.start_timestep + num_window_timesteps]
)

# %%
# Stack the diagrams into point clouds, carrying each timestep's opacity onto its
# own pairs. The pair count varies per timestep. Only H0 is drawn: the finite
# pairs, plus the essential H0 class born at the field minimum.
window_steps = np.arange(num_window_timesteps)
step_alphas = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * window_steps / max(num_window_timesteps - 1, 1)


def _cloud(
    diagrams: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    counts = np.array([diagram.shape[0] for diagram in diagrams], dtype=np.int64)
    return np.vstack(diagrams), np.repeat(step_alphas, counts)


rpo_window = rpo_persistence[(event.start_phase + window_steps) % period]
series = (
    (event_persistence, TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE, 2),
    (rpo_window, RPO_COLOR, "^", RPO_SIZE, 3),
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


# %%
# Render.
figure, ax = plt.subplots(figsize=(3.4, 2.9))

all_points = np.vstack([np.vstack(persistence.diagrams) for persistence, *_ in series])
birth_low, death_low = all_points.min(axis=0)
birth_high, death_high = all_points.max(axis=0)
all_births = np.concatenate(
    [all_points[:, 0], *(persistence.essential_births[:, 0] for persistence, *_ in series)]
)
pad = AXIS_PAD * max(birth_high - birth_low, death_high - death_low)
# The infinity line sits above every birth, so the essential class stays above
# the diagonal.
infinity = max(death_high, float(all_births.max())) + INFINITY_GAP * (death_high - death_low)

ax.axline((0.0, 0.0), slope=1.0, color="0.7", linewidth=0.6, zorder=1)
ax.axhline(infinity, color="0.7", linewidth=0.6, linestyle="--", zorder=1)
for persistence, color, marker, size, zorder in series:
    points, alphas = _cloud(persistence.diagrams)
    _scatter(ax, points[:, 0], points[:, 1], alphas, color, marker, size, zorder)
    minima = persistence.essential_births[:, 0]
    _scatter(ax, minima, np.full(minima.shape, infinity), step_alphas, color, marker, size, zorder)
ax.set_aspect("equal")
ax.set_xlim(all_births.min() - pad, all_births.max() + pad)
ax.set_ylim(death_low - pad, infinity + pad)

finite_ticks = [tick for tick in ax.get_yticks() if death_low - pad <= tick <= death_high + pad]
ax.set_yticks([*finite_ticks, infinity])
ax.set_yticklabels([*(f"{tick:g}" for tick in finite_ticks), r"$\infty$"])
ax.set_xlabel("Birth")
ax.set_ylabel("Death")

# Axis-break glyphs mark the compressed gap between the finite scale and the
# infinity line.
break_center = 0.5 * (death_high + pad + infinity)
break_delta = 0.06 * (infinity - death_high)
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

ax.legend(
    [_proxy(TRAJECTORY_COLOR, "o", TRAJECTORY_SIZE), _proxy(RPO_COLOR, "^", RPO_SIZE)],
    ["Chaotic trajectory", f"RPO {event.rpo_index + 1}"],
    loc="lower left",
    fontsize=6,
    handlelength=1.5,
    handletextpad=0.4,
    borderpad=0.25,
    labelspacing=0.25,
)

plt.show()
