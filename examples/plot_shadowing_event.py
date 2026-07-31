"""
Shadowing event: trajectory vs. RPO
====================================

A two-panel comparison of one shadowing event: the chaotic trajectory window on
top, the RPO field spatially aligned to it on the bottom. Black dashed lines
mark the event boundaries. When the trajectory shadows the RPO, the two panels
show a nearly identical field evolving in time.

The event shown is the one that shadows its RPO for the most orbital periods.
The
:ref:`distance-matrix example <sphx_glr_auto_examples_plot_shadowing_matrices.py>`
selects the same event.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing import (
    DOMAIN_SIZE,
    KSTrajectory,
    align_rpo_to_window,
    load_results,
    load_rpos,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
RESULT_PATH = REPO_ROOT / "examples" / "data" / "ssa_r2048.h5"
CONTEXT_MULTIPLE = 1.7

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load the fixture and pick the event covering the most RPO periods. Each RPO's
# period in trajectory rows is the length of its per-RPO trajectory, built at
# the sampling the detection run used.
metadata, trajectory, events = load_results(RESULT_PATH)
rpos = load_rpos(REPO_ROOT / metadata.rpo_file)
rpo_periods = [
    KSTrajectory.from_rpo(
        candidate_rpo, trajectory.resolution, metadata.downsample, metadata.native
    ).num_timesteps
    for candidate_rpo in rpos
]
event = max(
    events,
    key=lambda candidate: (
        (candidate.end_timestep - candidate.start_timestep) / rpo_periods[candidate.rpo_index]
    ),
)
rpo = rpos[event.rpo_index]

duration = event.end_timestep - event.start_timestep
context = int(duration * CONTEXT_MULTIPLE)
plot_start = max(0, event.start_timestep - context)
plot_end = min(len(trajectory), event.end_timestep + context)

# %%
# Build the trajectory and aligned-RPO panels.
trajectory_slice = trajectory[plot_start:plot_end]
trajectory_physical = trajectory_slice.to_physical()
aligned_rpo = align_rpo_to_window(rpo, event, plot_start, plot_end, trajectory)

dt = trajectory.dt
times = np.arange(plot_start, plot_end) * dt
relative_times = (np.arange(plot_start, plot_end) - event.start_timestep) * dt
space = np.linspace(0, DOMAIN_SIZE, trajectory.resolution, endpoint=False)
# Symmetric limits keep the diverging colormap's white at u = 0.
vmax = float(max(np.abs(trajectory_physical).max(), np.abs(aligned_rpo).max()))

# %%
# Render. The trajectory panel keeps absolute time; the RPO panel uses
# event-relative time. Both spans are equal, so the panels stay aligned.
figure, axes = plt.subplots(2, 1, figsize=(7.0, 3.7), sharey=True)

panels = (
    ("(a)", "Chaotic trajectory", trajectory_physical, times, event.start_timestep * dt),
    ("(b)", f"RPO {event.rpo_index} (aligned)", aligned_rpo, relative_times, 0.0),
)
for ax, (tag, name, field, x_values, event_start) in zip(axes, panels, strict=True):
    mesh = ax.pcolormesh(
        x_values,
        space,
        field.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title(tag, loc="left")
    ax.set_title(name)
    ax.set_ylabel("$x$")
    ax.axvline(event_start, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(event_start + duration * dt, color="black", linestyle="--", linewidth=1.0)
axes[0].set_xlabel("Time")
axes[1].set_xlabel("Time relative to event start")
figure.colorbar(mesh, ax=axes, label="$u(x, t)$", pad=0.02)

plt.show()
