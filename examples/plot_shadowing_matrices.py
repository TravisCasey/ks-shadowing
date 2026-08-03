"""
Distance matrices behind one shadowing event
============================================

The companion
:ref:`lab-frame example <sphx_glr_auto_examples_plot_shadowing_event.py>` shows
this same event as two space-time panels. Both examples select the event that
shadows its RPO for the most orbital periods, so both show the same event; here
it appears in the RPO's co-moving frame, alongside the two distance matrices the
detectors search.

Panels (a) and (b) are the chaotic trajectory and the shadowed RPO, both in the
co-moving frame and aligned by the event's per-timestep spatial shift. The
co-moving frame removes the orbit's drift, so the RPO is exactly periodic in it
and panel (b) visibly repeats. Inside the event the two fields are nearly
identical; outside it they diverge.

Panels (c) and (d) hold the SSA and PHA distance matrices over the same window,
indexed by trajectory timestep and RPO phase. Both detectors index the RPO by
absolute phase, so the two panels share axes exactly. A shadowing event appears
in both as a diagonal streak, because the RPO phase advances one step per
trajectory timestep; when an event runs longer than one orbital period its
streak wraps the phase axis, which is the recurrence made visible. The dashed
lines carry across all four panels, so the streak between them is the event that
panels (a) and (b) show.

The SSA matrix is the :math:`L_2` distance minimized over spatial shift; the PHA
matrix is the Wasserstein distance between persistence diagrams, which carries
no shift axis because persistence quotients out the spatial symmetry. Reducing
the shift axis is what makes the two commensurable. Both use a logarithmic color
scale. The fainter streaks elsewhere in the window are near recurrences, though
not sufficiently close or sufficiently long to be their own events.

The
:ref:`persistence-diagram example <sphx_glr_auto_examples_plot_shadowing_diagrams.py>`
shows this same event once more, as the persistence pairs whose Wasserstein
distances panel (d) holds. The
:ref:`event-extraction example <sphx_glr_auto_examples_plot_shadowing_paths.py>`
takes the step after panel (d), thresholding a matrix of this kind and searching
it for the paths that become events.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ks_shadowing import (
    DOMAIN_SIZE,
    KSTrajectory,
    assert_same_trajectory,
    load_results,
    load_rpos,
    shift_distances_sq,
)
from ks_shadowing.pha import KSPersistenceTrajectory, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
# Delay 1, max order 0: both PHA embedding stages reduce to the identity, so
# panel (d) is a snapshot-to-snapshot distance, the same kind of quantity panel
# (c) holds.
PHA_PATH = DATA_DIR / "pha_r2048_d1_o0.h5"
CONTEXT_MULTIPLE = 1.7

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load both detections and select the event that shadows its RPO for the most
# orbital periods.
ssa_metadata, trajectory, ssa_events = load_results(SSA_PATH)
pha_metadata, pha_trajectory, _ = load_results(PHA_PATH)
assert_same_trajectory(trajectory, pha_trajectory)

rpos = load_rpos(REPO_ROOT / ssa_metadata.rpo_file)
rpo_trajectories = [
    KSTrajectory.from_rpo(rpo, trajectory.resolution, ssa_metadata.downsample, ssa_metadata.native)
    for rpo in rpos
]
event = max(
    ssa_events,
    key=lambda candidate: (
        (candidate.end_timestep - candidate.start_timestep)
        / rpo_trajectories[candidate.rpo_index].num_timesteps
    ),
)
rpo = rpos[event.rpo_index]
rpo_trajectory = rpo_trajectories[event.rpo_index]
period = rpo_trajectory.num_timesteps

duration = event.end_timestep - event.start_timestep
context = int(duration * CONTEXT_MULTIPLE)
window_start = max(0, event.start_timestep - context)
window_end = min(len(trajectory), event.end_timestep + context)
num_window_timesteps = window_end - window_start

# %%
# Panels (a) and (b): both fields in the RPO's co-moving frame. ``start_time``
# is required because the co-moving phase factor is absolute and this slice does
# not begin at row 0.
dt = trajectory.dt
trajectory_comoving = trajectory[window_start:window_end].to_comoving(
    rpo.drift_rate, start_time=window_start * dt
)
rpo_comoving = rpo_trajectory.to_comoving(rpo.drift_rate)

window_timesteps = np.arange(window_start, window_end)
event_phases = (event.start_phase + window_timesteps - event.start_timestep) % period
event_shifts = event.shifts[
    np.clip(window_timesteps - event.start_timestep, 0, len(event.shifts) - 1)
]

trajectory_field = trajectory_comoving.to_physical()
rpo_physical = rpo_comoving.to_physical()
# The shift varies along the event, so each row is rolled separately rather than
# the whole array at once.
rpo_field = np.empty_like(trajectory_field)
for row in range(num_window_timesteps):
    rpo_field[row] = np.roll(rpo_physical[event_phases[row]], -int(event_shifts[row]))

# %%
# Panel (c): the SSA distance matrix, minimized over spatial shift. One
# ``shift_distances_sq`` call per RPO phase evaluates every shift at once; the
# clamp absorbs the numerical noise that can push a near-zero squared distance
# slightly negative.
l2_distances = np.empty((num_window_timesteps, period), dtype=np.float64)
for phase_index in range(period):
    rpo_slice_modes = np.broadcast_to(
        rpo_comoving.modes[phase_index], trajectory_comoving.modes.shape
    )
    distances_sq = shift_distances_sq(
        trajectory_comoving.modes, rpo_slice_modes, trajectory.resolution
    )
    l2_distances[:, phase_index] = np.sqrt(np.maximum(distances_sq.min(axis=1), 0.0))

# %%
# Panel (d): the PHA Wasserstein matrix over the same window. Persistence
# diagrams are computed on the lab frame fields, exactly as detection does; the
# distance is translation-invariant, so the co-moving transform is unnecessary
# here.
window_diagrams = KSPersistenceTrajectory.from_trajectory(trajectory[window_start:window_end])
rpo_diagrams = KSPersistenceTrajectory.from_trajectory(rpo_trajectory)
wasserstein_distances = wasserstein_matrix(window_diagrams, rpo_diagrams)

# %%
# Render. All four panels share the trajectory-time axis, so one vertical
# position lands on the same timestep everywhere.
figure, axes = plt.subplots(4, 1, figsize=(7.0, 6.3), sharex=True)
axes[1].sharey(axes[0])

times = window_timesteps * dt
space = np.linspace(0, DOMAIN_SIZE, trajectory.resolution, endpoint=False)
phases = np.arange(period)

# Symmetric limits keep the diverging colormap's white at u = 0.
field_limit = float(max(np.abs(trajectory_field).max(), np.abs(rpo_field).max()))
field_panels = (
    ("(a)", "Chaotic trajectory", trajectory_field),
    ("(b)", f"RPO {event.rpo_index} (aligned)", rpo_field),
)
for ax, (tag, name, field) in zip(axes[:2], field_panels, strict=True):
    field_mesh = ax.pcolormesh(
        times,
        space,
        field.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-field_limit,
        vmax=field_limit,
        rasterized=True,
    )
    ax.set_title(tag, loc="left")
    ax.set_title(name)
    ax.set_ylabel("$x$")
figure.colorbar(field_mesh, ax=axes[:2], label="$u(x, t)$", pad=0.02)

distance_panels = (
    ("(c)", "SSA Distance matrix", "$L_2$ distance", l2_distances),
    (
        "(d)",
        "PHA Distance matrix",
        "$W_2$ distance",
        wasserstein_distances,
    ),
)
for ax, (tag, name, label, matrix) in zip(axes[2:], distance_panels, strict=True):
    distance_mesh = ax.pcolormesh(
        times,
        phases,
        matrix.T,
        shading="auto",
        cmap="viridis_r",
        norm=LogNorm(vmin=matrix.min(), vmax=matrix.max()),
        rasterized=True,
    )
    ax.set_title(tag, loc="left")
    ax.set_title(name)
    ax.set_ylabel("RPO phase")
    figure.colorbar(distance_mesh, ax=ax, label=label, pad=0.02)

for ax in axes:
    ax.axvline(event.start_timestep * dt, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(event.end_timestep * dt, color="black", linestyle="--", linewidth=1.0)
axes[-1].set_xlabel("Time")

plt.show()
