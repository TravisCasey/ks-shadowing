"""
Matched events: SSA vs. PHA
============================

Each detection method produces a list of shadowing events keyed by RPO index,
start timestep, and end timestep. Pairing SSA and PHA events with the same RPO
and overlapping windows yields a scatter of (SSA length, PHA length) colored by
the intersection-over-union fraction of their windows. Points near the dashed
diagonal mark events the two methods agree on in duration; high IoU (yellow)
means the windows themselves overlap heavily, low IoU (purple) means they
barely intersect.

One panel per embedding axis: the delay-axis setting (delay 8, max order 0)
and the derivative-axis setting (delay 1, max order 2), each matched against
the same SSA run. The two embedding axes are shown independently: delays
greater than 1 are used only at max order 0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing import (
    assert_same_trajectory,
    load_results,
    match_events,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
SSA_PATH = REPO_ROOT / "examples" / "data" / "ssa_r2048.h5"
PHA_PATHS = [
    REPO_ROOT / "examples" / "data" / "pha_r2048_d8_o0.h5",  # delay axis
    REPO_ROOT / "examples" / "data" / "pha_r2048_d1_o2.h5",  # derivative axis
]

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load the SSA reference and match each PHA run against it.
_, ssa_trajectory, ssa_events = load_results(SSA_PATH)
dt = ssa_trajectory.dt

matched_runs = []
for pha_path in PHA_PATHS:
    pha_metadata, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    matches = match_events(ssa_events, pha_events)
    ssa_lengths = (
        np.array([m.ssa_event.end_timestep - m.ssa_event.start_timestep for m in matches]) * dt
    )
    pha_lengths = (
        np.array([m.pha_event.end_timestep - m.pha_event.start_timestep for m in matches]) * dt
    )
    iou = np.array([m.intersection_length / m.union_length for m in matches])
    matched_runs.append((pha_metadata, ssa_lengths, pha_lengths, iou))

# %%
# Render. Both detectors discard events shorter than their ``min_duration``
# timesteps, which is why the lower-left corner of each panel is empty.
# High-IoU points draw last so overplotting does not bury them; the shared
# limits and diagonal make the two panels directly comparable.
figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True)

limit = 1.02 * max(
    max(ssa_lengths.max(), pha_lengths.max()) for _, ssa_lengths, pha_lengths, _ in matched_runs
)
for ax, tag, (pha_metadata, ssa_lengths, pha_lengths, iou) in zip(
    axes, ("(a)", "(b)"), matched_runs, strict=True
):
    draw_order = np.argsort(iou)
    scatter = ax.scatter(
        ssa_lengths[draw_order],
        pha_lengths[draw_order],
        c=iou[draw_order],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=6,
        linewidths=0,
    )
    ax.plot([0, limit], [0, limit], color="0.5", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_title(tag, loc="left")
    ax.set_title(f"delay {pha_metadata.delay}, max order {pha_metadata.max_derivative_order}")
    ax.set_xlabel("SSA event length (time units)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.annotate(f"{len(iou)} matched pairs", xy=(0.03, 0.97), xycoords="axes fraction", va="top")
axes[0].set_ylabel("PHA event length (time units)")
figure.colorbar(scatter, ax=axes, label="Overlap (IoU)", pad=0.02)
plt.show()
