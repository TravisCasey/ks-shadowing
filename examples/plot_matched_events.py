r"""
Matched events: SSA vs. PHA
============================

A match between the two detection methods is a connected component of the
bipartite overlap graph: SSA and PHA events on the same RPO are linked whenever
their timestep windows overlap, and a match gathers every event reachable
through such links. Component agreement is scored by the Jaccard index of the
two composite windows: covered timesteps in common over covered timesteps in
either.

Each match is one point at its composite SSA and PHA lengths, colored by Jaccard
index. Events with no overlapping partner on the same RPO appear in the
"unmatched" strips beside the axes: a strip left of the vertical axis for
PHA-only events and a strip below the horizontal axis for SSA-only events,
jittered within the strip for visibility.

One panel per embedding axis: the delay-axis setting (:math:`w = 25`,
:math:`\lambda = 1`) and the derivative-axis setting (:math:`w = 1`,
:math:`\lambda = 3`), each matched against the same SSA run. :math:`w` is the
delay window and :math:`\lambda` the number of derivative orders averaged over,
one more than the ``max_derivative_order`` the filenames carry. The two
embedding axes are shown independently: :math:`w > 1` is used only at
:math:`\lambda = 1`.
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
    REPO_ROOT / "examples" / "data" / "pha_r2048_d25_o0.h5",  # delay axis
    REPO_ROOT / "examples" / "data" / "pha_r2048_d1_o2.h5",  # derivative axis
]

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load the SSA reference and match each PHA run against it. Each match's
# composite windows give its point's coordinates and Jaccard index.
ssa_metadata, ssa_trajectory, ssa_events = load_results(SSA_PATH)
dt = ssa_trajectory.dt

matched_runs = []
for pha_path in PHA_PATHS:
    pha_metadata, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    matches = match_events(ssa_events, pha_events)
    ssa_lengths = np.array([match.ssa_length for match in matches]) * dt
    pha_lengths = np.array([match.pha_length for match in matches]) * dt
    jaccard = np.array([match.intersection_length / match.union_length for match in matches])
    matched_ssa_ids = {id(event) for match in matches for event in match.ssa_events}
    matched_pha_ids = {id(event) for match in matches for event in match.pha_events}
    unmatched_ssa = (
        np.array(
            [e.end_timestep - e.start_timestep for e in ssa_events if id(e) not in matched_ssa_ids]
        )
        * dt
    )
    unmatched_pha = (
        np.array(
            [e.end_timestep - e.start_timestep for e in pha_events if id(e) not in matched_pha_ids]
        )
        * dt
    )
    matched_runs.append(
        (pha_metadata, ssa_lengths, pha_lengths, jaccard, unmatched_ssa, unmatched_pha)
    )

# %%
# Render. Axis ranges are fixture-tuned: the largest match reaches composite
# lengths near (85, 89) time units, so shared square limits to 92 hold every
# point with the diagonal in view. High-Jaccard points draw last.
HIGH = 92.0

shortest = dt * min(
    ssa_metadata.min_duration,
    *(pha_metadata.min_duration for pha_metadata, *_ in matched_runs),
)
pad = 0.03 * (HIGH - shortest)
low = shortest - pad
strip = 0.09 * (HIGH - low)
gap = 0.25 * strip
strip_low = low - gap - strip

figure, axes = plt.subplots(2, 1, figsize=(3.4, 7.6), sharex=True, sharey=True)
rng = np.random.default_rng(0)
for ax, tag, run in zip(axes, ("(a)", "(b)"), matched_runs, strict=True):
    pha_metadata, ssa_lengths, pha_lengths, jaccard, unmatched_ssa, unmatched_pha = run
    draw_order = np.argsort(jaccard)
    scatter = ax.scatter(
        ssa_lengths[draw_order],
        pha_lengths[draw_order],
        c=jaccard[draw_order],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=6,
        linewidths=0,
    )
    ax.scatter(
        low - gap - strip * rng.random(len(unmatched_pha)),
        unmatched_pha,
        c=np.zeros(len(unmatched_pha)),
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=6,
        linewidths=0,
    )
    ax.scatter(
        unmatched_ssa,
        low - gap - strip * rng.random(len(unmatched_ssa)),
        c=np.zeros(len(unmatched_ssa)),
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=6,
        linewidths=0,
    )
    ax.axvline(low, color="0.3", linewidth=0.6)
    ax.axhline(low, color="0.3", linewidth=0.6)
    ax.plot([low, HIGH], [low, HIGH], color="0.5", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(strip_low, HIGH)
    ax.set_ylim(strip_low, HIGH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("PHA length (time units)")
    ax.annotate(
        f"{len(jaccard)} matches",
        xy=(0.84, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    ax.annotate(
        "unmatched",
        xy=(low - gap - strip / 2, HIGH - 0.02 * (HIGH - strip_low)),
        ha="center",
        va="top",
        rotation=90,
        fontsize=6,
        color="0.3",
    )
    ax.annotate(
        "unmatched",
        xy=(HIGH - 0.02 * (HIGH - strip_low), low - gap - strip / 2),
        ha="right",
        va="center",
        fontsize=6,
        color="0.3",
    )
    ax.set_title(tag, loc="left")
    ax.set_title(
        rf"$w = {pha_metadata.delay}$, "
        rf"$\lambda = {pha_metadata.max_derivative_order + 1}$"
    )
axes[1].set_xlabel("SSA length (time units)")
figure.colorbar(
    scatter,
    ax=list(axes),
    orientation="horizontal",
    label="Jaccard index",
    pad=0.02,
)
plt.show()
