r"""
Matched events: SSA vs. PHA
============================

A match between the two detection methods links SSA and PHA events on the same
RPO whenever their timestep windows overlap. The first figure treats each
overlapping pair as its own match, scored by the Jaccard index of the two
windows: timesteps in common over timesteps in either.

An event may be well-represented by the other method only as several shorter
events that jointly cover it, so the second figure refines the pairing
transitively: a match is a connected component of the bipartite overlap graph,
gathering every event reachable through overlap links and scored by the Jaccard
index of the two composite windows. Events with no overlapping partner on the
same RPO are identical under both pairings and appear in the "unmatched" strips
beside the axes: a strip left of the vertical axis for PHA-only events and a
strip below the horizontal axis for SSA-only events.

Both figures share one layout. Rows are the two embedding axes, matched
against the same SSA run: the delay-axis setting (:math:`w = 17`,
:math:`\lambda = 1`) and the derivative-axis setting (:math:`w = 1`,
:math:`\lambda = 2`). :math:`w` is the delay window and :math:`\lambda` the
number of derivative orders averaged over; the two embedding axes are shown
independently (:math:`w > 1` only at :math:`\lambda = 1`). The left column draws
each match as one point at its SSA and PHA lengths, colored by Jaccard index,
with unmatched events jittered within their strip. The right column bins the
same matches into square pixels colored by the number of matches per bin on a
logarithmic scale shared by both figures, with the strips binned the same way,
so each row pairs the agreement view (Jaccard) with the density view (counts) of
the same matches.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import LogNorm

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
    REPO_ROOT / "examples" / "data" / "pha_r2048_d17_o0.h5",  # delay axis
    REPO_ROOT / "examples" / "data" / "pha_r2048_d1_o1.h5",  # derivative axis
]

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Load the SSA reference and match each PHA run against it under both pairings.
# A match's (composite) windows give its coordinates and Jaccard index; the
# unmatched events are the same either way, so they are computed once from the
# transitive matches.
ssa_metadata, ssa_trajectory, ssa_events = load_results(SSA_PATH)
dt = ssa_trajectory.dt

matched_runs = []
for pha_path in PHA_PATHS:
    pha_metadata, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    by_mode = {}
    for transitive in (False, True):
        matches = match_events(ssa_events, pha_events, transitive=transitive)
        by_mode[transitive] = (
            np.array([match.ssa_length for match in matches]) * dt,
            np.array([match.pha_length for match in matches]) * dt,
            np.array([match.intersection_length / match.union_length for match in matches]),
        )
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
    matched_runs.append((pha_metadata, by_mode, unmatched_ssa, unmatched_pha))

# %%
# Shared layout. Axis ranges are fixture-tuned. High-Jaccard points draw last.
HIGH = 55.0
BIN_WIDTH = 1.7
DENSITY_CMAP = "magma_r"

shortest = dt * min(
    ssa_metadata.min_duration,
    *(pha_metadata.min_duration for pha_metadata, *_ in matched_runs),
)
pad = 0.03 * (HIGH - shortest)
low = shortest - pad
strip = 0.09 * (HIGH - low)
gap = 0.25 * strip
strip_low = low - gap - strip
bin_edges = np.arange(low, HIGH + BIN_WIDTH, BIN_WIDTH)

max_count = 0
for _pha_metadata, by_mode, unmatched_ssa, unmatched_pha in matched_runs:
    for ssa_lengths, pha_lengths, _jaccard in by_mode.values():
        counts, _, _ = np.histogram2d(ssa_lengths, pha_lengths, bins=[bin_edges, bin_edges])
        max_count = max(max_count, counts.max())
    for unmatched in (unmatched_ssa, unmatched_pha):
        strip_counts, _ = np.histogram(unmatched, bins=bin_edges)
        max_count = max(max_count, strip_counts.max())
count_norm = LogNorm(vmin=1, vmax=max_count)


def draw_frame(ax, tag: str, pha_metadata) -> None:
    """Strip separators, diagonal, limits, and titles shared by every panel."""
    ax.axvline(low, color="0.3", linewidth=0.6)
    ax.axhline(low, color="0.3", linewidth=0.6)
    ax.plot([low, HIGH], [low, HIGH], color="0.5", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(strip_low, HIGH)
    ax.set_ylim(strip_low, HIGH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(tag, loc="left")
    ax.set_title(
        rf"$w = {pha_metadata.delay}$, "
        rf"$\lambda = {pha_metadata.max_derivative_order + 1}$"
    )


def draw_scatter(ax, run, transitive: bool, rng) -> PathCollection:
    """Jaccard-colored scatter with jittered unmatched strips."""
    _pha_metadata, by_mode, unmatched_ssa, unmatched_pha = run
    ssa_lengths, pha_lengths, jaccard = by_mode[transitive]
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
    return scatter


def draw_density(ax, run, transitive: bool) -> QuadMesh:
    """Binned pixels colored by matches per bin on the shared log scale."""
    _pha_metadata, by_mode, unmatched_ssa, unmatched_pha = run
    ssa_lengths, pha_lengths, _jaccard = by_mode[transitive]
    counts, _, _ = np.histogram2d(ssa_lengths, pha_lengths, bins=[bin_edges, bin_edges])
    pha_strip_counts, _ = np.histogram(unmatched_pha, bins=bin_edges)
    ssa_strip_counts, _ = np.histogram(unmatched_ssa, bins=bin_edges)

    # pcolormesh maps C rows to y, so the (x, y)-indexed histogram transposes;
    # empty bins become NaN so they render as background rather than count 0.
    mesh = ax.pcolormesh(
        bin_edges,
        bin_edges,
        np.where(counts > 0, counts, np.nan).T,
        cmap=DENSITY_CMAP,
        norm=count_norm,
    )
    strip_edges = np.array([low - gap - strip, low - gap])
    ax.pcolormesh(
        strip_edges,
        bin_edges,
        np.where(pha_strip_counts > 0, pha_strip_counts, np.nan)[:, np.newaxis],
        cmap=DENSITY_CMAP,
        norm=count_norm,
    )
    ax.pcolormesh(
        bin_edges,
        strip_edges,
        np.where(ssa_strip_counts > 0, ssa_strip_counts, np.nan)[np.newaxis, :],
        cmap=DENSITY_CMAP,
        norm=count_norm,
    )
    return mesh


def render_figure(transitive: bool, label: str) -> plt.Figure:
    """One 2x2 figure: rows are embedding settings, columns scatter/density."""
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 7.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    tags = np.array([["(a)", "(b)"], ["(c)", "(d)"]])
    for row, run in enumerate(matched_runs):
        scatter = draw_scatter(axes[row, 0], run, transitive, rng)
        mesh = draw_density(axes[row, 1], run, transitive)
        for column in range(2):
            draw_frame(axes[row, column], tags[row, column], run[0])
        axes[row, 0].set_ylabel("PHA length (time units)")
    for column in range(2):
        axes[1, column].set_xlabel("SSA length (time units)")
    figure.suptitle(label)
    figure.colorbar(
        scatter,
        ax=list(axes[:, 0]),
        orientation="horizontal",
        label="Jaccard index",
        pad=0.04,
    )
    figure.colorbar(
        mesh,
        ax=list(axes[:, 1]),
        orientation="horizontal",
        label="Matches per bin",
        pad=0.04,
    )
    return figure


# %%
# Non-transitive matching: every overlapping SSA/PHA pair is one match, so an
# event overlapping several events of the other method contributes several
# points.
figure = render_figure(transitive=False, label="Non-transitive matches")
plt.show()

# %%
# Transitive matching refines the same overlap graph into connected components:
# events that jointly cover a partner merge into one composite match, so the
# count drops and coverage-splitting no longer sabotages the Jaccard index.
figure = render_figure(transitive=True, label="Transitive matches")
plt.show()
