r"""
Matched events: SSA vs. PHA
============================

A match between the two detection methods links SSA and PHA events on the same
RPO whenever their timestep windows overlap. The first figure treats each
overlapping pair as its own match, scored two ways: by the Jaccard index of
the two windows, and by the overlap coefficient. The overlap coefficient equals
1 whenever the shorter window lies inside the longer one, so beside the Jaccard
index it separates the two ways a pair can disagree: a high overlap coefficient
at a low Jaccard index means well-aligned windows of mismatched length, while a
low overlap coefficient means genuine misalignment.

An event may be well-represented by the other method only as several shorter
events that jointly cover it, so the second figure refines the pairing
transitively: a match is a connected component of the bipartite overlap graph,
gathering every event reachable through overlap links and scored by the Jaccard
index of the two composite windows. Events with no overlapping partner on the
same RPO are identical under both pairings and appear once, in the "unmatched"
strips beside the first figure's axes: a strip left of the vertical axis for
PHA-only events and a strip below the horizontal axis for SSA-only events.

Rows of both figures are the two embedding strategies, matched against the same
``SSA`` run: ``PHA--DELAY`` (:math:`w = 17`, :math:`\lambda = 1`) and
``PHA--DERIV`` (:math:`w = 1`, :math:`\lambda = 2`). :math:`w` is the delay
window and :math:`\lambda` the number of derivative orders averaged over; the
two embedding axes are shown independently (:math:`w > 1` only at
:math:`\lambda = 1`). Scatter columns draw each match as one point at its SSA
and PHA lengths, colored by the panel's measure, with the unmatched events
jittered within their strip; the leftmost column of each figure bins the same
matches into square pixels colored by the number of matches per bin on a
logarithmic scale.
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
PHA_DELAY = "PHA\u2013DELAY"
PHA_DERIV = "PHA\u2013DERIV"

# %%
# Load the SSA reference and match each PHA run against it under both pairings.
# A match's (composite) windows give its coordinates, Jaccard index, and
# overlap coefficient; the unmatched events are the same either way, so they
# are computed once from the transitive matches.
ssa_metadata, ssa_trajectory, ssa_events = load_results(SSA_PATH)
dt = ssa_trajectory.dt

matched_runs = []
for pha_path in PHA_PATHS:
    pha_metadata, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    by_mode = {}
    for transitive in (False, True):
        matches = match_events(ssa_events, pha_events, transitive=transitive)
        ssa_lengths = np.array([match.ssa_length for match in matches])
        pha_lengths = np.array([match.pha_length for match in matches])
        intersections = np.array([match.intersection_length for match in matches])
        unions = np.array([match.union_length for match in matches])
        by_mode[transitive] = (
            ssa_lengths * dt,
            pha_lengths * dt,
            intersections / unions,
            intersections / np.minimum(ssa_lengths, pha_lengths),
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
# Shared layout. Axis ranges are fixture-tuned. High-metric points draw last.
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
    for ssa_lengths, pha_lengths, _jaccard, _overlap in by_mode.values():
        counts, _, _ = np.histogram2d(ssa_lengths, pha_lengths, bins=[bin_edges, bin_edges])
        max_count = max(max_count, counts.max())
    for unmatched in (unmatched_ssa, unmatched_pha):
        strip_counts, _ = np.histogram(unmatched, bins=bin_edges)
        max_count = max(max_count, strip_counts.max())
count_norm = LogNorm(vmin=1, vmax=max_count)


def draw_frame(ax, tag: str, pha_metadata, transitive: bool) -> None:
    """Strip separators and labels, diagonal, limits, and titles for every panel.

    The transitive figure omits the unmatched strips (identical to the
    non-transitive figure's), so its panels start at ``low``.
    """
    if not transitive:
        ax.axvline(low, color="0.3", linewidth=0.6)
        ax.axhline(low, color="0.3", linewidth=0.6)
        ax.annotate(
            "unmatched",
            xy=(HIGH - 0.02 * (HIGH - strip_low), low - gap - strip / 2),
            ha="right",
            va="center",
            fontsize=6,
            color="0.3",
        )
        ax.annotate(
            "unmatched",
            xy=(low - gap - strip / 2, HIGH - 0.02 * (HIGH - strip_low)),
            ha="right",
            va="center",
            rotation=90,
            rotation_mode="anchor",
            fontsize=6,
            color="0.3",
        )
    ax.plot([low, HIGH], [low, HIGH], color="0.5", linestyle="--", linewidth=0.8, zorder=0)
    lower = low if transitive else strip_low
    ax.set_xlim(lower, HIGH)
    ax.set_ylim(lower, HIGH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(tag, loc="left")
    ax.set_title(
        PHA_DELAY if pha_metadata.delay > 1 else PHA_DERIV,
        fontfamily="monospace",
    )


def draw_scatter(ax, run, transitive: bool, rng, metric: str = "jaccard") -> PathCollection:
    """Metric-colored scatter, with jittered unmatched strips when non-transitive."""
    _pha_metadata, by_mode, unmatched_ssa, unmatched_pha = run
    ssa_lengths, pha_lengths, jaccard, overlap = by_mode[transitive]
    values = jaccard if metric == "jaccard" else overlap
    draw_order = np.argsort(values)
    scatter = ax.scatter(
        ssa_lengths[draw_order],
        pha_lengths[draw_order],
        c=values[draw_order],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=2,
        linewidths=0,
    )
    if not transitive:
        ax.scatter(
            low - gap - strip * rng.random(len(unmatched_pha)),
            unmatched_pha,
            c=np.zeros(len(unmatched_pha)),
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=2,
            linewidths=0,
        )
        ax.scatter(
            unmatched_ssa,
            low - gap - strip * rng.random(len(unmatched_ssa)),
            c=np.zeros(len(unmatched_ssa)),
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=2,
            linewidths=0,
        )
    return scatter


def draw_density(ax, run, transitive: bool) -> QuadMesh:
    """Binned pixels colored by matches per bin on the shared log scale."""
    _pha_metadata, by_mode, unmatched_ssa, unmatched_pha = run
    ssa_lengths, pha_lengths, _jaccard, _overlap = by_mode[transitive]
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
    if not transitive:
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


def render_figure(
    transitive: bool, columns: tuple[str, ...], figsize: tuple[float, float], label: str
) -> plt.Figure:
    """One figure: rows are embedding settings, columns metric scatters/density."""
    colorbar_labels = {
        "jaccard": "Jaccard index",
        "overlap": "Overlap coefficient",
        "density": "Matches per bin",
    }
    figure, axes = plt.subplots(2, len(columns), figsize=figsize, sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    mappables: dict[str, PathCollection | QuadMesh] = {}
    for row, run in enumerate(matched_runs):
        for column, kind in enumerate(columns):
            if kind == "density":
                mappables[kind] = draw_density(axes[row, column], run, transitive)
            else:
                mappables[kind] = draw_scatter(axes[row, column], run, transitive, rng, metric=kind)
            tag = f"({'abcdef'[row * len(columns) + column]})"
            draw_frame(axes[row, column], tag, run[0], transitive)
        axes[row, 0].set_ylabel(r"$\mathtt{PHA}$ length (time units)")
    for column in range(len(columns)):
        axes[1, column].set_xlabel(r"$\mathtt{SSA}$ length (time units)")
    figure.suptitle(label)
    for column, kind in enumerate(columns):
        figure.colorbar(
            mappables[kind],
            ax=list(axes[:, column]),
            orientation="horizontal",
            label=colorbar_labels[kind],
            pad=0.04,
        )
    return figure


# %%
# Non-transitive matching: every overlapping SSA/PHA pair is one match, so an
# event overlapping several events of the other method contributes several
# points.
figure = render_figure(
    transitive=False,
    columns=("density", "jaccard", "overlap"),
    figsize=(7.0, 5.8),
    label="Matched events",
)
plt.show()

# %%
# Transitive matching refines the same overlap graph into connected components:
# events that jointly cover a partner merge into one composite match, so the
# count drops and coverage-splitting no longer sabotages the Jaccard index.
figure = render_figure(
    transitive=True,
    columns=("density", "jaccard"),
    figsize=(3.4, 4.4),
    label="Transitively matched events",
)
plt.show()
