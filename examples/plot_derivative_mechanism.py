"""
Two ingredients behind the derivative saturation
================================================

A small, illustrative probe of *why* the derivative sweep in the companion
:ref:`saturation example <sphx_glr_auto_examples_plot_derivative_saturation.py>` stops paying off.
The full-scale statistics live there; here we look at two mechanisms on two
disjoint trajectory windows, computed live from the public API at resolution 256
(persistence-diagram cardinality is resolution-independent). Only the two
shortest-period RPOs are used, and only two windows -- enough to show the shape
while keeping the phase count and runtime low.

Panel (a): scale. Differentiating a field multiplies Fourier mode ``q`` by
``(i q)^order``, so each added order inflates the Wasserstein magnitude it
produces. Across these windows the per-order scale grows roughly fivefold from
order 0 to order 5. PHA averages the per-order matrices with *equal* weight, so
order 5 alone accounts for about 45% of that unweighted mean of scales; a
high-order run is effectively dominated by its largest order. That is a
statement about the averaging step, not a literal detection vote but it is
exactly the kind of scale imbalance that would let extra orders stop helping.

Panel (b): redundancy. The high orders are also measuring nearly the same
thing. Rank-correlating each order's isolated per-timestep distance across the
pooled windows, orders 3-5 line up almost perfectly with one another (Spearman
around 0.85) while order 0 stands apart (around 0.3 against the rest). So the
added high orders are not contributing independent evidence; they re-measure
one signal at heavier weight. That is the source of the correct-RPO attribution
decline the saturation example shows at detection level.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import KSTrajectory, load_results, load_rpos
from ks_shadowing.pha import KSPersistenceTrajectory, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"

WINDOW_STARTS = (25000, 85000)  # disjoint, past the transient
WINDOW_TIMESTEPS = 150  # 300 pooled timesteps for the rank correlation
RESOLUTION = 256  # diagram cardinality is resolution-independent
NUM_RPOS = 2  # the two shortest-period RPOs (72, 79 phases)
NUM_ORDERS = 6
# Spearman cells above this are light (viridis) and need dark annotation text.
LIGHT_CELL = 0.6

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")

# %%
# Build the per-order RPO diagrams once, at the same sampling the committed
# detections used (read from the SSA metadata).
metadata, trajectory, _ = load_results(DATA_DIR / "ssa_r2048.h5")
rpos = load_rpos(REPO_ROOT / metadata.rpo_file)  # period-sorted; [:NUM_RPOS] shortest
rpo_trajectories = [
    KSTrajectory.from_rpo(
        rpos[index], RESOLUTION, downsample=metadata.downsample, native=metadata.native
    )
    for index in range(NUM_RPOS)
]
rpo_diagrams = [
    [
        KSPersistenceTrajectory.from_trajectory(rpo_trajectory, order=order)
        for order in range(NUM_ORDERS)
    ]
    for rpo_trajectory in rpo_trajectories
]

# %%
# Per (window, RPO, order): the Wasserstein scale (median of the full matrix) and the
# isolated per-timestep distance (min over phases, then over RPOs). The reduction
# order matters -- min over phases first, then over RPOs -- so it matches how
# detection collapses one order's matrix.
scale_samples: list[list[float]] = [[] for _ in range(NUM_ORDERS)]
isolated_columns: list[list[NDArray[np.float64]]] = [[] for _ in range(NUM_ORDERS)]
for start in WINDOW_STARTS:
    window = KSTrajectory(
        modes=trajectory[start : start + WINDOW_TIMESTEPS].modes,
        dt=trajectory.dt,
        resolution=RESOLUTION,
    )
    window_diagrams = [
        KSPersistenceTrajectory.from_trajectory(window, order=order) for order in range(NUM_ORDERS)
    ]
    for order in range(NUM_ORDERS):
        column = np.full(WINDOW_TIMESTEPS, np.inf)
        for rpo_index in range(NUM_RPOS):
            matrix = wasserstein_matrix(window_diagrams[order], rpo_diagrams[rpo_index][order])
            scale_samples[order].append(float(np.median(matrix)))
            column = np.minimum(column, matrix.min(axis=1))
        isolated_columns[order].append(column)

scales = np.array([np.median(samples) for samples in scale_samples])
relative = scales / scales[0]
vote_share = scales / scales.sum()  # order's share of the unweighted mean of scales
isolated = np.array([np.concatenate(columns) for columns in isolated_columns])  # (6, 300)
ranked = np.argsort(np.argsort(isolated, axis=1), axis=1)
spearman = np.corrcoef(ranked)  # (6, 6)

# %%
# Render: scale dominance (a) and inter-order redundancy (b). The relative
# scale and the vote share are the same curve up to a constant (both normalize
# ``scales``), so the annotation reports the share rather than a second line.
orders = list(range(NUM_ORDERS))
figure, (ax_scale, ax_redundancy) = plt.subplots(2, 1, figsize=(3.4, 5.6))

ax_scale.plot(orders, relative, color="black", marker="o")
ax_scale.set_title("(a)", loc="left")
ax_scale.set_xlabel("Derivative order")
ax_scale.set_ylabel("Wasserstein scale\n(relative to order 0)")
ax_scale.set_xticks(orders)
ax_scale.set_ylim(bottom=0)
ax_scale.annotate(
    f"order 5: {vote_share[5]:.0%} of the mean",
    xy=(5, relative[5]),
    xytext=(-8, 0),
    textcoords="offset points",
    ha="right",
    va="center",
)

image = ax_redundancy.imshow(spearman, cmap="viridis", vmin=0.0, vmax=1.0)
ax_redundancy.set_title("(b)", loc="left")
ax_redundancy.set_xticks(orders)
ax_redundancy.set_yticks(orders)
ax_redundancy.set_xlabel("Derivative order")
ax_redundancy.set_ylabel("Derivative order")
# The matrix is symmetric; annotate the lower triangle only.
for row in orders:
    for column in range(row + 1):
        value = spearman[row, column]
        ax_redundancy.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="black" if value > LIGHT_CELL else "white",
            fontsize=6,
        )
figure.colorbar(image, ax=ax_redundancy, label="Spearman correlation")

plt.show()
