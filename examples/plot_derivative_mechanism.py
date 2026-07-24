r"""
Two ingredients behind the derivative saturation
================================================

A small, illustrative probe of *why* the derivative sweep in the companion
:ref:`sphx_glr_auto_examples_plot_derivative_saturation.py` stops paying off.
The full-scale statistics live there; here we look at two mechanisms on two
disjoint trajectory windows, computed live from the public API at resolution 256
(persistence-diagram cardinality is resolution-independent). Only the two
shortest-period RPOs are used, and only two windows -- enough to show the shape
while keeping the phase count and runtime low.

Left panel: scale. Differentiating a field multiplies Fourier mode ``q`` by
``(i q)^order``, so each added order inflates the Wasserstein magnitude it
produces. Across these windows the per-order scale grows roughly fivefold from
order 0 to order 5. PHA averages the per-order matrices with *equal* weight, so
order 5 alone accounts for about 44% of that unweighted mean of scales; a
high-order run is effectively dominated by its largest order. That is a
statement about the averaging step, not a literal detection vote but it is
exactly the kind of scale imbalance that would let extra orders stop helping.

Right panel: redundancy. The high orders are also measuring nearly the same
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

from ks_shadowing import load_results, load_rpos
from ks_shadowing.core.trajectory import KSTrajectory
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
# Per (window, order): the Wasserstein scale (median of the full matrix) and the
# isolated per-timestep distance (min over phases, then over RPOs). The reduction
# order matters -- min over phases first, then over RPOs -- so it matches how
# detection collapses one order's matrix.
scale_samples: list[list[float]] = [[] for _ in range(NUM_ORDERS)]
isolated_columns: list[list[np.ndarray]] = [[] for _ in range(NUM_ORDERS)]
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
isolated = np.array([np.concatenate(columns) for columns in isolated_columns])  # (6, 450)
ranked = np.argsort(np.argsort(isolated, axis=1), axis=1)
spearman = np.corrcoef(ranked)  # (6, 6)

# %%
# Render: scale dominance (left) and inter-order redundancy (right).
orders = list(range(NUM_ORDERS))
figure, (ax_scale, ax_redundancy) = plt.subplots(1, 2, figsize=(13, 5))

# The relative scale and the vote share are the same curve up to a constant
# (both normalize ``scales``), so plot it once and let the right axis re-read it
# as a share of the unweighted mean rather than drawing a redundant second line.
ax_scale.plot(orders, relative, color="black", marker="o")
ax_scale.set_xlabel("Derivative order")
ax_scale.set_ylabel("Wasserstein scale (relative to order 0)")
ax_scale.set_title("Differentiation inflates the high orders (~5x)")
ax_scale.set_xticks(orders)
ax_scale.set_ylim(bottom=0)
ax_scale.annotate(
    f"order 5: {vote_share[5]:.0%} of the mean",
    xy=(5, relative[5]),
    xytext=(-8, 4),
    textcoords="offset points",
    ha="right",
    fontsize="small",
)

share_per_unit = scales[0] / scales.sum()  # relative-scale -> vote-share conversion
ax_share = ax_scale.twinx()
low, high = ax_scale.get_ylim()
ax_share.set_ylim(low * share_per_unit, high * share_per_unit)
ax_share.set_ylabel("Share of the unweighted mean of scales")

image = ax_redundancy.imshow(spearman, cmap="viridis", vmin=0.0, vmax=1.0)
ax_redundancy.set_xticks(orders)
ax_redundancy.set_yticks(orders)
ax_redundancy.set_xlabel("Derivative order")
ax_redundancy.set_ylabel("Derivative order")
ax_redundancy.set_title("High orders measure nearly the same thing")
for row in orders:
    for column in orders:
        value = spearman[row, column]
        ax_redundancy.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="black" if value > LIGHT_CELL else "white",
            fontsize="small",
        )
figure.colorbar(image, ax=ax_redundancy, label="Spearman correlation")

plt.tight_layout()
plt.show()
