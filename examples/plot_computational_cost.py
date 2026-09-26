r"""
Computational cost: resolution, diagram size, derivative order
==============================================================

Three views of detection cost, across two figures. :math:`m` is the embedding
length: the number of persistence diagrams each comparison averages over,
taken from consecutive timesteps (``PHA--DELAY``) or from derivative orders
:math:`0` to :math:`m - 1` (``PHA--DERIV``, one more than the
``max_derivative_order`` the filenames carry). The two embedding methods are
shown separately, never combined.

The first figure: wall-clock detection time against the spatial resolution
:math:`N`, the number of grid points the trajectory is loaded at, for ``SSA``,
the delay-embedded ``PHA--DELAY`` at :math:`m = 17`, and the derivative-embedded
``PHA--DERIV`` at :math:`m = 2`. SSA evaluates L2 distances in physical space,
so its cost grows with resolution; PHA computes Wasserstein distances between
persistence diagrams, whose cost is dominated by trajectory length rather than
grid size, so its curves stay nearly flat. Each extra derivative order adds one
Wasserstein matrix per phase, having rougher fields with more pairs per diagram
(panel (a) of the second figure), so the :math:`m = 2` curve repeats the flat
resolution profile at higher cost.

The second figure stacks the two derivative-cost views.

Panel (a): higher spatial derivatives introduce more critical points, so the
sublevel-set diagrams carry more pairs. Cardinality does not depend on the
spatial resolution the trajectory is loaded at -- the 17-mode truncation fixes
how many extrema a field can have, so the markers for every resolution
coincide. That is also why the PHA curves in the first figure stay flat.

Panel (b): recorded runtimes of the ``PHA--DERIV`` sweep at
resolution 2048 (:math:`m = 4` to :math:`6` exist only in that sweep),
against the cost the measured cardinalities predict. Hera's geometric auction
scales empirically as :math:`n^{1.6}` in the number of pairs per diagram, which
accounts for runtimes growing faster than the derivative count alone. The
predicted curve has no fitted parameters: it takes the measured cardinalities
from panel (a), applies the published exponent, and anchors to the
0-derivative runtime. It therefore inherits that run's fixed setup cost and
then multiplies it, which is why it sits above the recorded times at the top
of the range.
"""

import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing import KSTrajectory, load_results
from ks_shadowing.pha import KSPersistenceTrajectory

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATTERN = re.compile(r"^ssa_r(\d+)\.h5$")
PHA_PATTERN = re.compile(r"^pha_r(\d+)_d(\d+)_o(\d+)\.h5$")
TRAJECTORY_PATH = DATA_DIR / "ssa_r2048.h5"
DERIVATIVE_ORDERS = range(6)
EMBEDDING_LENGTHS = range(1, 7)
RESOLUTIONS = (256, 512, 2048)
REFERENCE_RESOLUTION = 2048
DELAY_EMBEDDING_LENGTH = 17
SAMPLE_TIMESTEPS = 400
SAMPLE_START = 20000
HERA_EXPONENT = 1.6
SECONDS_PER_MINUTE = 60.0

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# One fixed color and marker per derivative order, shared across the gallery
# figures: viridis sampled light to dark with increasing order. SSA is always
# black.
ORDER_COLORS = plt.get_cmap("viridis")(np.linspace(0.78, 0.0, 6))
ORDER_MARKERS = ("o", "s", "^", "v", "D", "P")
PHA_DELAY = "PHA\u2013DELAY"
PHA_DERIV = "PHA\u2013DERIV"


def _elapsed_seconds(path: Path) -> float:
    """Read the ``elapsed_seconds`` attribute without loading the trajectory."""
    with h5py.File(path, "r") as f:
        return float(f.attrs["elapsed_seconds"])


# %%
# SSA: one runtime per resolution.
ssa_runtimes: dict[int, float] = {}
for path in DATA_DIR.glob("ssa_r*.h5"):
    match = SSA_PATTERN.match(path.name)
    if match is None:
        continue
    ssa_runtimes[int(match.group(1))] = _elapsed_seconds(path)

# %%
# PHA: runtimes grouped by max_derivative_order, then resolution, then delay.
# The max-order-0 runs at delay 17 feed the PHA--DELAY curve; the higher orders
# feed the PHA--DERIV curve and the derivative-sweep panel.
pha_runtimes: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
for path in DATA_DIR.glob("pha_r*_d*_o*.h5"):
    match = PHA_PATTERN.match(path.name)
    if match is None:
        continue
    resolution, delay, max_order = (int(group) for group in match.groups())
    pha_runtimes[max_order][resolution][delay] = _elapsed_seconds(path)

# %%
# Mean pairs per diagram, per derivative order, at several spatial resolutions.
_, trajectory, _ = load_results(TRAJECTORY_PATH)
window = trajectory[SAMPLE_START : SAMPLE_START + SAMPLE_TIMESTEPS]

cardinalities: dict[int, list[float]] = {}
for resolution in RESOLUTIONS:
    resampled = KSTrajectory(modes=window.modes, dt=window.dt, resolution=resolution)
    cardinalities[resolution] = [
        float(
            np.mean(
                [
                    diagram.shape[0]
                    for diagram in KSPersistenceTrajectory.from_trajectory(
                        resampled, order=order
                    ).diagrams
                ]
            )
        )
        for order in DERIVATIVE_ORDERS
    ]

# %%
# Recorded runtimes of the derivative sweep, against the cost the
# measured cardinalities predict. ``HERA_EXPONENT`` is the empirical scaling of
# Hera's geometric auction in the number of pairs per diagram, reported by
# `Kerber, Morozov and Nigmetov (2017) <https://doi.org/10.1145/3064175>`_.
# Detection computes one Wasserstein matrix per order, so the predicted cost of
# a run over m orders is the cumulative sum over orders, anchored to
# the observed m = 1 runtime.
observed_minutes = (
    np.array([pha_runtimes[order][REFERENCE_RESOLUTION][1] for order in DERIVATIVE_ORDERS])
    / SECONDS_PER_MINUTE
)
predicted = np.cumsum(np.array(cardinalities[REFERENCE_RESOLUTION]) ** HERA_EXPONENT)
predicted = predicted / predicted[0] * observed_minutes[0]

# %%
# Runtime against spatial resolution.
figure_runtime, ax_runtime = plt.subplots(figsize=(3.4, 2.6))

ssa_resolutions = np.array(sorted(ssa_runtimes))
ssa_minutes = (
    np.array([ssa_runtimes[resolution] for resolution in ssa_resolutions]) / SECONDS_PER_MINUTE
)
ax_runtime.plot(ssa_resolutions, ssa_minutes, color="black", marker="o", label="SSA")

by_resolution = pha_runtimes[0]
pha_resolutions = np.array(sorted(by_resolution))
pha_minutes = (
    np.array([by_resolution[resolution][DELAY_EMBEDDING_LENGTH] for resolution in pha_resolutions])
    / SECONDS_PER_MINUTE
)
ax_runtime.plot(
    pha_resolutions,
    pha_minutes,
    color=ORDER_COLORS[0],
    marker=ORDER_MARKERS[0],
    label=rf"{PHA_DELAY} $m={DELAY_EMBEDDING_LENGTH}$",
)

# The derivative-embedded reference setting across the same resolutions.
deriv_by_resolution = pha_runtimes[1]
deriv_resolutions = np.array(sorted(deriv_by_resolution))
deriv_minutes = (
    np.array([deriv_by_resolution[resolution][1] for resolution in deriv_resolutions])
    / SECONDS_PER_MINUTE
)
ax_runtime.plot(
    deriv_resolutions,
    deriv_minutes,
    color=ORDER_COLORS[1],
    marker=ORDER_MARKERS[1],
    label=rf"{PHA_DERIV} $m=2$",
)

ax_runtime.set_ylim(bottom=0)
ax_runtime.set_xticks((256, 1024, 2048))
ax_runtime.set_xlabel("$N$")
ax_runtime.set_ylabel("Runtime (minutes)")
ax_runtime.legend(prop={"family": "monospace"})

plt.show()

# %%
# Diagram cardinality per derivative order, and the derivative-sweep runtime
# it predicts.
figure_derivative, (ax_pairs, ax_cost) = plt.subplots(2, 1, figsize=(3.4, 4.8))

# All resolutions produce the same cardinalities; concentric open markers of
# decreasing size make the coincidence visible instead of hiding the
# coincident curves.
for resolution, size in zip(RESOLUTIONS, (7.0, 4.5, 2.0), strict=True):
    ax_pairs.plot(
        list(DERIVATIVE_ORDERS),
        cardinalities[resolution],
        color="black",
        linestyle="-" if resolution == REFERENCE_RESOLUTION else "none",
        linewidth=0.8,
        marker="o",
        markersize=size,
        markerfacecolor="none",
        markeredgewidth=0.8,
        label=f"resolution {resolution}",
    )
ax_pairs.set_title("(a)", loc="left")
ax_pairs.set_xlabel("Derivative order")
ax_pairs.set_ylabel("Mean pairs per diagram")
ax_pairs.set_ylim(bottom=0)
ax_pairs.set_xticks(list(DERIVATIVE_ORDERS))
ax_pairs.legend()

ax_cost.plot(
    list(EMBEDDING_LENGTHS),
    predicted,
    color="0.45",
    linestyle="--",
    label="predicted",
)
ax_cost.plot(
    list(EMBEDDING_LENGTHS),
    observed_minutes,
    color="black",
    marker="o",
    linestyle="none",
    label="recorded runtime",
)
ax_cost.set_title("(b)", loc="left")
ax_cost.set_title(PHA_DERIV, fontfamily="monospace")
ax_cost.set_xlabel(r"Embedding length $m$")
ax_cost.set_ylabel("Runtime (minutes)")
ax_cost.set_ylim(bottom=0)
ax_cost.set_xticks(list(EMBEDDING_LENGTHS))
ax_cost.legend(loc="upper left")

plt.show()
