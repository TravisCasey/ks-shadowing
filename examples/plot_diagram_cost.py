"""
Persistence diagram size and detection cost
============================================

Higher spatial derivatives introduce more critical points, so the sublevel-set
diagrams carry more pairs. Hera's geometric auction scales empirically as
:math:`n^{1.6}` in the number of pairs per diagram, which accounts for recorded
runtimes growing faster than the derivative count alone.

The predicted curve has no fitted parameters: it takes the measured
cardinalities, applies the published exponent, and anchors to the 0-derivative
runtime. It therefore inherits that run's fixed setup cost and then multiplies
it, which is why it sits above the recorded times at the top of the range.

Cardinality does not depend on the spatial resolution the trajectory is loaded
at. The 17-mode truncation fixes how many extrema a field can have, so the
curves for every resolution coincide, which is also why the PHA runtime curves
in :ref:`the resolution example
<sphx_glr_auto_examples_plot_runtime_vs_resolution.py>` stay flat.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ks_shadowing import KSTrajectory, load_results
from ks_shadowing.pha import KSPersistenceTrajectory

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
TRAJECTORY_PATH = DATA_DIR / "ssa_r2048.h5"
PHA_PATHS = [DATA_DIR / f"pha_r2048_d1_o{order}.h5" for order in range(6)]
DERIVATIVE_ORDERS = range(6)
RESOLUTIONS = (256, 512, 2048)
REFERENCE_RESOLUTION = 2048
SAMPLE_TIMESTEPS = 400
SAMPLE_START = 20000
HERA_EXPONENT = 1.6
SECONDS_PER_MINUTE = 60.0
OBSERVED_COLOR = "black"
MODEL_COLOR = "0.45"

# %%
# Mean pairs per diagram, per derivative order, at several spatial resolutions.
_, trajectory, _ = load_results(TRAJECTORY_PATH)
window = trajectory[SAMPLE_START : SAMPLE_START + SAMPLE_TIMESTEPS]

cardinalities = {}
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
# Recorded runtimes of the derivative sweep, against the cost the measured
# cardinalities predict. ``HERA_EXPONENT`` is the empirical scaling of Hera's
# geometric auction in the number of pairs per diagram, reported by
# `Kerber, Morozov and Nigmetov (2017) <https://doi.org/10.1145/3064175>`_.
# Detection computes one Wasserstein matrix per order, so the predicted cost of
# a run with max order ``k`` is the cumulative sum over orders, anchored to
# the observed order-0 runtime.
observed_minutes = []
for path in PHA_PATHS:
    metadata, _, _ = load_results(path)
    observed_minutes.append(metadata.elapsed_seconds / SECONDS_PER_MINUTE)

predicted = np.cumsum(np.array(cardinalities[REFERENCE_RESOLUTION]) ** HERA_EXPONENT)
predicted = predicted / predicted[0] * observed_minutes[0]

# %%
# Render.
figure, (ax_pairs, ax_cost) = plt.subplots(1, 2, figsize=(13, 5))

for index, resolution in enumerate(RESOLUTIONS):
    ax_pairs.plot(
        list(DERIVATIVE_ORDERS),
        cardinalities[resolution],
        color=str(0.15 * index),
        marker=Line2D.filled_markers[index % len(Line2D.filled_markers)],
        linestyle=["-", "--", ":"][index % 3],
        label=f"resolution {resolution}",
    )
ax_pairs.set_xlabel("Derivative order")
ax_pairs.set_ylabel("Mean pairs per diagram")
ax_pairs.set_title("Diagram size grows with derivative order")
ax_pairs.set_ylim(bottom=0)
ax_pairs.legend()

ax_cost.plot(
    list(DERIVATIVE_ORDERS),
    predicted,
    color=MODEL_COLOR,
    linestyle="--",
    label=f"predicted, $n^{{{HERA_EXPONENT}}}$",
)
ax_cost.plot(
    list(DERIVATIVE_ORDERS),
    observed_minutes,
    color=OBSERVED_COLOR,
    marker="o",
    linestyle="none",
    label="recorded runtime",
)
ax_cost.set_xlabel("Max derivative order")
ax_cost.set_ylabel("Detection runtime (minutes)")
ax_cost.set_title("Cost follows diagram size")
ax_cost.set_ylim(bottom=0)
ax_cost.legend()

plt.tight_layout()
plt.show()
