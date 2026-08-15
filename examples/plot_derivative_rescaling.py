r"""
Per-order rescaling of the derivative sweep
===========================================

The ``rescale_orders`` option divides each spatial-derivative order's
Wasserstein column by a per-order median scale before the orders are averaged
together. The ``pha_r2048_d1_o{0..5}_rescaled.h5`` files repeat the
:math:`\lambda = 1` through :math:`6` sweep with that option on; the
``pha_r2048_d1_o{0..5}.h5`` files are the raw sweep with it off. Here
:math:`\lambda` is the number of derivative orders averaged over, one more than
the ``max_derivative_order`` the filenames carry.

Agreement is scored over the (RPO, timestep) cell grid, so a run that flags the
right timestep against the wrong orbit is penalized; the
:ref:`agreement example
<sphx_glr_auto_examples_plot_coverage_vs_embedding.py>` defines the measure.

The two sweeps track each other closely. At :math:`\lambda = 1` they are
identical; through :math:`\lambda = 3` the rescaled precision, F1, and
correct-RPO attribution differ from the raw curves only at the third decimal.
The sweeps separate only at high order: from :math:`\lambda = 4` onward
rescaling makes the decline gentler, but it does not lift the peak, which both
sweeps reach at :math:`\lambda = 3`.

Every quantity plotted here is agreement with the SSA reference, not absolute
correctness: the SSA detector is the yardstick, not ground truth. Rescaling
the orders does not beat simply capping the sweep at two or three derivative
orders. It slows the high-order erosion the
:ref:`saturation example <sphx_glr_auto_examples_plot_derivative_saturation.py>`
shows, without removing it.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ks_shadowing import assert_same_trajectory, events_to_union_mask, load_results, load_rpos

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
RAW_PATHS = [DATA_DIR / f"pha_r2048_d1_o{order}.h5" for order in range(6)]
RESCALED_PATHS = [DATA_DIR / f"pha_r2048_d1_o{order}_rescaled.h5" for order in range(6)]
LAMBDAS = range(1, 7)

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# One fixed color and marker per agreement metric, shared with the saturation
# figure (Tol bright palette). Sweeps are told apart by linestyle: raw solid
# with filled markers, rescaled dashed with open markers.
METRIC_STYLES: dict[str, dict[str, Any]] = {
    "Precision": {"color": "#4477AA", "marker": "o"},
    "F1": {"color": "#EE6677", "marker": "s"},
    "Recall": {"color": "#CCBB44", "marker": "^"},
}
RESCALED_STYLE: dict[str, Any] = {
    "linestyle": "--",
    "markerfacecolor": "none",
    "markeredgewidth": 0.8,
}

# %%
# SSA reference: per-RPO cell grid, plus the union mask the fixed common
# set in panel (c) is built from.
metadata, trajectory, ssa_events = load_results(SSA_PATH)
num_timesteps = trajectory.num_timesteps
num_rpos = len(load_rpos(REPO_ROOT / metadata.rpo_file))
ssa_mask = events_to_union_mask(ssa_events, num_timesteps)
ssa_rpo = np.zeros((num_rpos, num_timesteps), dtype=bool)
for event in ssa_events:
    ssa_rpo[event.rpo_index, event.start_timestep : event.end_timestep] = True


# %%
# Per derivative count: PHA cell grid plus union mask, for one sweep.
def _load_sweep(
    paths: list[Path],
) -> tuple[list[NDArray[np.bool_]], list[NDArray[np.bool_]]]:
    masks: list[NDArray[np.bool_]] = []
    rpo_masks: list[NDArray[np.bool_]] = []
    for path in paths:
        _, pha_trajectory, events = load_results(path)
        assert_same_trajectory(trajectory, pha_trajectory)
        masks.append(events_to_union_mask(events, num_timesteps))
        rpo = np.zeros((num_rpos, num_timesteps), dtype=bool)
        for event in events:
            rpo[event.rpo_index, event.start_timestep : event.end_timestep] = True
        rpo_masks.append(rpo)
    return masks, rpo_masks


raw_masks, raw_rpo_masks = _load_sweep(RAW_PATHS)
rescaled_masks, rescaled_rpo_masks = _load_sweep(RESCALED_PATHS)


# %%
# Agreement of each PHA cell grid with the SSA grid (SSA as reference).
def _agreement(pha_rpo_mask: NDArray[np.bool_]) -> tuple[float, float, float]:
    true_positives = float((ssa_rpo & pha_rpo_mask).sum())
    false_positives = float((~ssa_rpo & pha_rpo_mask).sum())
    false_negatives = float((ssa_rpo & ~pha_rpo_mask).sum())
    return (
        true_positives / (true_positives + false_positives),
        true_positives / (true_positives + false_negatives),
        2 * true_positives / (2 * true_positives + false_positives + false_negatives),
    )


# %%
# Attribution on a fixed common shared-set: the SSA timesteps that every PHA run
# -- all six raw and all six rescaled -- also covers. Holding one population
# across both sweeps scores the two curves against the same timesteps.
common = ssa_mask.copy()
for pha_mask in raw_masks + rescaled_masks:
    common &= pha_mask


def _attribution(rpo: NDArray[np.bool_]) -> float:
    match = (rpo & ssa_rpo).any(axis=0) & common
    return float(match.sum()) / float(common.sum())


raw_agreement = np.array([_agreement(m) for m in raw_rpo_masks])
rescaled_agreement = np.array([_agreement(m) for m in rescaled_rpo_masks])
raw_precision, raw_recall, raw_f1 = raw_agreement.T
rescaled_precision, rescaled_recall, rescaled_f1 = rescaled_agreement.T

raw_attribution = np.array([_attribution(rpo) for rpo in raw_rpo_masks])
rescaled_attribution = np.array([_attribution(rpo) for rpo in rescaled_rpo_masks])

# %%
# Render: agreement curves (a), their rescaled-minus-raw difference (b), and
# correct-RPO attribution (c). The difference panel carries the size of the
# rescaling effect, which the near-coincident curves in (a) cannot show.
lambdas = list(LAMBDAS)
figure, axes = plt.subplot_mosaic(
    [["agreement"], ["difference"], ["attribution"]],
    figsize=(3.4, 6.6),
    height_ratios=(3, 2, 3),
    sharex=True,
)

metrics = {
    "Precision": (raw_precision, rescaled_precision),
    "F1": (raw_f1, rescaled_f1),
    "Recall": (raw_recall, rescaled_recall),
}
for name, (raw_values, rescaled_values) in metrics.items():
    style = METRIC_STYLES[name]
    axes["agreement"].plot(lambdas, raw_values, **style)
    axes["agreement"].plot(lambdas, rescaled_values, **style, **RESCALED_STYLE)
    axes["difference"].plot(lambdas, rescaled_values - raw_values, **style)
axes["difference"].axhline(0, color="0.6", linewidth=0.6, zorder=0)

axes["attribution"].plot(lambdas, raw_attribution, color="black", marker="o")
axes["attribution"].plot(lambdas, rescaled_attribution, color="black", **RESCALED_STYLE, marker="o")

axes["agreement"].set_title("(a)", loc="left")
axes["agreement"].set_ylabel("Agreement with SSA grid")
axes["difference"].set_title("(b)", loc="left")
axes["difference"].set_ylabel("Rescaled $-$ raw")
axes["attribution"].set_title("(c)", loc="left")
axes["attribution"].set_ylabel("Per-RPO attribution vs. SSA")
axes["attribution"].set_xlabel(r"Derivative orders $\lambda$")
axes["attribution"].set_xticks(lambdas)

metric_handles = [Line2D([], [], label=name, **style) for name, style in METRIC_STYLES.items()]
sweep_handles = [
    Line2D([], [], color="black", linestyle="-", label="Raw"),
    Line2D([], [], color="black", linestyle="--", label="Rescaled"),
]
figure.legend(handles=metric_handles + sweep_handles, loc="outside upper center", ncols=3)

plt.show()
