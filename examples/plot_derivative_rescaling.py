"""
Per-order rescaling of the derivative sweep
===========================================

The ``rescale_orders`` option divides each spatial-derivative order's
Wasserstein column by a per-order median scale before the orders are averaged
together. The ``pha_r2048_d1_o{0..5}_rescaled.h5`` files repeat the
``max_derivative_order`` 0 through 5 sweep with that option on; the
``pha_r2048_d1_o{0..5}.h5`` files are the raw sweep with it off.

Scored against the SSA mask, the two sweeps track each other closely. At
``k = 0`` they are identical, since a single order has nothing to rescale
against. Through ``k = 2`` the rescaled precision, F1, and correct-RPO
attribution differ from the raw curves only in the third decimal. The sweeps
separate only at high order: from ``k = 3`` onward rescaling makes the decline
gentler -- at ``k = 5`` precision holds near 0.75 against the raw 0.73, F1 near
0.70 against 0.66, and attribution near 0.97 against 0.96 -- but it does not
lift the peak, which both sweeps reach around ``k = 2``-``3``.

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
RAW_PATHS = [DATA_DIR / f"pha_r2048_d1_o{k}.h5" for k in range(6)]
RESCALED_PATHS = [DATA_DIR / f"pha_r2048_d1_o{k}_rescaled.h5" for k in range(6)]
MAX_ORDERS = range(6)

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
# SSA reference: union mask plus per-RPO masks over the full trajectory.
metadata, trajectory, ssa_events = load_results(SSA_PATH)
num_timesteps = trajectory.num_timesteps
num_rpos = len(load_rpos(REPO_ROOT / metadata.rpo_file))
ssa_mask = events_to_union_mask(ssa_events, num_timesteps)
ssa_rpo = np.zeros((num_rpos, num_timesteps), dtype=bool)
for event in ssa_events:
    ssa_rpo[event.rpo_index, event.start_timestep : event.end_timestep] = True


# %%
# Per derivative count: PHA union mask plus per-RPO masks, for one sweep.
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
# Agreement of each PHA union mask with the SSA mask (SSA as reference).
def _agreement(pha_mask: NDArray[np.bool_]) -> tuple[float, float, float]:
    intersection = float((pha_mask & ssa_mask).sum())
    precision = intersection / pha_mask.sum()
    recall = intersection / ssa_mask.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


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


raw_agreement = np.array([_agreement(m) for m in raw_masks])
rescaled_agreement = np.array([_agreement(m) for m in rescaled_masks])
raw_precision, raw_recall, raw_f1 = raw_agreement.T
rescaled_precision, rescaled_recall, rescaled_f1 = rescaled_agreement.T

raw_attribution = np.array([_attribution(rpo) for rpo in raw_rpo_masks])
rescaled_attribution = np.array([_attribution(rpo) for rpo in rescaled_rpo_masks])

# %%
# Render: agreement curves (a), their rescaled-minus-raw difference (b), and
# correct-RPO attribution (c). The difference panel carries the size of the
# rescaling effect, which the near-coincident curves in (a) cannot show.
orders = list(MAX_ORDERS)
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
    axes["agreement"].plot(orders, raw_values, **style)
    axes["agreement"].plot(orders, rescaled_values, **style, **RESCALED_STYLE)
    axes["difference"].plot(orders, rescaled_values - raw_values, **style)
axes["difference"].axhline(0, color="0.6", linewidth=0.6, zorder=0)

axes["attribution"].plot(orders, raw_attribution, color="black", marker="o")
axes["attribution"].plot(orders, rescaled_attribution, color="black", **RESCALED_STYLE, marker="o")

axes["agreement"].set_title("(a)", loc="left")
axes["agreement"].set_ylabel("Agreement with SSA mask")
axes["difference"].set_title("(b)", loc="left")
axes["difference"].set_ylabel("Rescaled $-$ raw")
axes["attribution"].set_title("(c)", loc="left")
axes["attribution"].set_ylabel("Per-RPO attribution vs. SSA")
axes["attribution"].set_xlabel("Max derivative order")
axes["attribution"].set_xticks(orders)

metric_handles = [Line2D([], [], label=name, **style) for name, style in METRIC_STYLES.items()]
sweep_handles = [
    Line2D([], [], color="black", linestyle="-", label="Raw"),
    Line2D([], [], color="black", linestyle="--", label="Rescaled"),
]
figure.legend(handles=metric_handles + sweep_handles, loc="outside upper center", ncols=3)

plt.show()
