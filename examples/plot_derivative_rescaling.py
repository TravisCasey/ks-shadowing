r"""
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
correctness: the SSA detector is the yardstick, not ground truth. Prior
analysis found that rescaling the orders does not beat simply capping the sweep
at two or three derivative orders. It slows the high-order erosion the
:ref:`saturation example <sphx_glr_auto_examples_plot_derivative_saturation.py>`
shows, without removing it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ks_shadowing import events_to_union_mask, load_results

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
RAW_PATHS = [DATA_DIR / f"pha_r2048_d1_o{k}.h5" for k in range(6)]
RESCALED_PATHS = [DATA_DIR / f"pha_r2048_d1_o{k}_rescaled.h5" for k in range(6)]
MAX_ORDERS = range(6)

# Raw sweep: dark shades, one per metric. Rescaled sweep: lighter shades of the
# same three, so a sweep is identifiable by "how dark" independent of marker.
RAW_PRECISION_COLOR = "black"
RAW_F1_COLOR = "0.35"
RAW_RECALL_COLOR = "0.55"
RESCALED_PRECISION_COLOR = "0.6"
RESCALED_F1_COLOR = "0.75"
RESCALED_RECALL_COLOR = "0.85"

# %%
# SSA reference: union mask plus per-RPO masks over the full trajectory.
_, trajectory, ssa_events = load_results(SSA_PATH)
num_timesteps = trajectory.num_timesteps
num_rpos = max(event.rpo_index for event in ssa_events) + 1
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
        _, _, events = load_results(path)
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
# Render: agreement (left), correct-RPO attribution (right).
orders = list(MAX_ORDERS)
figure, (ax_agreement, ax_attribution) = plt.subplots(1, 2, figsize=(13, 5))

_LINE_KWARGS = {"linewidth": 1.3, "markersize": 5}
ax_agreement.plot(orders, raw_precision, color=RAW_PRECISION_COLOR, marker="o", **_LINE_KWARGS)
ax_agreement.plot(orders, raw_f1, color=RAW_F1_COLOR, marker="s", **_LINE_KWARGS)
ax_agreement.plot(
    orders, raw_recall, color=RAW_RECALL_COLOR, marker="^", linestyle="--", **_LINE_KWARGS
)
ax_agreement.plot(
    orders, rescaled_precision, color=RESCALED_PRECISION_COLOR, marker="o", **_LINE_KWARGS
)
ax_agreement.plot(orders, rescaled_f1, color=RESCALED_F1_COLOR, marker="s", **_LINE_KWARGS)
ax_agreement.plot(
    orders,
    rescaled_recall,
    color=RESCALED_RECALL_COLOR,
    marker="^",
    linestyle="--",
    **_LINE_KWARGS,
)
ax_agreement.set_xlabel("Max derivative order")
ax_agreement.set_ylabel("Agreement with SSA mask")
ax_agreement.set_title("Precision, F1, and recall track closely across sweeps")
ax_agreement.set_xticks(orders)
agreement_values = np.concatenate(
    [raw_precision, raw_f1, raw_recall, rescaled_precision, rescaled_f1, rescaled_recall]
)
agreement_margin = 0.03 * (agreement_values.max() - agreement_values.min())
ax_agreement.set_ylim(
    agreement_values.min() - agreement_margin, agreement_values.max() + agreement_margin
)
metric_handles = [
    Line2D([], [], color="0.3", marker="o", linestyle="-", label="Precision"),
    Line2D([], [], color="0.3", marker="s", linestyle="-", label="F1"),
    Line2D([], [], color="0.3", marker="^", linestyle="--", label="Recall"),
]
sweep_handles = [
    Line2D([], [], color="black", linestyle="-", label="Raw"),
    Line2D([], [], color="0.7", linestyle="-", label="Rescaled"),
]
metric_legend = ax_agreement.legend(
    handles=metric_handles, loc="lower left", fontsize="x-small", title="Metric"
)
ax_agreement.add_artist(metric_legend)
ax_agreement.legend(handles=sweep_handles, loc="lower right", fontsize="x-small", title="Sweep")

ax_attribution.plot(orders, raw_attribution, color="black", marker="o", label="Raw")
ax_attribution.plot(orders, rescaled_attribution, color="0.45", marker="s", label="Rescaled")
ax_attribution.set_xlabel("Max derivative order")
ax_attribution.set_ylabel("Per-RPO attribution vs. SSA")
ax_attribution.set_title("Rescaling slows the high-order decline")
ax_attribution.set_xticks(orders)
ax_attribution.legend(fontsize="small")

plt.tight_layout()
plt.show()
