r"""
Derivative embedding saturates against the SSA reference
========================================================

Running PHA with ``derivatives = k`` averages the persistence-diagram
Wasserstein distances over spatial-derivative orders ``0..k-1`` before reducing
over phases. The ``pha_r2048_d1_o{1..6}`` files are that sweep from one to six
orders, and scoring them against the SSA mask shows directly what each added
order buys us. Short version: not much past 2 or 3 orders, and eventually a
some harm.

Precision (left, black) is the line to trust. A fixed quantile (typically 0.4)
of all timesteps are considered shadowing some RPO; however, due to longest
pathfining and minimum event duration, less than that fraction of timesteps is
covered by events. However, precision, the number of timesteps correctly
(against SSA) predicted as shadowing normalized by the PHA coverage, accounts
for this coverage disparity. It falls monotonically with every added order, and
indicates extra orders falsely detecting where SSA sees nothing.

Recall, and therefore F1 score, are less clear as it is normalized by the fixed
SSA coverage instead. An increase in coverage, which does occur in the first
few orders, inflates this measure.

The right panel is the more telling consequence. Restricted to the timesteps
every run agrees are shadowing, attribution to the correct RPO peaks at
``k = 3`` and erodes afterward. This is the detection-level shadow of high-order
redundancy: at high ``k`` the added orders increasingly measure the same thing,
so they reinforce one another rather than contributing independent evidence.
Extra orders begin to indicate shadowing against the wrong orbit. The companion
:ref:`sphx_glr_auto_examples_plot_derivative_mechanism.py` shows that scale
inflation and redundancy directly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import events_to_union_mask, load_results

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
PHA_PATHS = [DATA_DIR / f"pha_r2048_d1_o{k}.h5" for k in range(1, 7)]
DERIVATIVE_COUNTS = range(1, 7)

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
# Per derivative count: PHA union mask plus per-RPO masks, built the same way.
pha_masks: list[NDArray[np.bool_]] = []
pha_rpo_masks: list[NDArray[np.bool_]] = []
for path in PHA_PATHS:
    _, _, events = load_results(path)
    pha_masks.append(events_to_union_mask(events, num_timesteps))
    rpo = np.zeros((num_rpos, num_timesteps), dtype=bool)
    for event in events:
        rpo[event.rpo_index, event.start_timestep : event.end_timestep] = True
    pha_rpo_masks.append(rpo)


# %%
# Agreement of each PHA union mask with the SSA mask (SSA as reference).
def _agreement(pha_mask: NDArray[np.bool_]) -> tuple[float, float, float, float, float]:
    intersection = float((pha_mask & ssa_mask).sum())
    precision = intersection / pha_mask.sum()
    recall = intersection / ssa_mask.sum()
    f1 = 2 * precision * recall / (precision + recall)
    iou = intersection / (pha_mask | ssa_mask).sum()
    observed = (pha_mask == ssa_mask).mean()
    p, q = pha_mask.mean(), ssa_mask.mean()
    expected = p * q + (1 - p) * (1 - q)
    kappa = (observed - expected) / (1 - expected)
    return precision, recall, f1, kappa, iou


# %%
# Attribution on a fixed common shared-set: the SSA timesteps that every PHA run
# also covers. Holding the set fixed across k removes the shifting-subset
# confound (a raw per-run overlap would compare different timestep populations).
common = ssa_mask.copy()
for pha_mask in pha_masks:
    common &= pha_mask


def _attribution(rpo: NDArray[np.bool_]) -> float:
    match = (rpo & ssa_rpo).any(axis=0) & common
    return float(match.sum()) / float(common.sum())


agreement = np.array([_agreement(pha_mask) for pha_mask in pha_masks])
precision, recall, f1, kappa, iou = agreement.T
attribution = np.array([_attribution(rpo) for rpo in pha_rpo_masks])

# %%
# Render: the effect (left) and its consequence (right).
counts = list(DERIVATIVE_COUNTS)
figure, (ax_agreement, ax_attribution) = plt.subplots(1, 2, figsize=(13, 5))

ax_agreement.plot(counts, precision, color="black", marker="o", label="Precision")
ax_agreement.plot(counts, f1, color="0.45", marker="s", label="F1")
ax_agreement.plot(counts, recall, color="0.7", marker="^", linestyle="--", label="Recall")
ax_agreement.set_xlabel("Derivatives")
ax_agreement.set_ylabel("Agreement with SSA mask")
ax_agreement.set_title("Precision falls as orders are added")
ax_agreement.set_xticks(counts)
ax_agreement.set_ylim(bottom=0)
ax_agreement.legend(fontsize="small")

ax_attribution.plot(counts, attribution, color="black", marker="o")
ax_attribution.set_xlabel("Derivatives")
ax_attribution.set_ylabel("Per-RPO attribution vs SSA")
ax_attribution.set_title("Correct-RPO attribution peaks at 3, then declines")
ax_attribution.set_xticks(counts)

plt.tight_layout()
plt.show()
