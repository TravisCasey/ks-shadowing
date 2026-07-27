"""
Derivative embedding saturates against the SSA reference
========================================================

Running PHA with ``max_derivative_order = k`` averages the persistence-diagram
Wasserstein distances over spatial-derivative orders ``0..k`` before reducing
over phases. The ``pha_r2048_d1_o{0..5}`` files are that sweep,
``max_derivative_order`` 0 through 5, and scoring them against the SSA mask
shows directly what each added order buys us. Short version: not much past max
order 2, and eventually some harm.

Precision (panel a) is the line to trust. A fixed quantile (typically 0.4)
of all timesteps is flagged as shadowing some RPO, but longest-path selection
and the minimum event duration mean fewer than that fraction end up covered by
events. Precision -- of the timesteps PHA flags, the fraction SSA also flags --
normalizes against PHA's own coverage, so it is unaffected by that disparity. It
falls monotonically with every added order: the extra orders detect shadowing
where SSA sees nothing.

Recall, and therefore the F1 score, is less clear, as recall is normalized by
the fixed SSA coverage instead. An increase in coverage, which does occur in
the first few orders, inflates this measure.

Panel (b) is the more telling consequence. Restricted to the timesteps
every run agrees are shadowing, attribution to the correct RPO peaks at
``k = 2`` and erodes afterward. This is the detection-level signature of
high-order redundancy: at high ``k`` the added orders increasingly measure the
same thing, so they reinforce one another rather than contributing independent
evidence. Extra orders begin to indicate shadowing against the wrong orbit. The
companion :ref:`mechanism example
<sphx_glr_auto_examples_plot_derivative_mechanism.py>` shows that scale inflation
and redundancy directly.
"""

from pathlib import Path
from typing import Any

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
PHA_PATHS = [DATA_DIR / f"pha_r2048_d1_o{k}.h5" for k in range(6)]
MAX_ORDERS = range(6)

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# One fixed color and marker per agreement metric, shared with the rescaling
# figure (Tol bright palette).
METRIC_STYLES: dict[str, dict[str, Any]] = {
    "Precision": {"color": "#4477AA", "marker": "o"},
    "F1": {"color": "#EE6677", "marker": "s"},
    "Recall": {"color": "#CCBB44", "marker": "^"},
}

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
def _agreement(pha_mask: NDArray[np.bool_]) -> tuple[float, float, float]:
    intersection = float((pha_mask & ssa_mask).sum())
    precision = intersection / pha_mask.sum()
    recall = intersection / ssa_mask.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


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
precision, recall, f1 = agreement.T
attribution = np.array([_attribution(rpo) for rpo in pha_rpo_masks])

# %%
# Render: the effect (a) and its consequence (b). Both panels use data-tight
# vertical limits; the agreement axis does not start at zero.
orders = list(MAX_ORDERS)
figure, (ax_agreement, ax_attribution) = plt.subplots(2, 1, figsize=(3.4, 4.6))

for name, values in (("Precision", precision), ("F1", f1), ("Recall", recall)):
    ax_agreement.plot(orders, values, label=name, **METRIC_STYLES[name])
ax_agreement.set_title("(a)", loc="left")
ax_agreement.set_xlabel("Max derivative order")
ax_agreement.set_ylabel("Agreement with SSA mask")
ax_agreement.set_xticks(orders)
ax_agreement.legend(loc="lower right")

ax_attribution.plot(orders, attribution, color="black", marker="o")
ax_attribution.set_title("(b)", loc="left")
ax_attribution.set_xlabel("Max derivative order")
ax_attribution.set_ylabel("Per-RPO attribution vs. SSA")
ax_attribution.set_xticks(orders)

plt.show()
