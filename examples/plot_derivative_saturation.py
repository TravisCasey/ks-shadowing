r"""
Derivative embedding saturates against the SSA reference
========================================================

Running PHA over :math:`\lambda` derivative orders averages the
persistence-diagram Wasserstein distances over spatial-derivative orders
:math:`0` to :math:`\lambda - 1` before reducing over phases, so
:math:`\lambda = 1` uses the field alone. In the code this is
``max_derivative_order``, one less than :math:`\lambda`. The
``pha_r2048_d1_o{0..5}`` files are that sweep, :math:`\lambda = 1` through
:math:`6`, and scoring them against SSA shows directly what each added order
contributes: not much past :math:`\lambda = 2`, and eventually some harm.

Panel (a) scores the (RPO, timestep) cell grid rather than the shadowing flag,
so a run that flags the right timestep against the wrong orbit is penalized for
it; the :ref:`agreement example
<sphx_glr_auto_examples_plot_coverage_vs_embedding.py>` defines the measure.
Precision falls monotonically with every added order, as the extra orders flag
cells SSA does not. Recall climbs steeply from :math:`\lambda = 1` to
:math:`\lambda = 2`, then declines with every further order, so F1 peaks at
:math:`\lambda = 2` as well.

Panel (b) isolates why: restricted to the timesteps every run agrees are
shadowing, attribution to the correct RPO peaks at :math:`\lambda = 2` and
erodes afterward. This is the detection-level signature of high-order
redundancy: at high :math:`\lambda` the added orders increasingly measure the
same thing, so they reinforce one another rather than contributing independent
evidence. Extra orders begin to indicate shadowing against the wrong orbit. The
companion :ref:`mechanism example
<sphx_glr_auto_examples_plot_derivative_mechanism.py>` shows the scale inflation
and the redundancy directly.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import assert_same_trajectory, events_to_union_mask, load_results, load_rpos

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
PHA_PATHS = [DATA_DIR / f"pha_r2048_d1_o{order}.h5" for order in range(6)]
LAMBDAS = range(1, 7)

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# One fixed color and marker per agreement metric, shared with the rescaling
# figure (Tol bright palette).
METRIC_STYLES: dict[str, dict[str, Any]] = {
    "Precision": {"color": "#4477AA", "marker": "o"},
    "F1": {"color": "#EE6677", "marker": "s"},
    "Recall": {"color": "#CCBB44", "marker": "^"},
}

# %%
# SSA reference: per-RPO cell grid, plus the union mask the fixed common
# set in panel (b) is built from.
metadata, trajectory, ssa_events = load_results(SSA_PATH)
num_timesteps = trajectory.num_timesteps
num_rpos = len(load_rpos(REPO_ROOT / metadata.rpo_file))
ssa_mask = events_to_union_mask(ssa_events, num_timesteps)
ssa_rpo = np.zeros((num_rpos, num_timesteps), dtype=bool)
for event in ssa_events:
    ssa_rpo[event.rpo_index, event.start_timestep : event.end_timestep] = True

# %%
# Per derivative count: PHA cell grid plus union mask, built the same way.
pha_masks: list[NDArray[np.bool_]] = []
pha_rpo_masks: list[NDArray[np.bool_]] = []
for path in PHA_PATHS:
    _, pha_trajectory, events = load_results(path)
    assert_same_trajectory(trajectory, pha_trajectory)
    pha_masks.append(events_to_union_mask(events, num_timesteps))
    rpo = np.zeros((num_rpos, num_timesteps), dtype=bool)
    for event in events:
        rpo[event.rpo_index, event.start_timestep : event.end_timestep] = True
    pha_rpo_masks.append(rpo)


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
# also covers. Holding the set fixed across k removes the shifting-subset
# confound (a raw per-run overlap would compare different timestep populations).
common = ssa_mask.copy()
for pha_mask in pha_masks:
    common &= pha_mask


def _attribution(rpo: NDArray[np.bool_]) -> float:
    match = (rpo & ssa_rpo).any(axis=0) & common
    return float(match.sum()) / float(common.sum())


agreement = np.array([_agreement(rpo_mask) for rpo_mask in pha_rpo_masks])
precision, recall, f1 = agreement.T
attribution = np.array([_attribution(rpo) for rpo in pha_rpo_masks])

# %%
# Render: the effect (a) and its consequence (b). Both panels use data-tight
# vertical limits; the agreement axis does not start at zero.
lambdas = list(LAMBDAS)
figure, (ax_agreement, ax_attribution) = plt.subplots(2, 1, figsize=(3.4, 4.6))

for name, values in (("Precision", precision), ("F1", f1), ("Recall", recall)):
    ax_agreement.plot(lambdas, values, label=name, **METRIC_STYLES[name])
ax_agreement.set_title("(a)", loc="left")
ax_agreement.set_xlabel(r"Derivative orders $\lambda$")
ax_agreement.set_ylabel("Agreement with SSA grid")
ax_agreement.set_xticks(lambdas)
ax_agreement.legend(loc="lower right")

ax_attribution.plot(lambdas, attribution, color="black", marker="o")
ax_attribution.set_title("(b)", loc="left")
ax_attribution.set_xlabel(r"Derivative orders $\lambda$")
ax_attribution.set_ylabel("Per-RPO attribution vs. SSA")
ax_attribution.set_xticks(lambdas)

plt.show()
