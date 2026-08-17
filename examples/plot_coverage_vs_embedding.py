r"""
Detection agreement along each embedding axis
=============================================

Both methods label every (RPO, timestep) cell of a ``(num_rpos, num_timesteps)``
grid: cell ``(r, i)`` is set when that method reports an event against RPO ``r``
covering timestep ``i``. Comparing the two grids cell by cell gives ``TP``
(cells both methods set), ``FP`` (only PHA) and ``FN`` (only SSA), from which

.. math::

   \mathrm{precision} = \frac{TP}{TP + FP}, \qquad
   \mathrm{recall} = \frac{TP}{TP + FN}, \qquad
   F_1 = \frac{2\,TP}{2\,TP + FP + FN}.

Counts are pooled across all RPOs before the ratio is taken, so that each orbit
contributes in proportion to how often it is active.

The two embedding axes are presented independently: delays greater than 1 are
used only at max order 0. The left column sweeps ``delay`` at max order 0; the
right column sweeps ``max_derivative_order`` at delay 1. The top row is
:math:`F_1`; the bottom row is the precision and recall it is built from.
Along the delay axis, precision falls from its ``delay = 1`` value to a minimum
around ``delay`` 9 to 11, partially recovers, then falls again toward the end of
the sweep. Recall rises steeply from ``delay = 1`` and saturates. :math:`F_1`
inherits recall's early rise, then flattens into a broad maximum spanning
roughly ``delay`` 21 to 27 before declining toward ``delay = 33``. Along the
derivative axis, :math:`F_1` has a sharp maximum at
``max_derivative_order = 2``. The rest of the gallery uses ``delay = 25`` as its
delay-axis setting.
"""

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import (
    ShadowingEvent,
    assert_same_trajectory,
    load_results,
    load_rpos,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
DELAY_PATTERN = re.compile(r"^pha_r2048_d(\d+)_o0\.h5$")
ORDER_PATHS = [DATA_DIR / f"pha_r2048_d1_o{order}.h5" for order in range(6)]

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# One fixed color and marker per agreement metric, shared with the saturation
# and rescaling figures (Tol bright palette).
METRIC_STYLES: dict[str, dict[str, Any]] = {
    "Precision": {"color": "#4477AA", "marker": "o"},
    "F1": {"color": "#EE6677", "marker": "s"},
    "Recall": {"color": "#CCBB44", "marker": "^"},
}

# %%
# Build the SSA reference grid once.
ssa_metadata, ssa_trajectory, ssa_events = load_results(SSA_PATH)
num_timesteps = ssa_trajectory.num_timesteps
num_rpos = len(load_rpos(REPO_ROOT / ssa_metadata.rpo_file))


def _per_rpo_mask(events: list[ShadowingEvent]) -> NDArray[np.bool_]:
    """Return the ``(num_rpos, num_timesteps)`` coverage grid of ``events``."""
    mask = np.zeros((num_rpos, num_timesteps), dtype=bool)
    for event in events:
        mask[event.rpo_index, event.start_timestep : event.end_timestep] = True
    return mask


ssa_mask = _per_rpo_mask(ssa_events)


def _scores(pha_path: Path) -> tuple[float, float, float]:
    """Return (precision, recall, F1) for one PHA result file."""
    _, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    pha_mask = _per_rpo_mask(pha_events)

    true_positives = float((ssa_mask & pha_mask).sum())
    false_positives = float((~ssa_mask & pha_mask).sum())
    false_negatives = float((ssa_mask & ~pha_mask).sum())
    return (
        true_positives / (true_positives + false_positives),
        true_positives / (true_positives + false_negatives),
        2 * true_positives / (2 * true_positives + false_positives + false_negatives),
    )


# %%
# Delay axis: every max-order-0 fixture, one point per delay.
delay_rows: list[tuple[int, float, float, float]] = []
for pha_path in DATA_DIR.glob("pha_r2048_d*_o0.h5"):
    match = DELAY_PATTERN.match(pha_path.name)
    if match is None:
        continue
    delay_rows.append((int(match.group(1)), *_scores(pha_path)))
delay_rows.sort()
delays = np.array([row[0] for row in delay_rows])
delay_scores: NDArray[np.float64] = np.array([row[1:] for row in delay_rows])

# %%
# Derivative axis: the w = 1 sweep, one point per lambda. ORDER_PATHS is indexed
# by max_derivative_order, so lambda is that index plus one.
lambdas = [order + 1 for order in range(len(ORDER_PATHS))]
order_scores: NDArray[np.float64] = np.array([_scores(path) for path in ORDER_PATHS])

# %%
# Render: one column per embedding axis, F1 on top and the precision and
# recall it is built from below, each row on a shared scale.
figure, axes = plt.subplot_mosaic(
    [["delay", "order"], ["parts_delay", "parts_order"]],
    figsize=(3.4, 4),
)
axes["order"].sharey(axes["delay"])
axes["parts_delay"].sharex(axes["delay"])
axes["parts_order"].sharex(axes["order"])
axes["parts_order"].sharey(axes["parts_delay"])

axes["delay"].plot(delays, delay_scores[:, 2], **METRIC_STYLES["F1"])
axes["delay"].set_title("(a)", loc="left")
axes["delay"].set_ylabel(r"$F_1$")
axes["delay"].tick_params(labelbottom=False)

axes["order"].plot(lambdas, order_scores[:, 2], **METRIC_STYLES["F1"])
axes["order"].set_title("(b)", loc="left")
axes["order"].set_xticks(lambdas)
axes["order"].tick_params(labelleft=False, labelbottom=False)

for panel, x_values, scores in (
    ("parts_delay", delays, delay_scores),
    ("parts_order", lambdas, order_scores),
):
    for metric, column in (("Precision", 0), ("Recall", 1)):
        axes[panel].plot(x_values, scores[:, column], label=metric, **METRIC_STYLES[metric])

axes["parts_delay"].set_title("(c)", loc="left")
axes["parts_delay"].set_ylabel("Precision, recall")
axes["parts_delay"].set_xlabel(r"Delay window $w$")
axes["parts_delay"].set_xticks((1, 9, 17, 25, 33))
axes["parts_delay"].legend()
axes["parts_order"].set_title("(d)", loc="left")
axes["parts_order"].set_xlabel(r"Derivative orders $\lambda$")
axes["parts_order"].tick_params(labelleft=False)

plt.show()
