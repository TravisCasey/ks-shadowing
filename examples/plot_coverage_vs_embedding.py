"""
Coverage agreement along each embedding axis
============================================

At each trajectory timestep, each detection method reports a binary shadowing
flag (union across RPOs). ``F_agree`` is the fraction of timesteps where SSA
and PHA report the same flag; ``F_disagree`` splits into ``F_ssa_only`` (SSA
shadowing, PHA not) and ``F_pha_only`` (the reverse). All three sum to 1.

The two embedding axes are presented independently: delays greater than 1 are
used only at max order 0. The left column sweeps ``delay`` at max order 0;
the right column sweeps ``max_derivative_order`` at delay 1. The top row is
``F_agree``; the bottom row decomposes the disagreement into its two channels
per axis. The delay-axis crossover in panel (c) motivates the ``delay = 8``
operating point; the derivative axis additionally gets the richer
precision/recall treatment in the
:ref:`saturation example <sphx_glr_auto_examples_plot_derivative_saturation.py>`.
"""

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import (
    assert_same_trajectory,
    events_to_union_mask,
    load_results,
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
# The F_agree curves use open markers so the top row reads differently from
# the solid black filled-marker "SSA only" channel in the disagreement row
# below (the line itself stays solid; dash patterns get messy between markers).
AGREE_STYLE: dict[str, Any] = {
    "color": "black",
    "marker": "o",
    "markerfacecolor": "none",
    "markeredgewidth": 0.8,
}
CHANNEL_STYLES: dict[str, dict[str, Any]] = {
    "SSA only": {"color": "black", "marker": "o"},
    "PHA only": {
        "color": "0.55",
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 0.8,
    },
}

# %%
# Build the SSA coverage mask once.
_, ssa_trajectory, ssa_events = load_results(SSA_PATH)
ssa_mask = events_to_union_mask(ssa_events, ssa_trajectory.num_timesteps)


def _coverage_fractions(pha_path: Path) -> tuple[float, float, float]:
    """Return (F_agree, F_ssa_only, F_pha_only) for one PHA result file."""
    _, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    pha_mask = events_to_union_mask(pha_events, ssa_trajectory.num_timesteps)
    f_agree = float((ssa_mask == pha_mask).mean())
    f_ssa_only = float((ssa_mask & ~pha_mask).mean())
    f_pha_only = float((~ssa_mask & pha_mask).mean())
    return f_agree, f_ssa_only, f_pha_only


# %%
# Delay axis: every max-order-0 fixture, one point per delay.
delay_rows: list[tuple[int, float, float, float]] = []
for pha_path in DATA_DIR.glob("pha_r2048_d*_o0.h5"):
    match = DELAY_PATTERN.match(pha_path.name)
    if match is None:
        continue
    f_agree, f_ssa_only, f_pha_only = _coverage_fractions(pha_path)
    delay_rows.append((int(match.group(1)), f_agree, f_ssa_only, f_pha_only))
delay_rows.sort()
delays = np.array([row[0] for row in delay_rows])
delay_fractions: NDArray[np.float64] = np.array([row[1:] for row in delay_rows])

# %%
# Derivative axis: the delay-1 sweep, one point per max order.
orders = list(range(len(ORDER_PATHS)))
order_fractions: NDArray[np.float64] = np.array([_coverage_fractions(path) for path in ORDER_PATHS])

# %%
# Render: one column per embedding axis, F_agree on top and the two
# disagreement channels below, each row on a shared scale.
figure, axes = plt.subplot_mosaic(
    [["delay", "order"], ["disagree_delay", "disagree_order"]],
    figsize=(3.4, 4.4),
)
axes["order"].sharey(axes["delay"])
axes["disagree_delay"].sharex(axes["delay"])
axes["disagree_order"].sharex(axes["order"])
axes["disagree_order"].sharey(axes["disagree_delay"])

axes["delay"].plot(delays, delay_fractions[:, 0], **AGREE_STYLE)
axes["delay"].set_title("(a)", loc="left")
axes["delay"].set_title("max order 0")
axes["delay"].set_ylabel(r"$F_{\mathrm{agree}}$")
axes["delay"].tick_params(labelbottom=False)

axes["order"].plot(orders, order_fractions[:, 0], **AGREE_STYLE)
axes["order"].set_title("(b)", loc="left")
axes["order"].set_title("delay 1")
axes["order"].set_xticks(orders)
axes["order"].tick_params(labelleft=False, labelbottom=False)

for panel, x_values, fractions in (
    ("disagree_delay", delays, delay_fractions),
    ("disagree_order", orders, order_fractions),
):
    for channel, column in (("SSA only", 1), ("PHA only", 2)):
        axes[panel].plot(x_values, fractions[:, column], label=channel, **CHANNEL_STYLES[channel])

axes["disagree_delay"].set_title("(c)", loc="left")
axes["disagree_delay"].set_ylabel(r"$F_{\mathrm{disagree}}$")
axes["disagree_delay"].set_xlabel("PHA delay")
axes["disagree_delay"].legend()
axes["disagree_order"].set_title("(d)", loc="left")
axes["disagree_order"].set_xlabel("Max derivative order")
axes["disagree_order"].tick_params(labelleft=False)

plt.show()
