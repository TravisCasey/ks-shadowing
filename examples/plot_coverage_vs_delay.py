"""
Coverage agreement vs. PHA delay
================================

At each trajectory timestep, each detection method reports a binary shadowing
flag (union across RPOs). ``F_agree`` is the fraction of timesteps where SSA and
PHA report the same flag; ``F_disagree`` splits into ``F_ssa_only`` (SSA
shadowing, PHA not) and ``F_pha_only`` (the reverse). All three sum to 1.
Plotted against PHA ``delay``, with one curve per ``max_derivative_order`` setting.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
PHA_PATTERN = re.compile(r"^pha_r2048_d(\d+)_o(\d+)\.h5$")

# %%
# Build the SSA coverage mask once.
_, ssa_trajectory, ssa_events = load_results(SSA_PATH)
ssa_mask = events_to_union_mask(ssa_events, ssa_trajectory.num_timesteps)

# %%
# For every PHA fixture, compute F_agree / F_ssa_only / F_pha_only and group by max order.
by_max_order: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
for pha_path in sorted(DATA_DIR.glob("pha_r2048_d*_o*.h5")):
    match = PHA_PATTERN.match(pha_path.name)
    if match is None:
        continue
    delay = int(match.group(1))
    max_order = int(match.group(2))

    _, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)

    pha_mask = events_to_union_mask(pha_events, ssa_trajectory.num_timesteps)
    f_agree = float((ssa_mask == pha_mask).mean())
    f_ssa_only = float((ssa_mask & ~pha_mask).mean())
    f_pha_only = float((~ssa_mask & pha_mask).mean())
    by_max_order[max_order].append((delay, f_agree, f_ssa_only, f_pha_only))

# %%
# Render.
figure, (ax_agree, ax_disagree) = plt.subplots(1, 2, figsize=(13, 5))

for max_order in sorted(by_max_order):
    rows = sorted(by_max_order[max_order])
    delays = np.array([r[0] for r in rows])
    f_agree = np.array([r[1] for r in rows])
    f_ssa_only = np.array([r[2] for r in rows])
    f_pha_only = np.array([r[3] for r in rows])

    color = f"C{max_order}"
    marker = Line2D.filled_markers[max_order % len(Line2D.filled_markers)]
    label = f"max order {max_order}"
    ax_agree.plot(delays, f_agree, marker=marker, color=color, label=label)
    ax_disagree.plot(
        delays,
        f_ssa_only,
        marker=marker,
        color=color,
        linestyle="--",
        label=f"SSA only, {label}",
    )
    ax_disagree.plot(
        delays,
        f_pha_only,
        marker=marker,
        color=color,
        linestyle=":",
        label=f"PHA only, {label}",
    )

ax_agree.set_xlabel("PHA delay")
ax_agree.set_ylabel(r"$F_{\mathrm{agree}}$")
ax_agree.set_title("Agreement")
ax_agree.legend(loc="upper right")

ax_disagree.set_xlabel("PHA delay")
ax_disagree.set_ylabel(r"$F_{\mathrm{disagree}}$")
ax_disagree.set_title("Disagreement")
ax_disagree.legend(fontsize="small")

plt.tight_layout()
plt.show()
