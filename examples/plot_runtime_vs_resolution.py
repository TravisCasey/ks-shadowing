"""
Detection runtime vs. spatial resolution
=========================================

Wall-clock detection time as a function of the spatial resolution the
trajectory is loaded at, with one curve for SSA and one per PHA ``derivatives``
setting. SSA evaluates L2 distances in physical space, so its cost grows with
resolution; PHA computes Wasserstein distances between persistence diagrams,
whose cost is dominated by trajectory length rather than grid size, so its
curves stay nearly flat and stack by ``derivatives``. Cost grows superlinearly
in ``derivatives``: the per-order increment is roughly constant up to 3
derivatives and then accelerates.

Fixtures outside resolution 2048 use PHA ``delay = 8``, and the 0-2 derivative
settings carry a full ``delay`` sweep at 2048, overlaid as a vertical cluster
whose small vertical extent shows that ``delay`` has little effect on runtime.
The PHA curve passes through the per-resolution mean. The 3-5 derivative
settings were run only at resolution 2048 and ``delay = 1``, so they appear as
single markers rather than curves.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ks_shadowing import load_results

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATTERN = re.compile(r"^ssa_r(\d+)\.h5$")
PHA_PATTERN = re.compile(r"^pha_r(\d+)_d(\d+)_o(\d+)\.h5$")

SSA_COLOR = "black"
SECONDS_PER_MINUTE = 60.0

# %%
# SSA: one runtime per resolution.
ssa_runtimes: dict[int, float] = {}
for path in DATA_DIR.glob("ssa_r*.h5"):
    match = SSA_PATTERN.match(path.name)
    if match is None:
        continue
    metadata, _, _ = load_results(path)
    ssa_runtimes[int(match.group(1))] = metadata.elapsed_seconds

# %%
# PHA: runtimes grouped by derivatives, then resolution, then delay. Only
# resolution 2048 at the 0-2 derivative settings carries more than one delay.
pha_runtimes: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
for path in DATA_DIR.glob("pha_r*_d*_o*.h5"):
    match = PHA_PATTERN.match(path.name)
    if match is None:
        continue
    resolution, delay, derivatives = (int(group) for group in match.groups())
    metadata, _, _ = load_results(path)
    pha_runtimes[derivatives][resolution][delay] = metadata.elapsed_seconds

# %%
# Render.
figure, ax = plt.subplots(figsize=(9, 6))

ssa_resolutions = np.array(sorted(ssa_runtimes))
ssa_minutes = np.array([ssa_runtimes[r] for r in ssa_resolutions]) / SECONDS_PER_MINUTE
ax.plot(ssa_resolutions, ssa_minutes, color=SSA_COLOR, marker="o", label="SSA")

for derivatives in sorted(pha_runtimes):
    by_resolution = pha_runtimes[derivatives]
    color = f"C{derivatives - 1}"
    marker = Line2D.filled_markers[(derivatives - 1) % len(Line2D.filled_markers)]
    resolutions = np.array(sorted(by_resolution))
    # Curve through the per-resolution mean over available delays.
    means = (
        np.array([np.mean(list(by_resolution[r].values())) for r in resolutions])
        / SECONDS_PER_MINUTE
    )
    ax.plot(
        resolutions, means, color=color, marker=marker, label=f"PHA, {derivatives - 1} derivatives"
    )
    # At resolutions with a delay sweep, scatter each delay to show the spread.
    for r in resolutions:
        delays = by_resolution[r]
        if len(delays) == 1:
            continue
        minutes = np.array(list(delays.values())) / SECONDS_PER_MINUTE
        ax.scatter(np.full(len(minutes), r), minutes, color=color, marker=marker, s=18, zorder=3)

cluster_top = max(pha_runtimes[3][2048].values()) / SECONDS_PER_MINUTE
ax.annotate(
    "resolution 2048: delays 1-17 (0-2 derivatives)",
    xy=(2048, cluster_top),
    xytext=(1180, cluster_top * 1.25),
    fontsize="small",
    ha="center",
    arrowprops={"arrowstyle": "->", "color": "0.4"},
)

ax.set_ylim(bottom=0)
ax.set_xticks(ssa_resolutions)
ax.set_xlabel("Spatial resolution (grid points)")
ax.set_ylabel("Detection runtime (minutes)")
ax.set_title("Detection runtime vs. spatial resolution")
ax.legend()
plt.tight_layout()
plt.show()
