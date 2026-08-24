r"""
Event duration distributions: SSA vs. PHA
==========================================

Per-bin event counts for SSA and three PHA settings on the same trajectory: no
embedding (:math:`w = 1`, :math:`\lambda = 1`), the delay-axis setting
(:math:`w = 17`, :math:`\lambda = 1`), and the derivative-axis setting
(:math:`w = 1`, :math:`\lambda = 2`). Each is plotted as step histograms over a
shared duration grid in trajectory-time units. :math:`w` is the delay window
and :math:`\lambda` the number of derivative orders averaged over, one more
than the ``max_derivative_order`` the filenames carry. The log count axis keeps
both the peak and the long tail readable. The two embedding axes are shown
independently: :math:`w > 1` is used only at :math:`\lambda = 1`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import (
    DetectionMetadata,
    assert_same_trajectory,
    load_results,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
PHA_PATHS = [
    DATA_DIR / "pha_r2048_d1_o0.h5",  # no embedding
    DATA_DIR / "pha_r2048_d17_o0.h5",  # delay axis
    DATA_DIR / "pha_r2048_d1_o1.h5",  # derivative axis
]
BIN_WIDTH_TIMESTEPS = 2

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# One fixed color per derivative order, shared across the gallery figures:
# viridis sampled light to dark with increasing order. SSA is always black.
ORDER_COLORS = plt.get_cmap("viridis")(np.linspace(0.78, 0.0, 6))

# %%
# Load all result files and convert event lengths to durations in time units.
_, ssa_trajectory, ssa_events = load_results(SSA_PATH)
dt = ssa_trajectory.dt
ssa_durations = np.array([e.end_timestep - e.start_timestep for e in ssa_events]) * dt

pha_runs: list[tuple[DetectionMetadata, NDArray[np.float64]]] = []
for pha_path in PHA_PATHS:
    pha_metadata, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(ssa_trajectory, pha_trajectory)
    pha_durations = np.array([e.end_timestep - e.start_timestep for e in pha_events]) * dt
    pha_runs.append((pha_metadata, pha_durations))

# %%
# Compute per-bin event counts on a shared duration grid.
bin_width = BIN_WIDTH_TIMESTEPS * dt
max_duration = max(ssa_durations.max(), *(durations.max() for _, durations in pha_runs))
bins = np.arange(0.0, max_duration + bin_width, bin_width)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# %%
# Render.
figure, ax = plt.subplots(figsize=(3.4, 2.4))
ssa_counts, _ = np.histogram(ssa_durations, bins=bins)
ax.plot(
    bin_centers,
    ssa_counts,
    drawstyle="steps-mid",
    color="black",
    label=f"SSA ({len(ssa_durations)} events)",
)
for pha_metadata, pha_durations in pha_runs:
    pha_counts, _ = np.histogram(pha_durations, bins=bins)
    max_order = pha_metadata.max_derivative_order
    # The unembedded baseline takes the recessive dashed style: dashes vanish
    # where curves overlap, and the baseline is the one curve that stands
    # clear of the cluster. The w = 17 run shares the baseline's lambda = 1
    # color and the lambda = 2 run shares its w = 1 setting, so the dashes
    # are what separate the baseline from each; the embedded runs stay solid.
    if pha_metadata.delay > 1:
        setting = rf"$w = {pha_metadata.delay}$"
        linestyle = "-"
    elif max_order > 0:
        setting = rf"$\lambda = {max_order + 1}$"
        linestyle = "-"
    else:
        setting = "no embedding"
        linestyle = "--"
    ax.plot(
        bin_centers,
        pha_counts,
        drawstyle="steps-mid",
        color=ORDER_COLORS[max_order],
        linestyle=linestyle,
        label=f"PHA, {setting} ({len(pha_durations)} events)",
    )
ax.set_yscale("log")
ax.set_xlim(0, bins[-1])
ax.set_xlabel("Event duration (time units)")
ax.set_ylabel("Number of events")
ax.legend()
plt.show()
