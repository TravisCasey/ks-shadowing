"""
Event duration distributions: SSA vs. PHA
==========================================

Per-bin event counts for SSA and one PHA configuration on the same trajectory,
plotted as lines over a shared duration grid in trajectory-time units.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing import (
    assert_same_trajectory,
    load_results,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
SSA_PATH = REPO_ROOT / "examples" / "data" / "ssa_r2048.h5"
PHA_PATH = REPO_ROOT / "examples" / "data" / "pha_r2048_d13_o1.h5"
BIN_WIDTH_TIMESTEPS = 2

# %%
# Load both result files and compute per-bin event counts on a shared grid.
_, ssa_trajectory, ssa_events = load_results(SSA_PATH)
pha_metadata, pha_trajectory, pha_events = load_results(PHA_PATH)
assert_same_trajectory(ssa_trajectory, pha_trajectory)

dt = ssa_trajectory.dt
ssa_durations = np.array([e.end_timestep - e.start_timestep for e in ssa_events]) * dt
pha_durations = np.array([e.end_timestep - e.start_timestep for e in pha_events]) * dt

bin_width = BIN_WIDTH_TIMESTEPS * dt
max_duration = max(ssa_durations.max(), pha_durations.max())
bins = np.arange(0.0, max_duration + bin_width, bin_width)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

ssa_counts, _ = np.histogram(ssa_durations, bins=bins)
pha_counts, _ = np.histogram(pha_durations, bins=bins)

# %%
# Render.
figure, ax = plt.subplots(figsize=(10, 5))
ax.plot(bin_centers, ssa_counts, color="tab:blue", label=f"SSA ({len(ssa_events)} events)")
ax.plot(
    bin_centers,
    pha_counts,
    color="tab:orange",
    label=(
        f"PHA delay={pha_metadata.delay}, derivatives={pha_metadata.derivatives} "
        f"({len(pha_events)} events)"
    ),
)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Event duration (time units)")
ax.set_ylabel("Number of events")
ax.set_title(f"Shadowing event durations, bin = {bin_width:.2f} time units")
ax.legend()
plt.tight_layout()
plt.show()
