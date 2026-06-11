"""
Event duration distributions: SSA vs. PHA
==========================================

Per-bin event counts for SSA and three PHA derivative orders at delay 8 on the
same trajectory, plotted as lines over a shared duration grid in
trajectory-time units.
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
PHA_PATHS = [DATA_DIR / f"pha_r2048_d8_o{derivatives}.h5" for derivatives in (1, 2, 3)]
BIN_WIDTH_TIMESTEPS = 2

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
figure, ax = plt.subplots(figsize=(10, 5))
ssa_counts, _ = np.histogram(ssa_durations, bins=bins)
ax.plot(bin_centers, ssa_counts, color="black", label=f"SSA ({len(ssa_durations)} events)")
for pha_metadata, pha_durations in pha_runs:
    pha_counts, _ = np.histogram(pha_durations, bins=bins)
    ax.plot(
        bin_centers,
        pha_counts,
        color=f"C{pha_metadata.derivatives - 1}",
        label=(
            f"PHA delay={pha_metadata.delay}, {pha_metadata.derivatives - 1} derivatives "
            f"({len(pha_durations)} events)"
        ),
    )
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Event duration (time units)")
ax.set_ylabel("Number of events")
ax.set_title("Shadowing event duration")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend()
plt.tight_layout()
plt.show()
