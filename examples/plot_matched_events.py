"""
Matched events: SSA vs. PHA
============================

Each detection method produces a list of shadowing events keyed by RPO index,
start timestep, and end timestep. Pairing SSA and PHA events with the same RPO
and overlapping windows yields a scatter of (SSA length, PHA length) colored by
the intersection-over-union fraction of their windows. Points near the diagonal
mark events the two methods agree on in duration; high IoU (yellow) means the
windows themselves overlap heavily, low IoU (purple) means they barely intersect.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing import (
    assert_same_trajectory,
    load_results,
    match_events,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
SSA_PATH = REPO_ROOT / "examples" / "data" / "ssa_r2048.h5"
PHA_PATH = REPO_ROOT / "examples" / "data" / "pha_r2048_d8_o2.h5"

# %%
# Load both result files and verify they share a trajectory.
_, ssa_trajectory, ssa_events = load_results(SSA_PATH)
pha_metadata, pha_trajectory, pha_events = load_results(PHA_PATH)
assert_same_trajectory(ssa_trajectory, pha_trajectory)

matches = match_events(ssa_events, pha_events)
ssa_lengths = np.array([m.ssa_event.end_timestep - m.ssa_event.start_timestep for m in matches])
pha_lengths = np.array([m.pha_event.end_timestep - m.pha_event.start_timestep for m in matches])
iou = np.array([m.intersection_length / m.union_length for m in matches])

# %%
# Render. Both detectors discard events shorter than their ``min_duration``
# (here ``pha_metadata.min_duration`` timesteps), which is why the lower-left
# corner of the plot is empty.
figure, ax = plt.subplots(figsize=(10, 7.5))
scatter = ax.scatter(ssa_lengths, pha_lengths, c=iou, cmap="viridis", vmin=0, vmax=1)
figure.colorbar(scatter, ax=ax, label="Overlap (IoU)")
ax.set_xlabel("SSA event length (timesteps)")
ax.set_ylabel("PHA event length (timesteps)")
ax.set_title(
    f"{len(matches)} matched pairs "
    f"(PHA delay={pha_metadata.delay}, {pha_metadata.derivatives - 1} derivative)"
)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.show()
