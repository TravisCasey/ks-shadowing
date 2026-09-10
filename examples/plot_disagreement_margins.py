r"""
Distance-level agreement between the methods
============================================

One point per uniformly sampled (timestep, RPO) row: each method's distance to
that RPO, minimized over the RPO's phases and normalized by that method's own
detection threshold. Thresholds are the dotted reference lines and both axes are
logarithmic. Blue marks rows below both thresholds, red rows below exactly one,
gray neither; the annotation is the Spearman rank correlation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from ks_shadowing import (
    DetectionMetadata,
    KSTrajectory,
    assert_same_trajectory,
    load_results,
    load_rpos,
    shift_distances_sq,
)
from ks_shadowing.pha import (
    KSPersistenceTrajectory,
    apply_delay_embedding,
    wasserstein_matrix,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
SETTINGS = [
    ("PHA\u2013DELAY", DATA_DIR / "pha_r2048_d17_o0.h5"),
    ("PHA\u2013DERIV", DATA_DIR / "pha_r2048_d1_o1.h5"),
]
NUM_ROWS = 2500
BOTH_COLOR = "#4477AA"
DISCORDANT_COLOR = "#EE6677"
NEITHER_COLOR = "0.62"

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
rng = np.random.default_rng(0)

# %%
# Load the SSA result, the shared trajectory, and both PHA settings; build
# each RPO's co-moving modes and per-order persistence diagrams once.
ssa_metadata, trajectory, _ = load_results(SSA_PATH)
num_timesteps = trajectory.num_timesteps
resolution = trajectory.resolution
rpos = load_rpos(REPO_ROOT / ssa_metadata.rpo_file)

setting_metadata: list[DetectionMetadata] = []
for _, pha_path in SETTINGS:
    pha_metadata, pha_trajectory, _ = load_results(pha_path)
    assert_same_trajectory(trajectory, pha_trajectory)
    setting_metadata.append(pha_metadata)
max_order = max(metadata.max_derivative_order for metadata in setting_metadata)

rpo_trajectories = [
    KSTrajectory.from_rpo(rpo, resolution, ssa_metadata.downsample, ssa_metadata.native)
    for rpo in rpos
]
rpo_comoving = [
    rpo_trajectory.to_comoving(rpo.drift_rate).modes
    for rpo, rpo_trajectory in zip(rpos, rpo_trajectories, strict=True)
]
rpo_persistence = [
    [
        KSPersistenceTrajectory.from_trajectory(rpo_trajectory, order=order)
        for rpo_trajectory in rpo_trajectories
    ]
    for order in range(max_order + 1)
]
periods = [rpo_trajectory.num_timesteps for rpo_trajectory in rpo_trajectories]


# %%
# Min-over-phase distances at one (timestep, RPO) row. Window slicing is exact
# for both methods: the SSA minimum is taken over all spatial shifts, so the
# co-moving offset of a nonzero window start drops out, and the Wasserstein
# distance never leaves the window beyond the delay padding.
def _ssa_min(timestep: int, rpo_index: int) -> float:
    """Smallest SSA L2 distance to any phase of the RPO, min over shifts."""
    window = trajectory[timestep : timestep + 1]
    modes = window.to_comoving(rpos[rpo_index].drift_rate, start_time=timestep * trajectory.dt)
    repeated = np.broadcast_to(modes.modes[0], (periods[rpo_index], 17))
    distances_sq = shift_distances_sq(repeated, rpo_comoving[rpo_index], resolution)
    return float(np.sqrt(max(distances_sq.min(), 0.0)))


def _pha_min(timestep: int, rpo_index: int, metadata: DetectionMetadata) -> float:
    """Smallest embedded PHA distance to any phase of the RPO."""
    center = (metadata.delay - 1) // 2
    window = trajectory[timestep - center : timestep + metadata.delay - center]
    raw = np.mean(
        [
            wasserstein_matrix(
                KSPersistenceTrajectory.from_trajectory(window, order=order),
                rpo_persistence[order][rpo_index],
            )
            for order in range(metadata.max_derivative_order + 1)
        ],
        axis=0,
    )
    return float(apply_delay_embedding(raw, metadata.delay).min())


# %%
# Sample rows uniformly, clear of the delay padding, and normalize each
# method's distances by its own threshold.
margin = max(metadata.delay for metadata in setting_metadata)
rows = [
    (int(rng.integers(margin, num_timesteps - margin)), int(rng.integers(0, len(rpos))))
    for _ in range(NUM_ROWS)
]

ssa_margins = (
    np.array([_ssa_min(timestep, rpo_index) for timestep, rpo_index in rows])
    / ssa_metadata.threshold
)
pha_margins = {
    setting_name: (
        np.array([_pha_min(timestep, rpo_index, metadata) for timestep, rpo_index in rows])
        / metadata.threshold
    )
    for (setting_name, _), metadata in zip(SETTINGS, setting_metadata, strict=True)
}

# %%
# Render: one panel per setting, points colored by which thresholds the row
# falls below.
figure, axes = plt.subplots(2, 1, figsize=(3.4, 5.6), sharex=True, sharey=True)
for ax, tag, (setting_name, _) in zip(axes, ("(a)", "(b)"), SETTINGS, strict=True):
    x = ssa_margins
    y = pha_margins[setting_name]
    both = (x < 1) & (y < 1)
    discordant = (x < 1) != (y < 1)
    neither = ~both & ~discordant
    ax.scatter(x[neither], y[neither], s=3, color=NEITHER_COLOR, alpha=0.4, linewidths=0)
    ax.scatter(x[both], y[both], s=3, color=BOTH_COLOR, alpha=0.6, linewidths=0)
    ax.scatter(x[discordant], y[discordant], s=5, color=DISCORDANT_COLOR, alpha=0.85, linewidths=0)
    ax.axvline(1.0, color="0.15", linewidth=0.7, linestyle=":")
    ax.axhline(1.0, color="0.15", linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 12)
    ax.set_ylim(0.25, 12)
    ax.set_aspect("equal")
    ax.set_title(tag, loc="left")
    ax.set_title(setting_name, fontfamily="monospace")
    rank_correlation = spearmanr(x, y).statistic
    ax.annotate(f"$\\rho_s = {rank_correlation:.2f}$", xy=(0.05, 0.9), xycoords="axes fraction")
    ax.set_ylabel(r"$\mathtt{PHA}$ distance / threshold")
axes[-1].set_xlabel(r"$\mathtt{SSA}$ distance / threshold")
plt.show()
