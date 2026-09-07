r"""
Close-pass margins at method disagreement
=========================================

At a grid cell (timestep, RPO phase), a method registers a close pass when its
distance is below its own threshold. For SSA we minimize over spatial shifts,
which projects away the third axis not present in PHA never had. A disagreeing
cell is one where only one method's criterion holds. Each panel shows the
probability density of the failing method's distance over its own threshold at
those cells. Solid ticks on the baseline mark each curve's median.

Every density peaks at the threshold itself and decays monotonically. Cells are
subsampled with a fixed seed, and densities are Gaussian kernel estimates
reflected about the boundary at 1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

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
# (paper name, result file, color) per embedding-axis reference setting.
SETTINGS = [
    ("PHA\u2013DELAY", DATA_DIR / "pha_r2048_d17_o0.h5", "#EE6677"),
    ("PHA\u2013DERIV", DATA_DIR / "pha_r2048_d1_o1.h5", "#CCBB44"),
]
# Rows are (timestep, RPO) pairs. The SSA-pass direction scans many rows with
# the cheap SSA distance and evaluates PHA only where SSA passes; the PHA-pass
# direction, whose cells are several times more common, evaluates a smaller
# sample outright.
NUM_SCAN_ROWS = 12000
NUM_BRUTE_ROWS = 6000
GRID = np.linspace(1.0, 3.0, 400)

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
for _, pha_path, _ in SETTINGS:
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
# One row of each method's distance grid: the distance to every phase of one
# RPO at one timestep.
def _ssa_row(timestep: int, rpo_index: int) -> NDArray[np.float64]:
    """SSA L2 distance, minimized over spatial shifts, at every RPO phase."""
    window = trajectory[timestep : timestep + 1]
    modes = window.to_comoving(rpos[rpo_index].drift_rate, start_time=timestep * trajectory.dt)
    repeated = np.broadcast_to(modes.modes[0], (periods[rpo_index], 17))
    distances_sq = shift_distances_sq(repeated, rpo_comoving[rpo_index], resolution)
    return np.sqrt(np.maximum(distances_sq.min(axis=1), 0.0))


def _pha_row(timestep: int, rpo_index: int, metadata: DetectionMetadata) -> NDArray[np.float64]:
    """Embedded PHA distance at every RPO phase, attributed to window centers."""
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
    embedded = apply_delay_embedding(raw, metadata.delay)
    period = periods[rpo_index]
    return embedded[0][(np.arange(period) - center) % period]


margin = max(metadata.delay for metadata in setting_metadata)


def _draw_rows(count: int) -> list[tuple[int, int]]:
    """Uniformly random (timestep, RPO) rows, clear of the delay padding."""
    return [
        (int(rng.integers(margin, num_timesteps - margin)), int(rng.integers(0, len(rpos))))
        for _ in range(count)
    ]


# %%
# Sample disagreeing cells. Scanning rows with the SSA distance first is an
# exact filter for the SSA-pass direction: the kept rows are precisely those
# containing cells of its population.
margins: dict[tuple[str, str], NDArray[np.float64]] = {}

scan_hits = []
for timestep, rpo_index in _draw_rows(NUM_SCAN_ROWS):
    ssa_distances = _ssa_row(timestep, rpo_index)
    passes = ssa_distances < ssa_metadata.threshold
    if passes.any():
        scan_hits.append((timestep, rpo_index, passes))
for (setting_name, _, _), metadata in zip(SETTINGS, setting_metadata, strict=True):
    values = []
    for timestep, rpo_index, passes in scan_hits:
        pha_distances = _pha_row(timestep, rpo_index, metadata)
        fails = passes & (pha_distances >= metadata.threshold)
        values.append(pha_distances[fails] / metadata.threshold)
    margins[setting_name, "a"] = np.concatenate(values)

brute_rows = _draw_rows(NUM_BRUTE_ROWS)
brute_ssa = [_ssa_row(timestep, rpo_index) for timestep, rpo_index in brute_rows]
for (setting_name, _, _), metadata in zip(SETTINGS, setting_metadata, strict=True):
    values = []
    for (timestep, rpo_index), ssa_distances in zip(brute_rows, brute_ssa, strict=True):
        pha_distances = _pha_row(timestep, rpo_index, metadata)
        fails = (pha_distances < metadata.threshold) & (ssa_distances >= ssa_metadata.threshold)
        values.append(ssa_distances[fails] / ssa_metadata.threshold)
    margins[setting_name, "b"] = np.concatenate(values)


# %%
# Render: one panel per disagreement direction, densities filled, colored by
# embedding setting.
def _reflected_pdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gaussian KDE on ``GRID``, reflected about the boundary at 1."""
    kde = gaussian_kde(values)
    return kde(GRID) + kde(2.0 - GRID)


figure, axes = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True, sharey=True)
panel_titles = {
    "a": r"$\mathtt{SSA}$ close pass; $\mathtt{PHA}$ distance",
    "b": r"$\mathtt{PHA}$ close pass; $\mathtt{SSA}$ distance",
}
for ax, panel in zip(axes, ("a", "b"), strict=True):
    for setting_name, _, color in SETTINGS:
        values = margins[setting_name, panel]
        density = _reflected_pdf(values)
        ax.fill_between(GRID, density, color=color, alpha=0.22, linewidth=0)
        ax.plot(GRID, density, color=color, label=setting_name)
        ax.plot(
            [np.median(values)],
            [0],
            marker="|",
            markersize=7,
            markeredgewidth=1.3,
            color=color,
            clip_on=False,
        )
    ax.set_xlim(1, 3)
    ax.set_ylim(bottom=0)
    ax.set_title(f"({panel})", loc="left")
    ax.set_title(panel_titles[panel])
    ax.set_ylabel("Probability density")
axes[0].legend(prop={"family": "monospace"}, loc="upper right")
axes[-1].set_xlabel("Distance / threshold")
plt.show()
