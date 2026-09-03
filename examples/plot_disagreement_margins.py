r"""
Distance margins at method disagreement
=======================================

Where one method reports shadowing against an RPO and the other does not, how
far past its threshold is the non-detecting method? For each timestep inside an
event, the missing method's distance to the RPO, minimized over that RPO's
phases, is normalized by that method's detection threshold. Each panel splits
these margins into three populations: timesteps the other method also covers
(both detect), timesteps it does not (disagreement), and a background of
(timestep, RPO) pairs inside neither method's events. Solid curves use the
delay-embedding reference ``PHA--DELAY`` (:math:`w = 17`); dashed curves the
derivative-embedding reference ``PHA--DERIV`` (:math:`\lambda = 2`).

Disagreement concentrates just above the threshold, well separated from the
background: the states one method misses are near-misses, not states the
missing method scores as far from the orbit. The part of each disagreement
curve left of the threshold line was below threshold and excluded instead by the
minimal-duration filter.
"""

from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ks_shadowing import (
    DetectionMetadata,
    ShadowingEvent,
    assert_same_trajectory,
    load_results,
    load_rpos,
    pha,
    ssa,
)

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
SSA_PATH = DATA_DIR / "ssa_r2048.h5"
# (paper name, result file, linestyle) per embedding-axis reference setting.
SETTINGS: list[tuple[str, Path, Literal["-", "--"]]] = [
    ("PHA\u2013DELAY", DATA_DIR / "pha_r2048_d17_o0.h5", "-"),
    ("PHA\u2013DERIV", DATA_DIR / "pha_r2048_d1_o1.h5", "--"),
]
# Events sampled per panel, and the cap on how much of each is scored.
NUM_EVENTS = 100
MAX_WINDOW_TIMESTEPS = 40
# Background (timestep, RPO) samples are drawn as short blocks, shared
# between the two settings so their background curves differ only where the
# embedding does.
NUM_BACKGROUND_BLOCKS = 50
BACKGROUND_BLOCK_TIMESTEPS = 8
CATEGORY_COLORS = {
    "Both detect": "#4477AA",
    "Disagreement": "#EE6677",
    "Neither detects": "0.5",
}

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
rng = np.random.default_rng(0)

# %%
# Load the SSA result and the trajectory both methods share.
ssa_metadata, trajectory, ssa_events = load_results(SSA_PATH)
num_timesteps = trajectory.num_timesteps
rpos = load_rpos(REPO_ROOT / ssa_metadata.rpo_file)


def _per_rpo_masks(events: list[ShadowingEvent]) -> dict[int, NDArray[np.bool_]]:
    """Union coverage mask per RPO index; missing RPOs read as all-False."""
    events_by_rpo: dict[int, list[ShadowingEvent]] = defaultdict(list)
    for event in events:
        events_by_rpo[event.rpo_index].append(event)
    masks: dict[int, NDArray[np.bool_]] = defaultdict(lambda: np.zeros(num_timesteps, dtype=bool))
    for rpo_index, rpo_events in events_by_rpo.items():
        mask = np.zeros(num_timesteps, dtype=bool)
        for event in rpo_events:
            mask[event.start_timestep : event.end_timestep] = True
        masks[rpo_index] = mask
    return masks


ssa_masks = _per_rpo_masks(ssa_events)


# %%
# Margins on a trajectory window against a single RPO. Slicing is exact for
# both methods: the SSA minimum is taken over all spatial shifts, so the
# constant co-moving offset a nonzero window start introduces drops out, and
# the Wasserstein distance never leaves the window (beyond the delay padding).
def _ssa_margins(window_start: int, window_end: int, rpo_index: int) -> NDArray[np.float64]:
    """SSA min-over-phase distances on the window, over the SSA threshold."""
    distances = ssa.compute_min_distances(trajectory[window_start:window_end], [rpos[rpo_index]])
    return distances / ssa_metadata.threshold


def _pha_margins(
    window_start: int, window_end: int, rpo_index: int, metadata: DetectionMetadata
) -> NDArray[np.float64] | None:
    """Embedded PHA min-over-phase margins, or None if the delay padding
    would leave the trajectory."""
    pad_left = (metadata.delay - 1) // 2
    start = window_start - pad_left
    end = window_end + metadata.delay // 2
    if start < 0 or end > num_timesteps:
        return None
    distances = pha.compute_min_distances(
        trajectory[start:end],
        [rpos[rpo_index]],
        delay=metadata.delay,
        max_derivative_order=metadata.max_derivative_order,
    )
    return distances[pad_left : pad_left + window_end - window_start] / metadata.threshold


def _sample(pool: list, size: int) -> list:
    """Uniform subsample without replacement."""
    if len(pool) <= size:
        return pool
    return [pool[index] for index in rng.choice(len(pool), size=size, replace=False)]


# %%
# Collect the three populations for both settings. Host events supply the
# both-detect and disagreement margins, split by the other method's per-RPO
# coverage; background blocks are (timestep, RPO) pairs neither method covers.
curves: dict[tuple[str, str, str], NDArray[np.float64]] = {}
setting_data = []
for setting_name, pha_path, _ in SETTINGS:
    pha_metadata, pha_trajectory, pha_events = load_results(pha_path)
    assert_same_trajectory(trajectory, pha_trajectory)
    setting_data.append((setting_name, pha_metadata, pha_events, _per_rpo_masks(pha_events)))

for setting_name, pha_metadata, pha_events, pha_masks in setting_data:
    for panel, host_events, other_masks in (
        ("a", ssa_events, pha_masks),
        ("b", pha_events, ssa_masks),
    ):
        agree: list[NDArray[np.float64]] = []
        disagree: list[NDArray[np.float64]] = []
        for event in _sample(host_events, NUM_EVENTS):
            window_end = min(event.end_timestep, event.start_timestep + MAX_WINDOW_TIMESTEPS)
            if panel == "a":
                margins = _pha_margins(
                    event.start_timestep, window_end, event.rpo_index, pha_metadata
                )
            else:
                margins = _ssa_margins(event.start_timestep, window_end, event.rpo_index)
            if margins is None:
                continue
            covered = other_masks[event.rpo_index][event.start_timestep : window_end]
            agree.append(margins[covered])
            disagree.append(margins[~covered])
        curves[setting_name, panel, "Both detect"] = np.concatenate(agree)
        curves[setting_name, panel, "Disagreement"] = np.concatenate(disagree)

max_delay = max(metadata.delay for _, metadata, _, _ in setting_data)
blocks: list[tuple[int, int]] = []
while len(blocks) < NUM_BACKGROUND_BLOCKS:
    block_start = int(rng.integers(max_delay, num_timesteps - MAX_WINDOW_TIMESTEPS))
    block_end = block_start + BACKGROUND_BLOCK_TIMESTEPS
    rpo_index = int(rng.integers(0, len(rpos)))
    if ssa_masks[rpo_index][block_start:block_end].any() or any(
        pha_masks[rpo_index][block_start:block_end].any() for _, _, _, pha_masks in setting_data
    ):
        continue
    blocks.append((block_start, rpo_index))

ssa_background = np.concatenate(
    [
        _ssa_margins(block_start, block_start + BACKGROUND_BLOCK_TIMESTEPS, rpo_index)
        for block_start, rpo_index in blocks
    ]
)
for setting_name, pha_metadata, _, _ in setting_data:
    pha_background: list[NDArray[np.float64]] = []
    for block_start, rpo_index in blocks:
        pha_margins = _pha_margins(
            block_start, block_start + BACKGROUND_BLOCK_TIMESTEPS, rpo_index, pha_metadata
        )
        assert pha_margins is not None
        pha_background.append(pha_margins)
    curves[setting_name, "a", "Neither detects"] = np.concatenate(pha_background)
    curves[setting_name, "b", "Neither detects"] = ssa_background

# %%
# Render: empirical CDFs of the margins, one panel per disagreement direction,
# one linestyle per embedding setting.
figure, axes = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)
panel_titles = {
    "a": r"$\mathtt{SSA}$ detects; $\mathtt{PHA}$ distance",
    "b": r"$\mathtt{PHA}$ detects; $\mathtt{SSA}$ distance",
}
for ax, panel in zip(axes, ("a", "b"), strict=True):
    for setting_name, _, linestyle in SETTINGS:
        for category, color in CATEGORY_COLORS.items():
            values = np.sort(curves[setting_name, panel, category])
            fractions = np.arange(1, values.size + 1) / values.size
            ax.step(values, fractions, where="post", color=color, linestyle=linestyle)
    ax.axvline(1.0, color="0.15", linewidth=0.7, linestyle=":")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.set_title(f"({panel})", loc="left")
    ax.set_title(panel_titles[panel])
    ax.set_ylabel("Fraction of timesteps")
axes[-1].set_xlabel("Distance / threshold (min over RPO phase)")

category_handles = [
    Line2D([], [], color=color, label=category) for category, color in CATEGORY_COLORS.items()
]
setting_handles = [
    Line2D([], [], color="0.15", linestyle=linestyle, label=setting_name)
    for setting_name, _, linestyle in SETTINGS
]
axes[0].legend(handles=category_handles, loc="lower right")
axes[1].legend(handles=setting_handles, loc="lower right", prop={"family": "monospace"})
plt.show()
