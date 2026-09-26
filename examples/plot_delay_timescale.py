r"""
The best delay is a time, not a number of timesteps
===================================================

``PHA--DELAY`` averages the Wasserstein distances of :math:`m` consecutive
timesteps, so the span it actually looks at is :math:`(m - 1)\,\Delta t`, where
:math:`\Delta t` is the trajectory's sampling step. Scoring a sweep of
:math:`m` against SSA picks out a best value, but that value alone does not say
whether it belongs to the orbit dynamics or to the sampling.

Separating the two needs one trajectory sampled several ways. This figure uses
a single integration, stored at four steps from :math:`\Delta t = 0.92` down to
:math:`0.12`, with a ``PHA--DELAY`` sweep and its own SSA reference at each.
Against :math:`m` the four :math:`F_1` curves peak at :math:`m = 9`, 17, 31 and
61, a spread that tracks the eightfold spread in :math:`\Delta t`. Plotted
against :math:`(m - 1)\,\Delta t`, as here, the peaks land together, a little
above 7 time units whatever the sampling.

The window is therefore the quantity worth quoting, and an :math:`m` tuned at
one sampling step carries over to another by holding that window fixed. The
dashed guide is one attempt at choosing it with no reference
detection to score against: the integrated correlation time of the trajectory's
own persistence diagrams. With :math:`D^2(\tau)` the mean squared Wasserstein
distance between diagrams a lag :math:`\tau` apart and :math:`D^2_\infty` its
value between unrelated snapshots, :math:`C(\tau) = 1 - D^2(\tau) / D^2_\infty`
plays the role of an autocorrelation, and its integral over :math:`\tau`, taken
to the first zero crossing because :math:`C` oscillates beyond it, is the time
over which one diagram stops saying much about the next. For an exponential
decay this equals the :math:`1/e` time, but it needs no threshold level. It is
the right order of magnitude but sits below the observed optima: a
window averaging :math:`m` distances gains only once it spans several
correlation times, so read the guide as a floor on a sensible window rather
than a prediction of the best one.

The peak heights are not directly comparable across steps. Each sweep is scored
against the SSA run at its own :math:`\Delta t`, and SSA is itself somewhat
step-dependent, finding fewer and longer events as sampling refines.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing import (
    KSTrajectory,
    ShadowingEvent,
    assert_same_trajectory,
    load_results,
    load_rpos,
)
from ks_shadowing.pha import KSPersistenceTrajectory, wasserstein_matrix

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
# Provisional: this sweep is not yet part of the committed fixtures, and unlike
# the rest of the gallery it needs four trajectories rather than one.
DATA_DIR = REPO_ROOT / "results" / "dt_study" / "phase1"
RESOLUTION = 512
# Sampling step in time units, and the subdirectory holding its runs.
SAMPLING_STEPS = ((0.92, "s46"), (0.46, "s23"), (0.24, "s12"), (0.12, "s6"))
# Diagram drift is a property of the trajectory, so any one sampling step
# resolves it. The estimate needs a long segment rather than a finely sampled
# one: a short segment biases the saturation level and with it the whole
# correlation curve.
DRIFT_DIRECTORY = "s23"
DRIFT_TIMESTEPS = 2500
# Distances are computed from every timestep to every ``DRIFT_ANCHOR_STRIDE``-th
# one. Neighbouring timesteps are strongly correlated, so denser anchors add
# cost without adding independent samples.
DRIFT_ANCHOR_STRIDE = 10
DRIFT_MAX_LAG = 24.0
DRIFT_BASELINE_LAG = 100.0

plt.style.use(REPO_ROOT / "examples" / "gallery.mplstyle")
# Viridis light to dark as the sampling step refines, following the derivative
# figures' convention of darker markers further along the swept axis.
STEP_COLORS = plt.get_cmap("viridis")(np.linspace(0.78, 0.0, len(SAMPLING_STEPS)))
STEP_MARKERS = ("o", "s", "^", "D")
GUIDE_COLOR = "0.45"
PHA_DELAY = "PHA\u2013DELAY"


# %%
# Agreement is measured on the (RPO, timestep) cell grid, as in the
# :ref:`agreement example <sphx_glr_auto_examples_plot_coverage_vs_embedding.py>`:
# a run that flags the right timestep against the wrong orbit is penalized.
def _per_rpo_mask(
    events: list[ShadowingEvent], num_rpos: int, num_timesteps: int
) -> NDArray[np.bool_]:
    """Return the ``(num_rpos, num_timesteps)`` coverage grid of ``events``."""
    mask = np.zeros((num_rpos, num_timesteps), dtype=bool)
    for event in events:
        mask[event.rpo_index, event.start_timestep : event.end_timestep] = True
    return mask


def _f1(reference: NDArray[np.bool_], candidate: NDArray[np.bool_]) -> float:
    """Return the F1 score of ``candidate`` against ``reference``."""
    true_positives = float((reference & candidate).sum())
    false_positives = float((~reference & candidate).sum())
    false_negatives = float((reference & ~candidate).sum())
    return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)


# %%
# One sweep per sampling step, each scored against the SSA run that shares its
# trajectory.
embedding_lengths: dict[float, NDArray[np.int64]] = {}
scores: dict[float, NDArray[np.float64]] = {}
for sampling_step, directory in SAMPLING_STEPS:
    run_directory = DATA_DIR / directory
    metadata, trajectory, ssa_events = load_results(run_directory / f"ssa_r{RESOLUTION}.h5")
    num_rpos = len(load_rpos(REPO_ROOT / metadata.rpo_file))
    ssa_mask = _per_rpo_mask(ssa_events, num_rpos, trajectory.num_timesteps)

    sweep: list[tuple[int, float]] = []
    for path in run_directory.glob(f"pha_r{RESOLUTION}_d*_o0.h5"):
        pha_metadata, pha_trajectory, pha_events = load_results(path)
        assert_same_trajectory(trajectory, pha_trajectory)
        pha_mask = _per_rpo_mask(pha_events, num_rpos, trajectory.num_timesteps)
        sweep.append((pha_metadata.delay, _f1(ssa_mask, pha_mask)))

    sweep.sort()
    embedding_lengths[sampling_step] = np.array([length for length, _ in sweep])
    scores[sampling_step] = np.array([score for _, score in sweep])

# %%
# How fast the trajectory's diagrams drift away from one another, against the
# lag between them. Column ``j`` of the squared Wasserstein matrix holds the
# distances from every timestep to anchor timestep ``anchors[j]``, so the
# entries ``(anchors[j] + k, j)`` average to the mean squared distance between
# timesteps k steps apart; pairs far beyond any correlation fix the level it
# tends to. Squared distances are what make ``1 - drift / unrelated`` an
# autocorrelation: for a stationary signal the mean squared difference at a
# lag is ``2 * variance * (1 - autocorrelation)``.
drift_trajectory = KSTrajectory.load(DATA_DIR / DRIFT_DIRECTORY / "trajectory.h5", RESOLUTION)
drift_step = drift_trajectory.dt
drift_diagrams = KSPersistenceTrajectory.from_trajectory(drift_trajectory[:DRIFT_TIMESTEPS])
anchors = np.arange(0, DRIFT_TIMESTEPS, DRIFT_ANCHOR_STRIDE)
anchor_distances_sq = wasserstein_matrix(drift_diagrams, drift_diagrams[anchors]) ** 2

lag_count = int(DRIFT_MAX_LAG / drift_step)
lags = np.arange(lag_count + 1) * drift_step
# Anchors within ``lag_count`` of the end are dropped so every lag averages the
# same set of anchors.
interior = np.flatnonzero(anchors + lag_count < DRIFT_TIMESTEPS)
drift_sq = np.array(
    [anchor_distances_sq[anchors[interior] + lag, interior].mean() for lag in range(lag_count + 1)]
)
separations = np.abs(np.arange(DRIFT_TIMESTEPS)[:, np.newaxis] - anchors[np.newaxis, :])
unrelated_sq = anchor_distances_sq[separations >= int(DRIFT_BASELINE_LAG / drift_step)].mean()
# Integrated rather than read off a threshold crossing: no level to choose, and
# the result is not quantized to the sampling step. The integral stops at the
# first zero crossing because the correlation does not decay monotonically: it
# swings negative past the crossing and keeps oscillating, and integrating
# through those lobes would cancel the leading one.
correlation = 1.0 - drift_sq / unrelated_sq
if not (correlation <= 0.0).any():
    raise ValueError(f"diagram correlation stays positive out to lag {DRIFT_MAX_LAG}")
first_zero = int(np.argmax(correlation <= 0.0))
correlation_time = float(np.trapezoid(correlation[: first_zero + 1], lags[: first_zero + 1]))

# %%
# Render: every sweep against the window it spans rather than against
# :math:`m`.
figure, ax = plt.subplots(figsize=(3.4, 2.7))

for index, (sampling_step, _) in enumerate(SAMPLING_STEPS):
    ax.plot(
        (embedding_lengths[sampling_step] - 1) * sampling_step,
        scores[sampling_step],
        color=STEP_COLORS[index],
        marker=STEP_MARKERS[index],
        label=f"{sampling_step:g}",
    )

# The curves fill the panel, so the legend gets its own band of headroom
# rather than a corner.
lower, upper = ax.get_ylim()
ax.set_ylim(lower, upper + 0.3 * (upper - lower))

# The guide sits beneath the data, labelled at its foot where the curves leave
# room, and stops short of the headroom above them, which belongs to the legend.
ax.vlines(
    correlation_time, lower, upper, color=GUIDE_COLOR, linestyle="--", linewidth=0.8, zorder=0
)
ax.annotate(
    "Correlation time",
    xy=(correlation_time, lower),
    xytext=(4, 2),
    textcoords="offset points",
    color=GUIDE_COLOR,
    fontsize=7,
    va="bottom",
)
ax.set_title(PHA_DELAY, fontfamily="monospace")
ax.set_xlabel(r"Embedding window $(m - 1)\,\Delta t$")
ax.set_ylabel(r"$F_1$")
ax.legend(
    title=r"Sampling step $\Delta t$",
    ncol=len(SAMPLING_STEPS),
    loc="upper center",
    columnspacing=1.0,
    handletextpad=0.4,
)

plt.show()
