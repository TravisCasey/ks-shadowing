r"""
Spectral shift under derivative embedding
==========================================

PHA's derivative embedding computes persistence diagrams of
:math:`\partial^{k} u / \partial x^{k}`, which in Fourier space multiplies mode
``q`` by :math:`(2 \pi q / L)^{k}`. The Kuramoto-Sivashinsky attractor at
``L = 22`` holds 99.98% of its energy below ``q = 8``, so each added order
shifts the field the diagrams actually see toward modes the dynamics barely
populate: by 5 derivatives, a quarter of the energy sits above that cutoff.

The state is stored as 17 complex Fourier modes with the constant and Nyquist
modes identically zero, so the spectrum spans ``q = 1`` to ``q = 15`` and the
proportions below are specific to that truncation.

This is the mechanism behind the saturation of the derivative sweep: the orders
that stop helping are the ones weighted toward wavenumbers that carry no
dynamics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ks_shadowing import DOMAIN_SIZE, load_results

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "examples" / "data"
TRAJECTORY_PATH = DATA_DIR / "ssa_r2048.h5"
DERIVATIVE_ORDERS = range(6)
HIGH_WAVENUMBER = 8
# The stored trajectory's highest modes relax onto the attractor within the
# first few hundred rows; the spectrum before that is not representative.
TRANSIENT_TIMESTEPS = 500
CUTOFF_COLOR = "black"

# %%
# Mean energy per Fourier mode on the attractor.
_, trajectory, _ = load_results(TRAJECTORY_PATH)
modes = trajectory[TRANSIENT_TIMESTEPS : trajectory.num_timesteps].modes
wavenumbers = 2.0 * np.pi * np.arange(modes.shape[1]) / DOMAIN_SIZE
energy = (np.abs(modes) ** 2).mean(axis=0)

# %%
# Reweight that spectrum by the derivative multiplier for each order, and record
# how much of each reweighted spectrum sits above the cutoff.
mode_indices = np.arange(modes.shape[1])
spectra = []
high_fractions = []
for order in DERIVATIVE_ORDERS:
    weighted = energy * wavenumbers ** (2 * order)
    weighted = weighted / weighted.sum()
    spectra.append(weighted)
    high_fractions.append(float(weighted[HIGH_WAVENUMBER:].sum()))

# %%
# Render.
figure, (ax_spectrum, ax_fraction) = plt.subplots(1, 2, figsize=(13, 5))

for order, weighted in zip(DERIVATIVE_ORDERS, spectra, strict=True):
    ax_spectrum.plot(
        mode_indices[1:-1],
        weighted[1:-1],
        color=f"C{order}",
        marker=Line2D.filled_markers[order % len(Line2D.filled_markers)],
        label=f"order {order}",
    )
ax_spectrum.axvline(HIGH_WAVENUMBER, color=CUTOFF_COLOR, linestyle="--", linewidth=1.0)
ax_spectrum.set_yscale("log")
ax_spectrum.set_ylim(bottom=1e-8)
ax_spectrum.set_xlabel("Fourier mode index $q$")
ax_spectrum.set_ylabel("Fraction of energy")
ax_spectrum.set_title("Energy distribution after differentiation")
ax_spectrum.legend(fontsize="small")

ax_fraction.plot(list(DERIVATIVE_ORDERS), high_fractions, color=CUTOFF_COLOR, marker="o")
for order, fraction in zip(DERIVATIVE_ORDERS, high_fractions, strict=True):
    ax_fraction.annotate(
        f"{fraction:.2%}",
        xy=(order, fraction),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize="small",
    )
ax_fraction.set_xlabel("Derivative order")
ax_fraction.set_ylabel(f"Fraction of energy above $q = {HIGH_WAVENUMBER}$")
ax_fraction.set_title("Energy pushed into modes the attractor barely uses")
ax_fraction.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
