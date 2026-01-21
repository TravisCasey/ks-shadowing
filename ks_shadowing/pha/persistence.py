"""Persistence diagram computation for PHA shadowing detection.

This module provides functionality for computing persistence diagrams of 1D
periodic fields using sublevel-set filtration. The persistence diagrams
quotient out the continuous translational symmetry, making them ideal for
comparing trajectories with RPOs.
"""

from dataclasses import dataclass
from typing import Self

import gudhi
import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical
from ks_shadowing.pha.wasserstein import wasserstein_distance


def compute_persistence_diagram(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sublevel-set persistence diagram for a 1D periodic field.

    Uses GUDHI's periodic cubical complex for persistence on 1D periodic domains.
    The persistence diagram captures the birth and death of connected components
    in the sublevel sets as the threshold increases. Notably, this
    representation is invariant to spatial translations.

    Args:
        field: 1D array of field values at grid points.

    Returns:
        Array of shape `(n_points, 2)` with `(birth, death)` pairs. Points with
        infinite death are excluded (the single essential class).
    """
    # GUDHI periodic cubical complex for 1D periodic domain
    cubical_complex = gudhi.PeriodicCubicalComplex(  # ty: ignore[unresolved-attribute]
        top_dimensional_cells=field,
        periodic_dimensions=[True],
    )

    # Compute persistence
    cubical_complex.compute_persistence()

    # Extract 0-dimensional persistence (connected components)
    # Filter out infinite death values (essential class)
    persistence_pairs = cubical_complex.persistence_intervals_in_dimension(0)
    finite_pairs = persistence_pairs[np.isfinite(persistence_pairs[:, 1])]

    return finite_pairs.astype(np.float64)


def compute_trajectory_diagrams(
    trajectory_physical: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Compute persistence diagrams for each timestep of a trajectory.

    Args:
        trajectory_physical: Trajectory in physical space, shape `(num_timesteps, resolution)`.

    Returns:
        List of persistence diagrams, one per timestep. Each diagram is an
        array of shape `(n_points, 2)` with `(birth, death)` pairs.
    """
    return [compute_persistence_diagram(field) for field in trajectory_physical]


def compute_wasserstein_matrix(
    traj_diagrams: list[NDArray[np.float64]],
    rpo_diagrams: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute Wasserstein distance matrix between trajectory and RPO diagrams.

    Computes `W(i, j) = d_W2(PD(u(t_i)), PD(u_r(tau_j)))` for all `(i, j)`.

    Args:
        traj_diagrams: List of `I` persistence diagrams for trajectory.
        rpo_diagrams: List of `J` persistence diagrams for RPO.

    Returns:
        Matrix of shape `(I, J)` with Wasserstein distances.
    """
    num_traj = len(traj_diagrams)
    num_rpo = len(rpo_diagrams)

    distances = np.empty((num_traj, num_rpo), dtype=np.float64)

    for i, traj_diag in enumerate(traj_diagrams):
        for j, rpo_diag in enumerate(rpo_diagrams):
            distances[i, j] = wasserstein_distance(traj_diag, rpo_diag)

    return distances


def apply_delay_embedding(
    wasserstein_matrix: NDArray[np.float64],
    delay: int,
) -> NDArray[np.float64]:
    """Apply time-delay embedding to Wasserstein distance matrix.

    Computes `W^w(i, j) = sum_{l=0}^{w-1} W(i+l, (j+l) mod J)` where `w` is the
    delay. This increases the effective dimensionality of the comparison by
    considering a window of consecutive timesteps rather than single snapshots.

    Args:
        wasserstein_matrix: Original distance matrix of shape `(I, J)`.
        delay: Time-delay embedding window size (w in the paper).

    Returns:
        Embedded distance matrix of shape `(I - delay + 1, J)`.
    """
    trajectory_timesteps, rpo_timesteps = wasserstein_matrix.shape

    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if delay > trajectory_timesteps:
        raise ValueError(f"delay ({delay}) exceeds trajectory length ({trajectory_timesteps})")

    # Tile RPO dimension to handle wraparound: append first (delay-1) columns
    if delay > 1:
        tiled = np.concatenate([wasserstein_matrix, wasserstein_matrix[:, : delay - 1]], axis=1)
    else:
        tiled = wasserstein_matrix

    # Output dimensions
    delayed_timesteps = trajectory_timesteps - delay + 1
    delayed = np.zeros((delayed_timesteps, rpo_timesteps), dtype=np.float64)

    for offset in range(delay):
        # At offset l: trajectory index is (i + l), RPO index is (j + l)
        delayed += tiled[offset : offset + delayed_timesteps, offset : offset + rpo_timesteps]

    return delayed


@dataclass
class RPOPersistence:
    """Precomputed RPO persistence data for PHA detection.

    Holds both the source RPO metadata and its precomputed persistence diagrams.
    This is the PHA equivalent of `RPOStateSpace` used by SSA.

    Attributes:
        rpo: Source RPO containing metadata (index, period, spatial_shift).
        diagrams: List of persistence diagrams, one per timestep of the RPO period.
    """

    rpo: RPO
    diagrams: list[NDArray[np.float64]]

    @classmethod
    def from_rpo(cls, rpo: RPO, resolution: int) -> Self:
        """Integrate an RPO and compute persistence diagrams.

        Integrates the RPO for one full period using its native timestep,
        transforms to physical space, and computes persistence diagrams
        for each timestep.

        Args:
            rpo: The RPO to process.
            resolution: Spatial resolution for physical space representation.

        Returns:
            RPOPersistence with precomputed diagrams.
        """
        rpo_dt = rpo.period / rpo.time_steps
        fourier_trajectory = ksint(rpo.fourier_coeffs, rpo_dt, rpo.time_steps)[:-1]
        physical_trajectory = interleaved_to_physical(fourier_trajectory, resolution)
        diagrams = compute_trajectory_diagrams(physical_trajectory)
        return cls(rpo=rpo, diagrams=diagrams)

    @property
    def time_steps(self) -> int:
        """Number of timesteps in one RPO period."""
        return self.rpo.time_steps

    @property
    def index(self) -> int:
        """Index of the source RPO."""
        return self.rpo.index
