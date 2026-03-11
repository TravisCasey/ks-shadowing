"""Persistence diagram computation with `GUDHI <https://gudhi.inria.fr/>`_ for
PHA shadowing detection.
"""

from dataclasses import dataclass
from typing import Self

import gudhi
import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical


def _compute_persistence_diagram(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sublevel-set persistence diagram for a 1D periodic field.

    Uses GUDHI's periodic cubical complex for persistence on 1D periodic
    domains. The persistence diagram captures the birth and death of connected
    components in the sublevel sets as the threshold increases. This
    representation is invariant to spatial translations.

    Parameters
    ----------
    field : NDArray[np.float64], shape (resolution,)
        Field values at grid points.

    Returns
    -------
    NDArray[np.float64], shape (n_points, 2)
        Persistence pairs ``(birth, death)``. Points with infinite death are
        excluded (the single essential class).
    """
    cubical_complex = gudhi.PeriodicCubicalComplex(  # ty: ignore[unresolved-attribute]
        top_dimensional_cells=field,
        periodic_dimensions=[True],
    )

    cubical_complex.compute_persistence()

    persistence_pairs = cubical_complex.persistence_intervals_in_dimension(0)
    finite_pairs = persistence_pairs[np.isfinite(persistence_pairs[:, 1])]

    return finite_pairs.astype(np.float64)


def _compute_trajectory_diagrams(
    trajectory_physical: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Compute persistence diagrams for each timestep of a trajectory.

    Parameters
    ----------
    trajectory_physical : NDArray[np.float64], shape (num_timesteps, resolution)
        Trajectory in physical space.

    Returns
    -------
    list[NDArray[np.float64]]
        One persistence diagram per timestep. Each diagram has shape
        ``(n_points, 2)`` with ``(birth, death)`` pairs.
    """
    return [_compute_persistence_diagram(field) for field in trajectory_physical]


def _apply_delay_embedding(
    wasserstein_matrix: NDArray[np.float64],
    delay: int,
) -> NDArray[np.float64]:
    r"""Apply time-delay embedding to a Wasserstein distance matrix.

    Computes :math:`W^w(i, j) = \sum_{l=0}^{w-1} W(i+l, (j+l) \bmod J)` where
    :math:`w` is the delay window. This increases the effective dimensionality
    of the comparison by considering consecutive timesteps rather than single
    snapshots.

    Parameters
    ----------
    wasserstein_matrix : NDArray[np.float64], shape (I, J)
        Original Wasserstein distance matrix.
    delay : int
        Time-delay embedding window size (:math:`w`).

    Returns
    -------
    NDArray[np.float64], shape (I - delay + 1, J)
        Embedded distance matrix.

    Raises
    ------
    ValueError
        If ``delay < 1`` or ``delay`` exceeds the trajectory length.
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
class _RPOPersistence:
    r"""Precomputed RPO persistence data for PHA detection.

    Holds the source RPO metadata and its precomputed persistence diagrams.
    This is the PHA equivalent of ``_RPOStateSpace`` in the SSA subpackage.

    Attributes
    ----------
    rpo : :class:`~ks_shadowing.core.rpo.RPO`
        Source RPO containing metadata (index, period, spatial_shift).
    diagrams : list[NDArray[np.float64]]
        Persistence diagrams, one per timestep of the RPO period.
    """

    rpo: RPO
    diagrams: list[NDArray[np.float64]]

    @classmethod
    def from_rpo(cls, rpo: RPO, resolution: int) -> Self:
        """Integrate an RPO and compute persistence diagrams.

        Parameters
        ----------
        rpo : :class:`~ks_shadowing.core.rpo.RPO`
            The RPO to process.
        resolution : int
            Spatial resolution for physical-space representation.

        Returns
        -------
        Self
            Instance with precomputed diagrams.
        """
        rpo_dt = rpo.period / rpo.time_steps
        fourier_trajectory = ksint(rpo.fourier_coeffs, rpo_dt, rpo.time_steps)[:-1]
        physical_trajectory = interleaved_to_physical(fourier_trajectory, resolution)
        diagrams = _compute_trajectory_diagrams(physical_trajectory)
        return cls(rpo=rpo, diagrams=diagrams)

    @property
    def time_steps(self) -> int:
        """Number of timesteps in one RPO period."""
        return self.rpo.time_steps

    @property
    def index(self) -> int:
        """Index of the source RPO."""
        return self.rpo.index
