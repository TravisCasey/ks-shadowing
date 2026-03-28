"""Persistence diagram computation for PHA shadowing detection."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical


def _compute_persistence_diagram(field: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Compute sublevel-set persistence diagram for a 1D periodic field.

    Computes :math:`H_0` sublevel-set persistence on a circle (1D periodic
    domain). Entires are processed in order of increasing field value; connected
    components are tracked with union-find to record birth-death pairs when two
    distinct components merge.

    Each local minimum of the discrete field births a connected component in the
    sublevel set :math:`\{x : f(x) \le t\}`. When two components merge (at an
    entry between two distinct minima), the younger component (higher birth
    value) dies. The resulting diagram is invariant to spatial translations.

    Parameters
    ----------
    field : NDArray[np.float64], shape (resolution,)
        Field values at grid points.

    Returns
    -------
    NDArray[np.float64], shape (n_points, 2)
        Persistence pairs ``(birth, death)`` with ``birth < death``. The single
        essential class (infinite death) is excluded.
    """
    if field.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    ordered = np.argsort(field, kind="stable")

    # Union-find data
    parent = list(range(field.size))
    comp_birth = list(field)
    active = [False] * field.size
    pairs: list[tuple[float, float]] = []

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for entry in ordered:
        active[entry] = True

        left = (entry - 1) % field.size
        right = (entry + 1) % field.size

        neighbor_roots: list[int] = []
        if active[left]:
            neighbor_roots.append(_find(left))
        if active[right]:
            root_right = _find(right)
            if not neighbor_roots or root_right != neighbor_roots[0]:
                neighbor_roots.append(root_right)

        if not neighbor_roots:
            pass  # new component; birth already in comp_birth[vertex]
        elif len(neighbor_roots) == 1:
            parent[entry] = neighbor_roots[0]
        else:
            root_a, root_b = neighbor_roots
            # Elder rule: component with lower birth survives
            if comp_birth[root_a] > comp_birth[root_b]:
                root_a, root_b = root_b, root_a
            # root_a is elder, root_b is younger and dies
            death = float(field[entry])
            if comp_birth[root_b] < death:
                pairs.append((comp_birth[root_b], death))
            # Merge: younger and vertex attach to elder
            parent[root_b] = root_a
            parent[entry] = root_a

    if not pairs:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(pairs, dtype=np.float64)


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
