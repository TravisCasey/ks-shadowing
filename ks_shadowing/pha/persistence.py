"""Persistence diagram computation for PHA shadowing detection."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core import DEFAULT_CHUNK_SIZE, DOMAIN_SIZE
from ks_shadowing.core.trajectory import KSTrajectory


@dataclass(frozen=True, slots=True)
class KSPersistenceTrajectory:
    """A Kuramoto-Sivashinsky trajectory in the space of persistence diagrams.

    Each element of ``diagrams`` is a ``(num_pairs, 2)`` array containing birth
    and death pairs of the sublevel-set zeroth persistence diagram of each point
    in physical space of a trajectory with constant timestep ``dt``. The
    diagonal points and the single essential class (having infinite death) are
    not included.

    Prefer
    :meth:`~ks_shadowing.pha.persistence.KSPersistenceTrajectory.from_trajectory`
    for computation of the diagrams of a
    :class:`~ks_shadowing.core.trajectory.KSTrajectory`.

    Attributes
    ----------
    diagrams : list[NDArray[np.float64]]
        One persistence diagram per timestep. Each diagram has shape
        ``(num_pairs, 2)`` of ``(birth, death)`` pairs with ``birth < death``.
        The essential class (infinite death) is excluded.
    dt : float
        The constant timestep of the trajectory between diagrams.
    """

    diagrams: list[NDArray[np.float64]]
    dt: float

    @classmethod
    def from_trajectory(
        cls,
        trajectory: KSTrajectory,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        order: int = 0,
    ) -> Self:
        r"""Compute the persistence diagrams of each point of ``trajectory``.

        This effectively passes ``trajectory`` to the space of persistence
        diagrams for the Persistent Homology Approach (PHA) for shadowing
        detection.

        Parameters
        ----------
        trajectory : :class:`~ks_shadowing.core.trajectory.KSTrajectory`
            Kuramoto-Sivashinsky trajectory to convert.
        chunk_size : int, optional
            Largest number of physical space timesteps of ``trajectory``
            manifested at once; controls memory usage. Default value is
            :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.
        order : int, optional
            Spatial-derivative order applied in Fourier space before the
            inverse FFT. The zeroth persistence diagram is then computed on
            the resulting field. ``0`` (default) leaves the field unchanged;
            higher orders compute :math:`\partial^{n} u / \partial x^{n}`
            first. Must be non-negative.
        """
        if order == 0:
            source = trajectory
        else:
            wavenumbers = 2.0 * np.pi * np.arange(trajectory.modes.shape[1]) / DOMAIN_SIZE
            multiplier = (1j * wavenumbers) ** order
            source = KSTrajectory(
                modes=trajectory.modes * multiplier,
                dt=trajectory.dt,
                resolution=trajectory.resolution,
            )

        diagrams: list[NDArray[np.float64]] = []
        for _, physical_chunk in source.chunks_physical(chunk_size):
            diagrams.extend(_zeroth_persistence_diagram_periodic(field) for field in physical_chunk)

        return cls(diagrams, source.dt)

    def __len__(self) -> int:
        """Number of persistence diagrams in the trajectory."""
        return len(self.diagrams)

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Iterate over persistence diagrams in trajectory order."""
        return iter(self.diagrams)

    def _flatten(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Flatten ``self.diagrams`` into a single flat array with an additional
        offset array for the batched Hera API.

        This is the expected format for the ``diagrams_a`` and ``offsets_a``
        arguments to the ``_wasserstein_column`` function.

        Returns
        -------
        flat_diagrams : NDArray[np.float64], shape (offsets[-1], 2)
            Contiguous, row-major array that is the row-wise concatenation of
            ``self.diagrams``.
        offsets : NDArray[np.int64], shape (len(self.diagrams) + 1,)
            Starting point of each diagram in the concatenated array; the final
            entry is equal to ``flat_diagrams.shape[0]``. If
            ``length_i = diagrams[i].shape[0]``, then
            ``offsets[i + 1] - offsets[i] = length_i``.
        """
        if not self.diagrams:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64)

        lengths = np.array([dgm.shape[0] for dgm in self.diagrams], dtype=np.int64)
        offsets = np.zeros(len(self.diagrams) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if offsets[-1] == 0:
            return np.zeros((0, 2), dtype=np.float64), offsets

        nonempty = [diagram for diagram in self.diagrams if diagram.shape[0] > 0]
        flat_diagrams = np.vstack(nonempty).astype(np.float64, copy=False)

        return np.ascontiguousarray(flat_diagrams), offsets


def _zeroth_persistence_diagram_periodic(field: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Compute sublevel-set persistence diagram for a 1D periodic field.

    Computes :math:`H_0` sublevel-set persistence on a circle (1D periodic
    domain). Each local minimum of the discrete field births a connected
    component in the sublevel set :math:`\{x : f(x) \le t\}`. When two
    components merge (at an entry between two distinct minima), the younger
    component dies. The resulting diagram is invariant to spatial translations
    of ``field``.

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

    ordered = np.argsort(field)

    # Union-find data
    parent = list(range(field.size))
    active = [False] * field.size
    pairs: list[tuple[float, float]] = []

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
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
            # Only add if the root is different from the left
            if not neighbor_roots or root_right != neighbor_roots[0]:
                neighbor_roots.append(root_right)

        if not neighbor_roots:
            pass  # new component
        elif len(neighbor_roots) == 1:
            parent[entry] = neighbor_roots[0]
        else:
            root_a, root_b = neighbor_roots

            # Component with lower birth survives
            if field[root_a] > field[root_b]:
                root_a, root_b = root_b, root_a
            death = float(field[entry])

            # Don't take any equal (birth, death) pairs
            if field[root_b] < death:
                pairs.append((field[root_b], death))

            # Union: younger and vertex attach to elder
            parent[root_b] = root_a
            parent[entry] = root_a

    if not pairs:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(pairs, dtype=np.float64)
