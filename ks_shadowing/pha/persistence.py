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

    Each timestep of a trajectory with constant timestep ``dt`` is represented
    by the sublevel-set persistence diagram of its physical-space field. The
    finite :math:`H_0` pairs of timestep ``i`` are ``diagrams[i]``; the two
    essential classes, which have infinite death, are recorded by their births
    in ``essential_births[i]``. Diagonal points are not included.

    Prefer
    :meth:`~ks_shadowing.pha.persistence.KSPersistenceTrajectory.from_trajectory`
    for computation of the diagrams of a
    :class:`~ks_shadowing.core.trajectory.KSTrajectory`.

    Attributes
    ----------
    diagrams : list[NDArray[np.float64]]
        One finite diagram per timestep. Each has shape ``(num_pairs, 2)`` of
        ``(birth, death)`` pairs with ``birth < death``.
    essential_births : NDArray[np.float64], shape (len(diagrams), 2)
        Births of the two essential classes per timestep: column 0 is the
        essential :math:`H_0` class (the field minimum), column 1 the
        essential :math:`H_1` class (the field maximum). Both have infinite
        death, so in a Wasserstein matching each pairs with its counterpart
        in the other diagram at cost equal to the birth difference.
    dt : float
        The constant timestep of the trajectory between diagrams.

    Raises
    ------
    ValueError
        If ``essential_births`` does not have shape ``(len(diagrams), 2)``.
    """

    diagrams: list[NDArray[np.float64]]
    essential_births: NDArray[np.float64]
    dt: float

    def __post_init__(self) -> None:
        expected = (len(self.diagrams), 2)
        if self.essential_births.shape != expected:
            raise ValueError(
                f"essential_births must have shape {expected}, got {self.essential_births.shape}"
            )

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
            inverse FFT. The persistence diagram is then computed on the
            resulting field. ``0`` (default) leaves the field unchanged;
            higher orders compute :math:`\partial^{n} u / \partial x^{n}`
            first. Must be non-negative.

        Returns
        -------
        Self
            Persistence trajectory with one diagram per timestep of
            ``trajectory``, at the same ``dt``.

        Raises
        ------
        ValueError
            If ``order`` is negative.
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

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
        essential_births: list[NDArray[np.float64]] = []
        for _, physical_chunk in source.chunks_physical(chunk_size):
            for field in physical_chunk:
                finite_pairs, births = _persistence_diagram_periodic(field)
                diagrams.append(finite_pairs)
                essential_births.append(births)

        return cls(
            diagrams,
            np.array(essential_births, dtype=np.float64).reshape(len(diagrams), 2),
            source.dt,
        )

    def __len__(self) -> int:
        """Number of persistence diagrams in the trajectory."""
        return len(self.diagrams)

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Iterate over the finite pairs of each timestep, in trajectory order."""
        return iter(self.diagrams)

    def __getitem__(self, key: slice | NDArray[np.int64]) -> Self:
        """Select timesteps.

        Parameters
        ----------
        key : slice | NDArray[np.int64]
            Timestep slice or 1D integer index array.

        Returns
        -------
        Self
            New persistence trajectory holding the selected timesteps'
            diagrams and essential births, at the same ``dt``.

        Raises
        ------
        TypeError
            If ``key`` is neither a slice nor an integer array.
        """
        if isinstance(key, slice):
            return type(self)(self.diagrams[key], self.essential_births[key], self.dt)
        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.integer):
            return type(self)(
                [self.diagrams[index] for index in key.tolist()],
                self.essential_births[key],
                self.dt,
            )
        raise TypeError("KSPersistenceTrajectory indices must be slices or integer arrays")

    def _flatten(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """Flatten ``self.diagrams`` into a single flat array with an additional
        offset array for the batched Hera API, alongside the essential births.

        This is the expected format for the ``diagrams_a``, ``offsets_a``, and
        ``essential_births_a`` arguments to the ``_wasserstein_column``
        function.

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
        essential_births : NDArray[np.float64], shape (len(self.diagrams), 2)
            ``self.essential_births``, contiguous.
        """
        essential_births = np.ascontiguousarray(self.essential_births, dtype=np.float64)
        if not self.diagrams:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64), essential_births

        lengths = np.array([diagram.shape[0] for diagram in self.diagrams], dtype=np.int64)
        offsets = np.zeros(len(self.diagrams) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if offsets[-1] == 0:
            return np.zeros((0, 2), dtype=np.float64), offsets, essential_births

        nonempty = [diagram for diagram in self.diagrams if diagram.shape[0] > 0]
        flat_diagrams = np.vstack(nonempty).astype(np.float64, copy=False)

        return np.ascontiguousarray(flat_diagrams), offsets, essential_births


def _persistence_diagram_periodic(
    field: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the sublevel-set persistence diagram of a 1D periodic field.

    Computes sublevel-set persistence on a circle (1D periodic domain). Each
    local minimum of the discrete field births a connected component in the
    sublevel set :math:`\{x : f(x) \le t\}`. When two components merge (at an
    entry between two distinct minima), the younger component dies, giving a
    finite :math:`H_0` pair. Two classes never die: the component born at the
    global minimum (essential :math:`H_0`), and the loop born when the global
    maximum closes the circle (essential :math:`H_1`). :math:`H_1` has no
    finite classes, so these two births complete the diagram. The diagram is
    invariant to spatial translations of ``field``.

    Parameters
    ----------
    field : NDArray[np.float64], shape (resolution,)
        Field values at grid points.

    Returns
    -------
    finite_pairs : NDArray[np.float64], shape (num_pairs, 2)
        Finite :math:`H_0` pairs ``(birth, death)`` with ``birth < death``.
    essential_births : NDArray[np.float64], shape (2,)
        Births of the essential classes: ``[min(field), max(field)]`` for
        :math:`H_0` and :math:`H_1` respectively. Both have infinite death.
        ``[nan, nan]`` for an empty field.
    """
    if field.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.full(2, np.nan, dtype=np.float64)

    ordered = np.argsort(field)
    # The sweep adds vertices in ascending order: the first births the
    # essential H0 class, the last closes the circle and births essential H1.
    essential_births = np.array([field[ordered[0]], field[ordered[-1]]], dtype=np.float64)

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
        return np.empty((0, 2), dtype=np.float64), essential_births
    return np.array(pairs, dtype=np.float64), essential_births
