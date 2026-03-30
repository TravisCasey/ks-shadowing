r"""KS trajectory representation in spectral form.

Provides :class:`KSTrajectory`, a frozen dataclass storing KS field data
as 17 complex Fourier modes per timestep, and :func:`shift_distances_sq`,
which computes squared :math:`L_2` distances at all circular spatial shifts
using 17-mode FFT cross-correlation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy import fft

from ks_shadowing.core.integrator import DOMAIN_SIZE, ksint

DEFAULT_CHUNK_SIZE: int = 50000
"""Default number of trajectory timesteps to process at once.

Controls the memory-vectorization tradeoff for physical-space computations.
At resolution 2048, each chunk of 50000 steps uses approximately 780 MiB.
"""

_COMPLEX_MODES = 17


@dataclass(frozen=True, slots=True)
class KSTrajectory:
    """A sequence of KS equation states in spectral representation.

    Stores 17 complex Fourier modes per timestep as the canonical form.
    Provides transformations to physical space, co-moving frame, and
    chunked iteration for memory-efficient processing of large data.

    Attributes
    ----------
    modes : NDArray[np.complex128], shape (num_timesteps, 17)
        Complex Fourier modes: ``[0, a_1, a_2, ..., a_15, 0]`` where
        modes 0 and 16 (Nyquist) are zero.
    dt : float
        Integration timestep in time units.
    resolution : int
        Number of physical-space grid points for inverse FFT.
    """

    modes: NDArray[np.complex128]
    dt: float
    resolution: int

    def __post_init__(self) -> None:
        """Validate modes array shape."""
        _expected_ndim = 2
        if self.modes.ndim != _expected_ndim:
            raise ValueError(f"modes must be 2-dimensional, got ndim={self.modes.ndim}")
        if self.modes.shape[1] != _COMPLEX_MODES:
            raise ValueError(f"modes must have {_COMPLEX_MODES} columns, got {self.modes.shape[1]}")

    @classmethod
    def from_initial_state(
        cls,
        initial_state: NDArray[np.complex128],
        dt: float,
        steps: int,
        resolution: int,
    ) -> Self:
        """Integrate the KS equation from an initial condition.

        Parameters
        ----------
        initial_state : NDArray[np.complex128], shape (17,)
            Complex Fourier modes for the initial condition.
        dt : float
            Integration timestep in time units.
        steps : int
            Length of the resulting trajectory (including the initial
            condition). Internally calls ``ksint(initial_state, dt, steps - 1)``.
        resolution : int
            Number of physical-space grid points for inverse FFT.

        Returns
        -------
        Self
            Trajectory with ``len(result) == steps``.
        """
        modes = ksint(initial_state, dt, steps - 1)
        return cls(modes=modes, dt=dt, resolution=resolution)

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps in the trajectory."""
        return self.modes.shape[0]

    def __len__(self) -> int:
        return self.num_timesteps

    def to_physical(self) -> NDArray[np.float64]:
        """Transform to physical space via inverse rFFT.

        Returns
        -------
        NDArray[np.float64], shape (num_timesteps, resolution)
            Physical-space field values, scaled by ``resolution`` for
            normalization.
        """
        return self.resolution * fft.irfft(self.modes, self.resolution, axis=-1)

    def to_comoving(self, drift_rate: float, start_step: int = 0) -> Self:
        r"""Transform to co-moving frame by phase-shifting Fourier modes.

        Multiplies mode ``k`` at timestep ``t`` by
        :math:`\exp(2 \pi i \cdot k \cdot \text{drift\_rate}
        \cdot (\text{start\_step} + t) / L)` where :math:`L` is the
        domain size.

        Parameters
        ----------
        drift_rate : float
            Spatial drift per timestep in domain units. Callers compute
            ``rpo.spatial_shift / rpo.time_steps``.
        start_step : int, optional
            Absolute timestep offset for the first row. Default 0.

        Returns
        -------
        Self
            New trajectory in the co-moving frame with the same ``dt``
            and ``resolution``.
        """
        wavenumbers = np.arange(_COMPLEX_MODES)  # (17,)
        timesteps = np.arange(start_step, start_step + self.num_timesteps)  # (T,)
        phase = (
            2j
            * np.pi
            * wavenumbers[np.newaxis, :]
            * drift_rate
            * timesteps[:, np.newaxis]
            / DOMAIN_SIZE
        )
        comoving_modes = self.modes * np.exp(phase)
        return type(self)(modes=comoving_modes, dt=self.dt, resolution=self.resolution)

    def tile(self, target_length: int) -> Self:
        """Tile modes periodically to at least ``target_length`` timesteps.

        Parameters
        ----------
        target_length : int
            Minimum desired number of timesteps in the result.

        Returns
        -------
        Self
            New trajectory with ``len(result) >= target_length``.
        """
        period = self.num_timesteps
        if period >= target_length:
            return type(self)(modes=self.modes, dt=self.dt, resolution=self.resolution)

        tile_count = math.ceil(target_length / period)
        tiled_modes = np.tile(self.modes, (tile_count, 1))
        return type(self)(modes=tiled_modes, dt=self.dt, resolution=self.resolution)

    def __getitem__(self, key: int | slice) -> Self:
        """Slice along the timestep axis.

        Integer indexing returns a single-timestep trajectory with shape
        ``(1, 17)`` to keep the 2D invariant.

        Parameters
        ----------
        key : int or slice
            Timestep index or slice.

        Returns
        -------
        Self
            New trajectory containing the selected timesteps.
        """
        sliced_modes = self.modes[key : key + 1] if isinstance(key, int) else self.modes[key]
        return type(self)(modes=sliced_modes, dt=self.dt, resolution=self.resolution)

    def chunks_physical(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[tuple[int, NDArray[np.float64]]]:
        """Yield ``(start_index, physical_chunk)`` pairs.

        Each ``physical_chunk`` has shape ``(chunk_len, resolution)``.
        Only the current chunk is materialized in memory.

        Parameters
        ----------
        chunk_size : int, optional
            Maximum number of timesteps per chunk.
            Default is :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

        Yields
        ------
        tuple[int, NDArray[np.float64]]
            ``(start_index, physical_chunk)`` where ``physical_chunk``
            has shape ``(chunk_len, resolution)``.
        """
        for start in range(0, self.num_timesteps, chunk_size):
            end = min(start + chunk_size, self.num_timesteps)
            chunk_modes = self.modes[start:end]
            physical = self.resolution * fft.irfft(chunk_modes, self.resolution, axis=-1)
            yield start, physical

    def chunks_fourier(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[tuple[int, NDArray[np.complex128]]]:
        """Yield ``(start_index, modes_chunk)`` pairs.

        Each ``modes_chunk`` has shape ``(chunk_len, 17)``.

        Parameters
        ----------
        chunk_size : int, optional
            Maximum number of timesteps per chunk.
            Default is :data:`~ks_shadowing.core.DEFAULT_CHUNK_SIZE`.

        Yields
        ------
        tuple[int, NDArray[np.complex128]]
            ``(start_index, modes_chunk)`` where ``modes_chunk`` has
            shape ``(chunk_len, 17)``.
        """
        for start in range(0, self.num_timesteps, chunk_size):
            end = min(start + chunk_size, self.num_timesteps)
            yield start, self.modes[start:end]


def shift_distances_sq(
    modes_a: NDArray[np.complex128],
    modes_b: NDArray[np.complex128],
    resolution: int,
) -> NDArray[np.float64]:
    r"""Squared L2 distances between physical-space fields at all circular shifts.

    For each timestep ``t``, computes
    :math:`\| u_t - \mathrm{roll}(v_t, -s) \|^2` for all shifts
    ``s`` in ``[0, resolution)``, where ``u`` and ``v`` are the
    physical-space fields corresponding to ``modes_a`` and ``modes_b``.

    Computes the cross-correlation via ``irfft`` of the 17-mode product
    ``conj(modes_a) * modes_b``. The ``2 * resolution`` and
    ``resolution`` factors follow from Parseval's theorem applied to the
    ``to_physical`` normalization convention
    (``physical = resolution * irfft(modes, resolution)``).

    Parameters
    ----------
    modes_a : NDArray[np.complex128], shape (T, 17)
        Complex Fourier modes for the first set of fields.
    modes_b : NDArray[np.complex128], shape (T, 17)
        Complex Fourier modes for the second set of fields.
    resolution : int
        Number of physical-space grid points (determines number of shifts).

    Returns
    -------
    NDArray[np.float64], shape (T, resolution)
        Squared L2 distance at each timestep and shift.
    """
    norms_a = np.sum(np.abs(modes_a) ** 2, axis=-1)
    norms_b = np.sum(np.abs(modes_b) ** 2, axis=-1)
    cross_corr = fft.irfft(np.conj(modes_a) * modes_b, n=resolution, axis=-1)
    return (
        2 * resolution * (norms_a[:, np.newaxis] + norms_b[:, np.newaxis] - resolution * cross_corr)
    )
