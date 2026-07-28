r"""KS trajectory representation in spectral form.

Provides :class:`KSTrajectory`, a frozen dataclass storing KS field data
as 17 complex Fourier modes per timestep, and :func:`shift_distances_sq`,
which computes squared :math:`L_2` distances at all circular spatial shifts
using 17-mode FFT cross-correlation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import fft

from ks_shadowing.core.integrator import DOMAIN_SIZE, ksint

if TYPE_CHECKING:
    from ks_shadowing.core.rpo import RPO

DEFAULT_CHUNK_SIZE: int = 50000
"""Default number of trajectory timesteps to process at once.

Controls the memory-vectorization tradeoff for physical-space computations.
At resolution 2048, each chunk of 50000 steps uses approximately 780 MiB.
"""

_COMPLEX_MODES = 17
_MIN_RESOLUTION = 2 * (_COMPLEX_MODES - 1)


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
        if self.modes.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"modes must be 2-dimensional, got ndim={self.modes.ndim}")
        if self.modes.shape[1] != _COMPLEX_MODES:
            raise ValueError(f"modes must have {_COMPLEX_MODES} columns, got {self.modes.shape[1]}")

    @classmethod
    def from_initial_state(
        cls,
        initial_state: NDArray[np.complex128],
        dt: float,
        num_timesteps: int,
        resolution: int,
        save_interval: int = 1,
    ) -> Self:
        """Integrate the KS equation from an initial condition.

        Parameters
        ----------
        initial_state : NDArray[np.complex128], shape (17,)
            Complex Fourier modes for the initial condition.
        dt : float
            Integrator step in time units. Per-row spacing of the returned
            trajectory is ``dt * save_interval``.
        num_timesteps : int
            Length of the resulting trajectory (including the initial
            condition).
        resolution : int
            Number of physical-space grid points for inverse FFT.
        save_interval : int, optional
            Save every ``save_interval``-th integrated state. The integrator
            still steps at ``dt``; only the saved trajectory is coarsened. The
            resulting :attr:`dt` is ``dt * save_interval``. Default 1.

        Returns
        -------
        Self
            Trajectory with ``len(result) == num_timesteps`` and
            ``result.dt == dt * save_interval``.

        Raises
        ------
        ValueError
            If ``num_timesteps`` is less than 2.
        """
        if num_timesteps < 2:  # noqa: PLR2004
            raise ValueError(f"num_timesteps must be at least 2, got {num_timesteps}")

        integration_steps = (num_timesteps - 1) * save_interval
        modes = ksint(initial_state, dt, integration_steps, save_interval=save_interval)
        return cls(modes=modes, dt=dt * save_interval, resolution=resolution)

    @classmethod
    def from_rpo(
        cls,
        rpo: RPO,
        resolution: int,
        downsample: int = 1,
        native: bool = False,
    ) -> Self:
        """Build an RPO trajectory at sampling step ``downsample * rpo.dt``.

        Integrates ``rpo`` for one period at its native ``rpo.dt`` and
        produces the output in one of two variants:

        * ``native=False`` (default): keeps every ``downsample``-th row of
          the integration. Output length
          ``ceil(rpo.time_steps / downsample)``.

        * ``native=True``: visits every native phase by stepping through
          the stride-``downsample`` permutation, applying the relative-
          periodicity spatial-shift roll at each period wrap so the result
          is a valid forward KSE evolution at the coarser ``dt`` from
          ``rpo.modes``. Output length
          ``rpo.time_steps // gcd(downsample, rpo.time_steps)``.

        Both variants have ``dt = rpo.dt * downsample``. Defaults
        ``(downsample=1, native=False)`` give one full native period
        (``rpo.time_steps`` rows at ``dt = rpo.dt``).

        Parameters
        ----------
        rpo : :class:`~ks_shadowing.core.rpo.RPO`
            Source orbit. Integrated at its native ``rpo.dt`` to preserve
            relative-periodicity calibration.
        resolution : int
            Number of physical-space grid points for inverse FFT.
        downsample : int, optional
            Sampling stride. Default 1.
        native : bool, optional
            Selects the slicing or reordering variant described above.
            Default ``False``.

        Returns
        -------
        Self
            Trajectory with ``dt = rpo.dt * downsample``.

        Raises
        ------
        ValueError
            If ``downsample`` is less than 1.
        """
        if downsample < 1:
            raise ValueError(f"downsample must be positive, got {downsample}")

        integrated = ksint(rpo.modes, rpo.dt, rpo.time_steps - 1)

        if not native:
            modes = integrated[::downsample]
            return cls(modes=modes, dt=rpo.dt * downsample, resolution=resolution)

        cycle_length = rpo.time_steps // math.gcd(downsample, rpo.time_steps)
        indices = (np.arange(cycle_length) * downsample) % rpo.time_steps
        # period_wraps[r] counts how many full RPO periods the cumulative
        # native-row index r * downsample has passed. Multiplying mode k by
        # exp(+2pi*i*k*s/L) rolls the real-space signal by -s; rolling each row
        # by -period_wraps[r] * spatial_shift ensures the stored sequence is a
        # valid forward KSE evolution at dt_traj from integrated[0].
        period_wraps = (np.arange(cycle_length) * downsample) // rpo.time_steps
        wavenumbers = np.arange(_COMPLEX_MODES)
        shift_phase = np.exp(
            2j
            * np.pi
            * wavenumbers[np.newaxis, :]
            * period_wraps[:, np.newaxis]
            * rpo.spatial_shift
            / DOMAIN_SIZE
        )
        modes = integrated[indices] * shift_phase
        return cls(modes=modes, dt=rpo.dt * downsample, resolution=resolution)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist this trajectory's modes and ``dt`` to an HDF5 file.

        Resolution is not stored: it is an inverse-FFT grid parameter
        rather than a property of the 17 stored modes.

        Parameters
        ----------
        path : str or os.PathLike
            Destination ``.h5`` path. Parent directories are created if
            missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("modes", data=self.modes)
            f.attrs["dt"] = self.dt

    @classmethod
    def load(cls, path: str | os.PathLike[str], resolution: int) -> Self:
        """Load a trajectory from a file written by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            Source ``.h5`` file.
        resolution : int
            Number of physical-space grid points to associate with the
            loaded trajectory; controls the ``irfft`` grid in
            :meth:`to_physical`.

        Returns
        -------
        Self
            Trajectory reconstructed directly from stored modes; no
            re-integration is performed.
        """
        with h5py.File(path, "r") as f:
            modes = f["modes"][:]
            dt = float(f.attrs["dt"])
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

        Notes
        -----
        ``resolution`` below 32 truncates the higher-index Fourier modes
        during ``irfft``, silently corrupting the reconstructed field.
        """
        return self.resolution * fft.irfft(self.modes, self.resolution, axis=-1)

    def to_comoving(self, drift_rate: float, start_time: float = 0.0) -> Self:
        r"""Transform to co-moving frame given drift per unit time.

        Multiplies mode ``k`` at row ``r`` (``0 <= r < num_timesteps``) by
        :math:`\exp(-2 \pi i \cdot k \cdot \text{drift\_rate}
        \cdot (\text{start\_time} + r \cdot \text{self.dt}) / L)` where
        :math:`L` is the domain size. In physical space this is a circular roll
        by ``+drift_rate * t * resolution / L`` grid cells.

        Parameters
        ----------
        drift_rate : float
            Spatial drift per unit time, in domain-units per time-unit.
            Callers usually pass :attr:`~ks_shadowing.core.rpo.RPO.drift_rate`.
        start_time : float, optional
            Absolute time at row 0. Default 0.0.

        Returns
        -------
        Self
            New trajectory in the co-moving frame with the same ``dt``
            and ``resolution``.
        """
        wavenumbers = np.arange(_COMPLEX_MODES)  # (17,)
        times = start_time + np.arange(self.num_timesteps) * self.dt  # (T,)
        phase = (
            -2j
            * np.pi
            * wavenumbers[np.newaxis, :]
            * drift_rate
            * times[:, np.newaxis]
            / DOMAIN_SIZE
        )
        comoving_modes = self.modes * np.exp(phase)
        return type(self)(modes=comoving_modes, dt=self.dt, resolution=self.resolution)

    def __getitem__(self, key: slice) -> Self:
        """Slice along the timestep axis.

        Parameters
        ----------
        key : slice
            Timestep slice.

        Returns
        -------
        Self
            New trajectory containing the selected timesteps.
        """
        return type(self)(modes=self.modes[key], dt=self.dt, resolution=self.resolution)

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
            Default is :data:`DEFAULT_CHUNK_SIZE`.

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
            Default is :data:`DEFAULT_CHUNK_SIZE`.

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

    Raises
    ------
    ValueError
        If ``resolution`` is less than 32.
    """
    if resolution < _MIN_RESOLUTION:
        raise ValueError(
            f"resolution must be at least {_MIN_RESOLUTION}, got {resolution}: "
            "irfft crops the 17 stored modes below this size, corrupting the L2 distance"
        )

    norms_a = np.sum(np.abs(modes_a) ** 2, axis=-1)
    norms_b = np.sum(np.abs(modes_b) ** 2, axis=-1)
    cross_corr = fft.irfft(np.conj(modes_a) * modes_b, n=resolution, axis=-1)
    return (
        2 * resolution * (norms_a[:, np.newaxis] + norms_b[:, np.newaxis] - resolution * cross_corr)
    )


def _resolve_rpo_downsample(trajectory_dt: float, rpo: RPO) -> int:
    """Derive the per-RPO downsample stride implied by ``trajectory_dt``.

    Returns ``N = max(1, round(trajectory_dt / rpo.dt))``. Raises
    ``ValueError`` unless ``trajectory_dt`` is within 1.5% of ``rpo.dt * N``,
    i.e. the trajectory sampling must be an integer multiple of the orbit's
    native timestep.
    """
    downsample = max(1, round(trajectory_dt / rpo.dt))
    expected_dt = rpo.dt * downsample
    # 1.5%: passes per-orbit dt tuning (~1%), rejects between-grid sampling
    if not math.isclose(trajectory_dt, expected_dt, rel_tol=0.015):
        raise ValueError(
            f"trajectory dt={trajectory_dt} is not an integer multiple of rpo.dt="
            f"{rpo.dt} (nearest N={downsample} implies dt={expected_dt}); trajectory "
            f"sampling must be an integer multiple of the orbit's native timestep"
        )
    return downsample
