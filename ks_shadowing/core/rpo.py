"""Relative periodic orbit (RPO) data structure and I/O.

A relative periodic orbit is an orbit of the Kuramoto-Sivashinsky equation that
is periodic under a certain spatial shift.

RPO data is saved in npz files, with one array for each field:
  - ``fourier_coeffs``: initial condition as 17-mode complex128 coefficients
  - ``periods``: temporal period
  - ``time_steps``: number of time steps in a period
  - ``spatial_shifts``: accumulated spatial shift over one period.

An RPO can be loaded individually by referencing its index, or all RPOs can be
loaded simultaneously from a file with :func:`load_all_rpos`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RPO:
    r"""Relative periodic orbit (RPO) data.

    Represents an orbit of the Kuramoto-Sivashinsky equation with approximate
    spatial shift symmetry: :math:`u(x,\, t + T) = u(x - \phi,\, t)` where
    :math:`T` is the ``period`` and :math:`\phi` is the ``spatial_shift``.

    All RPOs are for domain size ``L = 22.0``. Each orbit has a native timestep
    ``period / time_steps``, which is approximately equal to ``dt = 0.02``.

    Attributes
    ----------
    index : int
        Index of this RPO in the data file it was loaded from.
    fourier_coeffs : NDArray[np.complex128], shape (17,)
        Initial Fourier coefficients as complex modes:
        ``[0, a_1, a_2, ..., a_15, 0]`` where mode 0 and the Nyquist
        mode (16) are zero.
    period : float
        Temporal period of the RPO.
    time_steps : int
        Number of integration steps in one period.
    spatial_shift : float
        Accumulated spatial shift over one period.
    """

    index: int
    fourier_coeffs: NDArray[np.complex128]
    period: float
    time_steps: int
    spatial_shift: float

    @classmethod
    def load(cls, path: Path, rpo_index: int) -> Self:
        """Load a single RPO from a .npz file by index.

        Parameters
        ----------
        path : Path
            Path to the .npz file containing RPO data.
        rpo_index : int
            Zero-based index of the RPO to load.

        Returns
        -------
        Self
            The loaded RPO.

        Raises
        ------
        IndexError
            If ``rpo_index`` is out of range for the file.
        """
        data = np.load(path)
        rpo_count = len(data["periods"])

        if rpo_index < 0 or rpo_index >= rpo_count:
            raise IndexError(
                f"RPO index {rpo_index} out of range [0, {rpo_count}) in RPO file at {path}"
            )

        return cls(
            index=rpo_index,
            fourier_coeffs=data["fourier_coeffs"][rpo_index].astype(np.complex128),
            period=float(data["periods"][rpo_index]),
            time_steps=int(data["time_steps"][rpo_index]),
            spatial_shift=float(data["spatial_shifts"][rpo_index]),
        )


def load_all_rpos(path: Path) -> list[RPO]:
    """Load all RPOs from a .npz file.

    Parameters
    ----------
    path : Path
        Path to the .npz file containing RPO data.

    Returns
    -------
    list[RPO]
        All RPOs in the file, ordered by index.
    """
    data = np.load(path)
    rpo_count = len(data["periods"])

    return [
        RPO(
            index=rpo_index,
            fourier_coeffs=data["fourier_coeffs"][rpo_index].astype(np.complex128),
            period=float(data["periods"][rpo_index]),
            time_steps=int(data["time_steps"][rpo_index]),
            spatial_shift=float(data["spatial_shifts"][rpo_index]),
        )
        for rpo_index in range(rpo_count)
    ]
