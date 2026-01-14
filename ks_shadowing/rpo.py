"""Relative Periodic Orbit (RPO) data loading."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RPO:
    """Relative Periodic Orbit data.

    Represents an unstable periodic orbit of the KS equation with spatial
    shift symmetry: u(x, t + T) = u(x - shift, t).

    # Fields

    - index: the index of this RPO in the data file it was loaded from.
    - fourier_coeffs: the initial fourier coefficients of this RPO in
    interleaved real/imaginary format.
    - period: The temporal period of the RPO.
    - time_steps: The total integration steps during the period.
    - spatial_shift: accumulated spatial shift of the RPO over one period.
    """

    index: int
    fourier_coeffs: NDArray[np.float64]
    period: float
    time_steps: int
    spatial_shift: float


def load_rpo(mat_path: Path, index: int) -> RPO:
    """Load a single RPO from a .mat file by index."""
    data = scipy.io.loadmat(mat_path)
    rpo_cell = data["rpo"][0]

    if index < 0 or index >= len(rpo_cell):
        raise IndexError(f"RPO index {index} out of range [0, {len(rpo_cell)})")

    return _parse_rpo_struct(rpo_cell[index], index)


def load_all_rpos(mat_path: Path) -> list[RPO]:
    """Load all RPOs from a .mat file."""
    data = scipy.io.loadmat(mat_path)
    rpo_cell = data["rpo"][0]

    return [_parse_rpo_struct(rpo_cell[i], i) for i in range(len(rpo_cell))]


def _parse_rpo_struct(rpo_data: np.ndarray, index: int) -> RPO:
    """Parse a single RPO struct from the array."""
    return RPO(
        index=index,
        fourier_coeffs=rpo_data[0][:, 0].astype(np.float64),
        period=float(rpo_data[1][0, 0]),
        time_steps=int(rpo_data[2][0, 0]),
        spatial_shift=float(rpo_data[4][0, 0]),
    )
