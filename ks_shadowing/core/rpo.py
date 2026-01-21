"""Relative Periodic Orbit (RPO) data loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RPO:
    """Relative Periodic Orbit data.

    Represents an unstable periodic orbit of the KS equation with spatial
    shift symmetry: `u(x, t + T) = u(x - shift, t)`.

    All RPOs are for domain size `L = 22.0`. Each orbit has a native timestep
    `period / time_steps`, which is approximately equal to `dt = 0.02`.

    Attributes:
        index: Index of this RPO in the data file it was loaded from.
        fourier_coeffs: Initial Fourier coefficients in interleaved
            real/imaginary format (size 30).
        period: Temporal period of the RPO.
        time_steps: Number of integration steps in one period.
        spatial_shift: Accumulated spatial shift over one period.
    """

    index: int
    fourier_coeffs: NDArray[np.float64]
    period: float
    time_steps: int
    spatial_shift: float

    @classmethod
    def load(cls, path: Path, rpo_index: int) -> Self:
        """Load a single RPO from a .npz file by index."""
        data = np.load(path)
        rpo_count = len(data["periods"])

        if rpo_index < 0 or rpo_index >= rpo_count:
            raise IndexError(f"RPO index {rpo_index} out of range [0, {rpo_count})")

        return cls(
            index=rpo_index,
            fourier_coeffs=data["fourier_coeffs"][rpo_index].astype(np.float64),
            period=float(data["periods"][rpo_index]),
            time_steps=int(data["time_steps"][rpo_index]),
            spatial_shift=float(data["spatial_shifts"][rpo_index]),
        )


def load_all_rpos(path: Path) -> list[RPO]:
    """Load all RPOs from a .npz file."""
    data = np.load(path)
    rpo_count = len(data["periods"])

    return [
        RPO(
            index=rpo_index,
            fourier_coeffs=data["fourier_coeffs"][rpo_index].astype(np.float64),
            period=float(data["periods"][rpo_index]),
            time_steps=int(data["time_steps"][rpo_index]),
            spatial_shift=float(data["spatial_shifts"][rpo_index]),
        )
        for rpo_index in range(rpo_count)
    ]
