"""Relative periodic orbit (RPO) data structure and I/O.

A relative periodic orbit is an orbit of the Kuramoto-Sivashinsky equation that
is periodic under a certain spatial shift. RPO collections are stored in
``.npz`` files with one parallel array per field:

* ``fourier_coeffs``: initial conditions as 17-mode complex128 coefficients
* ``periods``: temporal periods
* ``time_steps``: numbers of integration steps per period
* ``spatial_shifts``: accumulated spatial shifts over one period

Use :func:`load_rpos` to load every RPO from a file in a single call.
"""

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RPO:
    r"""Relative periodic orbit (RPO) data.

    Represents an orbit of the Kuramoto-Sivashinsky equation with approximate
    spatial shift symmetry: :math:`u(x,\, t + T) = u(x + \phi,\, t)` where
    :math:`T` is the ``period`` and :math:`\phi` is the ``spatial_shift``.

    All RPOs are for domain size ``L = 22.0``. Each orbit has a native timestep
    ``period / time_steps``, which is approximately equal to the trajectory
    timestep :data:`~ks_shadowing.core.INTEGRATION_DT`.

    Attributes
    ----------
    index : int
        Index of this RPO in the data file it was loaded from.
    modes : NDArray[np.complex128], shape (17,)
        Initial Fourier coefficients as complex modes:
        ``[0, a_1, a_2, ..., a_15, 0]`` where mode 0 and the Nyquist
        mode (16) are zero.
    period : float
        Temporal period of the RPO.
    time_steps : int
        Number of integration steps in one period.
    spatial_shift : float
        Accumulated spatial shift over one period, in domain units, under
        ``u(x, t + T) = u(x + spatial_shift, t)``. Positive values mean the
        pattern translates toward decreasing ``x``.
    """

    index: int
    modes: NDArray[np.complex128]
    period: float
    time_steps: int
    spatial_shift: float

    @property
    def dt(self) -> float:
        """Native integration timestep ``period / time_steps``.

        Close to :data:`~ks_shadowing.core.INTEGRATION_DT` but tuned per
        orbit to maximize relative periodicity.
        """
        return self.period / self.time_steps

    @property
    def drift_rate(self) -> float:
        """Spatial drift per unit time, ``spatial_shift / period``.

        Returns
        -------
        float
            Drift rate in domain-units per time-unit. Independent of any
            sampling rate at which a trajectory representing the orbit
            is stored. A positive rate means the pattern translates toward
            decreasing ``x``; :meth:`~ks_shadowing.core.trajectory.KSTrajectory.to_comoving` at this
            rate cancels that motion.
        """
        return self.spatial_shift / self.period


def load_rpos(path: str | os.PathLike[str]) -> list[RPO]:
    """Load all RPOs from a .npz file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the .npz file containing RPO data.

    Returns
    -------
    list[RPO]
        All RPOs in the file, ordered by index.
    """
    with np.load(path) as data:
        return [
            RPO(
                index=rpo_index,
                modes=data["fourier_coeffs"][rpo_index].astype(np.complex128),
                period=float(data["periods"][rpo_index]),
                time_steps=int(data["time_steps"][rpo_index]),
                spatial_shift=float(data["spatial_shifts"][rpo_index]),
            )
            for rpo_index in range(len(data["periods"]))
        ]
