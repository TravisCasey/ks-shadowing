r"""Wrapper for the C++ KS ETDRK4 integrator library.

Orbits of the Kuramoto-Sivashinsky equation are integrated with a C++
implementation of the
`ETDRK4 <https://epubs.siam.org/doi/10.1137/S1064827502410633>`_ method. This
module loads the C++ shared object and exposes the integrating function :func:`ksint`
with Python types and input validation.

The only domain size for the KS equation considered in this project is 22. The
integrator operates on floating point arrays of complex Fourier coefficients
with 17 modes, though the first and last (Nyquist mode) are zero. The remaining
15 modes are interleaved to form 30-length floating point arrays:

.. math::

   [0,\, a_1,\, a_2,\, \dots,\, a_{15},\, 0]
   \;\longleftrightarrow\;
   [\operatorname{Re}(a_1),\, \operatorname{Im}(a_1),\,
    \operatorname{Re}(a_2),\, \operatorname{Im}(a_2),\, \dots,\,
    \operatorname{Re}(a_{15}),\, \operatorname{Im}(a_{15})]
"""

from ctypes import CDLL, POINTER, c_double, c_int
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Spatial domain size for the Kuramoto-Sivashinsky equation.
DOMAIN_SIZE = 22.0
# Number of Fourier coefficients in the interleaved format: modes 1-15
# real/imaginary parts interleaved.
INTERLEAVED_COEFFS = 30


def load_library() -> CDLL:
    """Load the integrated shared object from the package directory."""
    lib_path = Path(__file__).parent / "libks2py.so"
    if not lib_path.exists():
        raise RuntimeError(f"Could not find libks2py.so at {lib_path}.")
    lib = CDLL(str(lib_path))

    lib.ksf.argtypes = [
        POINTER(c_double),  # out_trajectory
        POINTER(c_double),  # initial_state
        c_double,  # domain_size
        c_double,  # time_step
        c_int,  # num_steps
        c_int,  # save_interval
    ]
    lib.ksf.restype = None

    return lib


@cache
def get_lib() -> CDLL:
    """Return the cached library singleton. See :func:`load_library`."""
    return load_library()


def ksint(
    initial_state: NDArray[np.floating],
    dt: float,
    steps: int,
    save_every: int = 1,
) -> NDArray[np.float64]:
    r"""Integrate KS equation in Fourier space using
    `ETDRK4 <https://epubs.siam.org/doi/10.1137/S1064827502410633>`_.

    Parameters
    ----------
    initial_state : NDArray[np.floating], shape (30,)
        Fourier coefficients in interleaved format:
        :math:`[\operatorname{Re}(a_1),\, \operatorname{Im}(a_1),\, \dots,\,
        \operatorname{Re}(a_{15}),\, \operatorname{Im}(a_{15})]`.
    dt : float
        Integration timestep in time units.
    steps : int
        Number of integration steps.
    save_every : int, optional
        Step between saved trajectory points. Default is 1 (save all).

    Returns
    -------
    NDArray[np.float64], shape (steps // save_every + 1, 30)
        Trajectory in interleaved Fourier format. Row 0 is the initial
        condition.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    if save_every > steps:
        raise ValueError(f"save_every ({save_every}) cannot exceed steps ({steps})")

    # Ensure contiguous float64 array for the raw pointer passed to C++ lib.
    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.shape != (INTERLEAVED_COEFFS,):
        raise ValueError(
            f"initial_state must have shape ({INTERLEAVED_COEFFS},), got {initial_state.shape}"
        )

    lib = get_lib()

    # Eigen stores matrices column-major (Fortran, order = "F"), so each
    # timestep is one column of a (30, num_saved) matrix.
    num_saved = steps // save_every + 1
    trajectory = np.empty((INTERLEAVED_COEFFS, num_saved), dtype=np.float64, order="F")

    lib.ksf(
        trajectory.ctypes.data_as(POINTER(c_double)),
        initial_state.ctypes.data_as(POINTER(c_double)),
        c_double(DOMAIN_SIZE),
        c_double(dt),
        c_int(steps),
        c_int(save_every),
    )

    # Transpose to (num_saved, 30) so trajectory[i] gives timestep i. This is a
    # zero-copy view: the Fortran-ordered columns become C-ordered rows.
    return trajectory.T
