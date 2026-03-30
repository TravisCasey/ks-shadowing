r"""Wrapper for the C++ KS ETDRK4 integrator library.

Orbits of the Kuramoto-Sivashinsky equation are integrated with a C++
implementation of the
`ETDRK4 <https://epubs.siam.org/doi/10.1137/S1064827502410633>`_ method. This
module loads the C++ shared object and exposes the integrating function :func:`ksint`
with Python types and input validation.

The only domain size for the KS equation considered in this project is 22. The
integrator operates on 17-element complex Fourier coefficient arrays where mode 0
and the Nyquist mode (index 16) are always zero:

.. math::

   [0,\, a_1,\, a_2,\, \dots,\, a_{15},\, 0]

Internally, the C++ library uses a 30-element interleaved real format. The
conversion between complex and interleaved representations is handled inside
:func:`ksint`, so callers work exclusively with complex arrays.
"""

from ctypes import CDLL, POINTER, c_double, c_int
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Spatial domain size for the Kuramoto-Sivashinsky equation.
DOMAIN_SIZE = 22.0
# Number of complex Fourier modes (0, modes 1-15, Nyquist).
_COMPLEX_MODES = 17
# Number of Fourier coefficients in the interleaved format used by the C library:
# modes 1-15 real/imaginary parts interleaved.
_INTERLEAVED_COEFFS = 30


def _interleaved_to_complex(interleaved: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Convert interleaved real/imaginary coefficients to complex Fourier modes.

    Parameters
    ----------
    interleaved : NDArray[np.float64], shape (..., 30)
        Interleaved Fourier coefficients.

    Returns
    -------
    NDArray[np.complex128], shape (..., 17)
        Complex Fourier modes ``[0, a_1, ..., a_15, 0]``.
    """
    real_parts = interleaved[..., 0::2]
    imag_parts = interleaved[..., 1::2]
    modes = real_parts + 1j * imag_parts

    shape = interleaved.shape[:-1]
    zeros = np.zeros((*shape, 1), dtype=np.complex128)
    return np.concatenate([zeros, modes, zeros], axis=-1)


def _complex_to_interleaved(modes: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert 17-mode complex Fourier coefficients to interleaved format.

    Parameters
    ----------
    modes : NDArray[np.complex128], shape (..., 17)
        Complex Fourier modes ``[0, a_1, ..., a_15, 0]``.

    Returns
    -------
    NDArray[np.float64], shape (..., 30)
        Interleaved format ``[Re(a_1), Im(a_1), ..., Re(a_15), Im(a_15)]``.
    """
    active_modes = modes[..., 1:16]  # modes 1-15, shape (..., 15)
    interleaved = np.empty((*modes.shape[:-1], _INTERLEAVED_COEFFS), dtype=np.float64)
    interleaved[..., 0::2] = active_modes.real
    interleaved[..., 1::2] = active_modes.imag
    return interleaved


def _load_library() -> CDLL:
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
def _get_lib() -> CDLL:
    """Return the cached library singleton. See :func:`_load_library`."""
    return _load_library()


def ksint(
    initial_state: NDArray[np.complex128],
    dt: float,
    steps: int,
    save_every: int = 1,
) -> NDArray[np.complex128]:
    r"""Integrate KS equation in Fourier space using
    `ETDRK4 <https://epubs.siam.org/doi/10.1137/S1064827502410633>`_.

    Parameters
    ----------
    initial_state : NDArray[np.complex128], shape (17,)
        Complex Fourier modes:
        :math:`[0,\, a_1,\, a_2,\, \dots,\, a_{15},\, 0]`.
    dt : float
        Integration timestep in time units.
    steps : int
        Number of integration steps.
    save_every : int, optional
        Step between saved trajectory points. Default is 1 (save all).

    Returns
    -------
    NDArray[np.complex128], shape (steps // save_every + 1, 17)
        Trajectory in complex Fourier format. Row 0 is the initial
        condition.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    if save_every > steps:
        raise ValueError(f"save_every ({save_every}) cannot exceed steps ({steps})")

    initial_state = np.asarray(initial_state, dtype=np.complex128)
    if initial_state.shape != (_COMPLEX_MODES,):
        raise ValueError(
            f"initial_state must have shape ({_COMPLEX_MODES},), got {initial_state.shape}"
        )

    # Convert to interleaved format for the C library.
    interleaved_input = _complex_to_interleaved(initial_state)
    interleaved_input = np.ascontiguousarray(interleaved_input, dtype=np.float64)

    lib = _get_lib()

    # Eigen stores matrices column-major (Fortran, order = "F"), so each
    # timestep is one column of a (30, num_saved) matrix.
    num_saved = steps // save_every + 1
    trajectory = np.empty((_INTERLEAVED_COEFFS, num_saved), dtype=np.float64, order="F")

    lib.ksf(
        trajectory.ctypes.data_as(POINTER(c_double)),
        interleaved_input.ctypes.data_as(POINTER(c_double)),
        c_double(DOMAIN_SIZE),
        c_double(dt),
        c_int(steps),
        c_int(save_every),
    )

    # Transpose to (num_saved, 30) so trajectory[i] gives timestep i. This is a
    # zero-copy view: the Fortran-ordered columns become C-ordered rows.
    # Then convert from interleaved to complex format.
    return _interleaved_to_complex(trajectory.T)
