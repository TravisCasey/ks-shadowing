"""Wrapper for the C++ KS ETDRK4 integrator library."""

from ctypes import CDLL, POINTER, c_double, c_int
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Number of Fourier coefficients: modes 1-15, real and imaginary parts interleaved.
_N_COEFFS = 30


def load_library() -> CDLL:
    """Load libks2py.so from the package directory."""
    package_dir = Path(__file__).parent
    lib_path = package_dir / "libks2py.so"

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
    """Return the cached library singleton."""
    return load_library()


def ksint(
    initial_state: NDArray[np.floating],
    dt: float,
    steps: int,
    save_every: int = 1,
    domain_size: float = 22.0,
) -> NDArray[np.float64]:
    """Integrate KS equation in Fourier space using ETDRK4.

    Takes initial Fourier coefficients in interleaved format
    [Re(a_1), Im(a_1), ..., Re(a_15), Im(a_15)] and integrates forward in time.

    Returns trajectory of shape `(steps // save_every + 1, 30)` where row 0
    is the initial condition.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    if save_every > steps:
        raise ValueError(f"save_every ({save_every}) cannot exceed steps ({steps})")

    initial_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if initial_state.shape != (_N_COEFFS,):
        raise ValueError(f"initial_state must have shape ({_N_COEFFS},), got {initial_state.shape}")

    lib = get_lib()

    n_out = steps // save_every + 1
    # Eigen uses column-major: C++ returns (_N_COEFFS, n_out)
    trajectory = np.empty((_N_COEFFS, n_out), dtype=np.float64, order="F")

    lib.ksf(
        trajectory.ctypes.data_as(POINTER(c_double)),
        initial_state.ctypes.data_as(POINTER(c_double)),
        c_double(domain_size),
        c_double(dt),
        c_int(steps),
        c_int(save_every),
    )

    return trajectory.T
