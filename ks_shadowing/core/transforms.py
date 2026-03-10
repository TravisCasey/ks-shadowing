"""FFT utilities and spatial transform operations."""

import numpy as np
from numpy.typing import NDArray
from scipy import fft


def interleaved_to_complex(interleaved: NDArray[np.floating]) -> NDArray[np.complex128]:
    r"""Convert interleaved real/imaginary coefficients to complex Fourier modes.

    Takes the 30-element interleaved format from
    :func:`~ks_shadowing.core.integrator.ksint` and returns 17 complex
    coefficients: zero for mode 0, modes 1-15, and zero for the Nyquist mode.

    Parameters
    ----------
    interleaved : NDArray[np.floating], shape (..., 30)
        Final axis are interleaved Fourier coefficients:
        :math:`[\operatorname{Re}(a_1),\, \operatorname{Im}(a_1),\, \dots,\,
        \operatorname{Re}(a_{15}),\, \operatorname{Im}(a_{15})]`.
        Leading axes are preserved.

    Returns
    -------
    NDArray[np.complex128], shape (..., 17)
        Complex Fourier modes: :math:`[0,\, a_1,\, a_2,\, \dots,\, a_{15},\, 0]`.
    """
    real_parts = interleaved[..., 0::2]
    imag_parts = interleaved[..., 1::2]
    modes = real_parts + 1j * imag_parts

    # Pad with zeros for mode 0 and Nyquist
    shape = interleaved.shape[:-1]
    zeros = np.zeros((*shape, 1), dtype=np.complex128)
    return np.concatenate([zeros, modes, zeros], axis=-1)


def to_physical(
    fourier_coeffs: NDArray[np.complexfloating],
    resolution: int,
) -> NDArray[np.float64]:
    """Transform Fourier coefficients to physical space via inverse rFFT.

    Parameters
    ----------
    fourier_coeffs : NDArray[np.complexfloating], shape (..., 17)
        Complex Fourier modes (e.g., from :func:`interleaved_to_complex`).
    resolution : int
        Number of grid points in the physical-space output.

    Returns
    -------
    NDArray[np.float64], shape (..., resolution)
        Physical-space field values, scaled by ``resolution`` for
        normalization.
    """
    return resolution * fft.irfft(fourier_coeffs, resolution, axis=-1)


def interleaved_to_physical(
    interleaved: NDArray[np.floating],
    resolution: int,
) -> NDArray[np.float64]:
    """Convert interleaved Fourier coefficients directly to physical space.

    Convenience wrapper combining :func:`interleaved_to_complex` and
    :func:`to_physical`.

    Parameters
    ----------
    interleaved : NDArray[np.floating], shape (..., 30)
        Interleaved Fourier coefficients from
        :func:`~ks_shadowing.core.integrator.ksint`.
    resolution : int
        Number of grid points in the physical-space output.

    Returns
    -------
    NDArray[np.float64], shape (..., resolution)
        Physical-space field values.
    """
    complex_coeffs = interleaved_to_complex(interleaved)
    return to_physical(complex_coeffs, resolution)


def to_comoving_frame(
    trajectory: NDArray[np.float64],
    drift_per_step: float,
) -> NDArray[np.float64]:
    """Transform trajectory to a co-moving reference frame.

    At timestep ``i``, the field is shifted left by ``drift_per_step * i`` grid
    cells (sub-grid accuracy via the Fourier shift theorem). In this frame, an
    RPO with the corresponding drift rate becomes truly periodic.

    Parameters
    ----------
    trajectory : NDArray[np.float64], shape (num_steps, resolution)
        Physical-space trajectory.
    drift_per_step : float
        Spatial drift rate in grid cells per timestep.

    Returns
    -------
    NDArray[np.float64], shape (num_steps, resolution)
        Trajectory in the co-moving frame.
    """
    step_count, resolution = trajectory.shape

    # Compute FFT of trajectory
    traj_fft = fft.rfft(trajectory, axis=-1)

    # Wavenumbers for phase shift: k = 0, 1, 2, ..., resolution//2
    mode_count = traj_fft.shape[-1]
    wavenumbers = np.arange(mode_count)

    # Fourier shift theorem: to shift by d grid cells to the left (i.e., f(x) -> f(x+d)),
    # multiply in Fourier space by exp(2pi*i*k*d/N).
    # At timestep i, we shift left by drift_per_step * i to undo rightward drift.
    timesteps = np.arange(step_count)[:, np.newaxis]
    phase_shift = np.exp(2j * np.pi * wavenumbers * drift_per_step * timesteps / resolution)

    # Apply phase shift and transform back
    traj_shifted_fft = traj_fft * phase_shift
    return fft.irfft(traj_shifted_fft, resolution, axis=-1)


def tile_periodic(field: NDArray[np.float64], target_length: int) -> NDArray[np.float64]:
    """Tile a periodic field along axis 0 to at least ``target_length``.

    Parameters
    ----------
    field : NDArray[np.float64], shape (period, ...)
        One full period of the signal.
    target_length : int
        Minimum desired length along axis 0.

    Returns
    -------
    NDArray[np.float64], shape (tiled_length, ...)
        Tiled array where ``tiled_length >= target_length``.
    """
    period = field.shape[0]
    if period >= target_length:
        return field

    tile_count = (target_length + period - 1) // period
    return np.tile(field, (tile_count,) + (1,) * (field.ndim - 1))
