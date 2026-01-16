"""FFT utilities and spatial transform operations."""

import numpy as np
from numpy.typing import NDArray
from scipy import fft


def interleaved_to_complex(interleaved: NDArray[np.floating]) -> NDArray[np.complex128]:
    """Convert interleaved real/imaginary coefficients to complex Fourier modes.

    Takes the 30-element interleaved format from the `ksint` function and
    returns 17 complex coefficients: a zero for mode 0, modes 1-15, and a zero
    for the Nyquist mode.

    Input:  [..., 30] -> [Re(a1), Im(a1), Re(a2), Im(a2), ..., Re(a15), Im(a15)]
    Output: [..., 17] -> [0, a1, a2, ..., a15, 0]
    """
    real_parts = interleaved[..., 0::2]
    imag_parts = interleaved[..., 1::2]
    modes = real_parts + 1j * imag_parts

    # Pad with zeros for mode 0 and Nyquist
    shape = interleaved.shape[:-1]
    zeros = np.zeros((*shape, 1), dtype=np.complex128)
    return np.concatenate([zeros, modes, zeros], axis=-1)


def to_physical(
    fourier_coeffs: NDArray[np.complex128],
    resolution: int,
) -> NDArray[np.float64]:
    """Transform Fourier coefficients to physical space via inverse rFFT.

    The output is scaled by the given spatial resolution for normalization.
    Accepts any array dimension of at least one, and the FFT is computed along
    the last axis.
    """
    return resolution * fft.irfft(fourier_coeffs, resolution, axis=-1)


def interleaved_to_physical(
    interleaved: NDArray[np.floating],
    resolution: int,
) -> NDArray[np.float64]:
    """Convert interleaved Fourier coefficients directly to physical space.

    Combines `interleaved_to_complex` and `to_physical` into a single operation
    for convenience when working with trajectories from `ksint`.

    Args:
        interleaved: Coefficients in interleaved format, shape `(..., 30)`.
        resolution: Spatial resolution for the physical space output.

    Returns:
        Physical space representation with shape `(..., resolution)`.
    """
    complex_coeffs = interleaved_to_complex(interleaved)
    return to_physical(complex_coeffs, resolution)


def l2_distance_all_shifts(
    field_u: NDArray[np.float64],
    field_v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute L2 distance from `field_u` to all spatial shifts of `field_v`.

    Uses the identity ||u - S_phi(v)||^2 = ||u||^2 + ||v||^2 - 2<u, S_phi(v)>
    where the cross-correlation for all shifts is computed via FFT in O(N log N).

    Supports batched computation: if field_v has shape (M, N), computes distances
    for each of the M fields against field_u, returning shape (M, N).

    Args:
        field_u: Shape (N,) - single reference field
        field_v: Shape (N,) or (M, N) - one or more fields to compare

    Returns:
        Shape (N,) if field_v is 1D, or (M, N) if field_v is 2D.
        Each row contains distances for all N spatial shifts.
    """
    n = field_u.shape[-1]
    norm_u_sq = np.sum(field_u**2)
    norm_v_sq = np.sum(field_v**2, axis=-1, keepdims=True)

    # Cross-correlation via FFT: IFFT(conj(FFT(u)) * FFT(v))
    u_fft = fft.rfft(field_u)
    v_fft = fft.rfft(field_v, axis=-1)
    cross_corr = fft.irfft(np.conj(u_fft) * v_fft, n, axis=-1)

    dist_sq = np.maximum(norm_u_sq + norm_v_sq - 2 * cross_corr, 0.0)
    return np.sqrt(dist_sq)


def to_comoving_frame(
    trajectory: NDArray[np.float64],
    drift_per_step: float,
) -> NDArray[np.float64]:
    """Transform trajectory to a co-moving reference frame.

    Applies a cumulative spatial rotation to each timestep: at timestep `i`, the
    field is rotated by `-drift_per_step * i` grid cells. This transforms the
    trajectory into a frame moving at constant velocity, where an RPO with the
    corresponding drift rate becomes truly periodic.

    Args:
        trajectory: Physical space trajectory of shape `(num_steps, resolution)`.
        drift_per_step: Spatial drift rate in grid cells per timestep.

    Returns:
        Transformed trajectory with same shape as input.
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
    """Tile a periodic field along axis 0 to at least `target_length`.

    Concatenates copies of the input array until the first axis has length
    no less than `target_length`. The input is assumed to represent one period
    of a periodic signal.

    Args:
        field: Array of shape `(period, ...)` representing one period.
        target_length: Minimum desired length along axis 0.

    Returns:
        Tiled array of shape `(tiled_length, ...)` where
            `tiled_length >= target_length`.
    """
    period = field.shape[0]
    if period >= target_length:
        return field

    tile_count = (target_length + period - 1) // period
    return np.tile(field, (tile_count,) + (1,) * (field.ndim - 1))
