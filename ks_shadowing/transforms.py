"""FFT utilities and spatial transform operations."""

import numpy as np
from numpy.typing import NDArray
from scipy import fft


def interleaved_to_complex(coeffs: NDArray[np.floating]) -> NDArray[np.complex128]:
    """Convert interleaved real/imaginary coefficients to complex Fourier modes.

    Takes the 30-element interleaved format from the `ksint` function and
    returns 16 complex coefficients with zero-padding for the zeroth mode and
    the Nyquist (last) mode.

    Input:  [..., 30] -> [Re(a1), Im(a1), Re(a2), Im(a2), ..., Re(a15), Im(a15)]
    Output: [..., 16] -> [0, a1, a2, ..., a15, 0]
    """
    real_parts = coeffs[..., 0::2]
    imag_parts = coeffs[..., 1::2]
    modes = real_parts + 1j * imag_parts

    # Pad with zeros for mode 0 and Nyquist
    shape = coeffs.shape[:-1]
    zeros = np.zeros((*shape, 1), dtype=np.complex128)
    return np.concatenate([zeros, modes, zeros], axis=-1)


def to_physical(
    fourier_coeffs: NDArray[np.complex128],
    resolution: int,
) -> NDArray[np.float64]:
    """Transform Fourier coefficients to physical space via inverse rFFT.

    The output is scaled by the given spatial resolution for normalization.
    """
    return resolution * fft.irfft(fourier_coeffs, resolution, axis=-1)


def l2_distance_all_shifts(
    field_u: NDArray[np.float64],
    field_v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute L2 distance from `field_u` to all spatial shifts of `field_v`.

    Uses the identity ||u - S_phi(v)||^2 = ||u||^2 + ||v||^2 - 2<u, S_phi(v)>
    where the cross-correlation for all shifts is computed via FFT in O(N log N).

    Returns array of length N with distances for each shift index.
    """
    n = len(field_u)
    norm_u_sq = np.sum(field_u**2)
    norm_v_sq = np.sum(field_v**2)

    # Cross-correlation via FFT: IFFT(conj(FFT(u)) * FFT(v))
    u_fft = fft.rfft(field_u)
    v_fft = fft.rfft(field_v)
    cross_corr = fft.irfft(np.conj(u_fft) * v_fft, n)

    dist_sq = np.maximum(norm_u_sq + norm_v_sq - 2 * cross_corr, 0.0)
    return np.sqrt(dist_sq)
