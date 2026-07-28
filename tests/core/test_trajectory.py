"""Tests for KSTrajectory and shift_distances_sq."""

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import fft

from ks_shadowing.core.integrator import DOMAIN_SIZE, ksint
from ks_shadowing.core.rpo import load_rpos
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq


def test_post_init_rejects_invalid_modes() -> None:
    """``__post_init__`` raises ``ValueError`` for ``modes`` arrays that are
    not 2D or do not have 17 columns."""
    with pytest.raises(ValueError, match="2-dimensional"):
        KSTrajectory(modes=np.zeros(17, dtype=np.complex128), dt=0.02, resolution=64)
    with pytest.raises(ValueError, match="17 columns"):
        KSTrajectory(modes=np.zeros((5, 10), dtype=np.complex128), dt=0.02, resolution=64)


def test_from_initial_state_length(sample_initial_state: NDArray[np.complex128]) -> None:
    """``from_initial_state(num_timesteps=N)`` produces a trajectory of
    length ``N``."""
    result = KSTrajectory.from_initial_state(
        sample_initial_state, dt=0.02, num_timesteps=30, resolution=64
    )
    assert len(result) == 30


def test_to_physical_parseval(random_trajectory: KSTrajectory) -> None:
    r"""``to_physical`` satisfies Parseval's identity:
    :math:`\sum_x |u(x)|^2 = 2R \sum_k |\hat{u}_k|^2`, where :math:`R` is
    ``resolution``."""
    physical = random_trajectory.to_physical()
    physical_energy = np.sum(physical**2)
    spectral_energy = (
        2 * random_trajectory.resolution * np.sum(np.abs(random_trajectory.modes) ** 2)
    )
    np.testing.assert_allclose(physical_energy, spectral_energy, rtol=1e-6, atol=1e-6)


def test_to_comoving_round_trip(random_trajectory: KSTrajectory) -> None:
    """Applying ``to_comoving`` with opposite ``drift_rate`` values recovers
    the original ``modes``."""
    drift = 0.35
    forward = random_trajectory.to_comoving(drift_rate=drift)
    recovered = forward.to_comoving(drift_rate=-drift)
    np.testing.assert_allclose(recovered.modes, random_trajectory.modes, atol=1e-6)


def test_to_comoving_start_time_offset(random_trajectory: KSTrajectory) -> None:
    """``to_comoving(d, start_time=t)`` on ``random_trajectory[k:]`` (where
    ``t`` is the time at row ``k``) matches rows ``[k:]`` of
    ``to_comoving(d, start_time=0)`` on the full trajectory."""
    drift = 0.27
    offset = 5
    offset_time = offset * random_trajectory.dt
    full = random_trajectory.to_comoving(drift_rate=drift, start_time=0.0)
    sliced = random_trajectory[offset:].to_comoving(drift_rate=drift, start_time=offset_time)
    np.testing.assert_allclose(sliced.modes, full.modes[offset:], atol=1e-12)


def test_to_comoving_closes_rpo_period(rpo_data_path: Path) -> None:
    """``to_comoving`` at ``RPO.drift_rate`` makes an RPO periodic."""
    rpo = load_rpos(rpo_data_path)[0]
    trajectory = KSTrajectory.from_initial_state(
        rpo.modes, rpo.dt, rpo.time_steps + 1, resolution=64
    )
    comoving = trajectory.to_comoving(rpo.drift_rate)
    np.testing.assert_allclose(comoving.modes[-1], comoving.modes[0], rtol=1e-6, atol=1e-6)


def test_to_comoving_rolls_physical_field(rng: np.random.Generator) -> None:
    """``to_comoving`` shifts the physical field right by the drift cell count,
    pinning the frame's absolute roll direction against ``to_physical``/``np.roll``.

    Independent of RPO data: mode ``k`` at row 1 is multiplied by
    ``exp(-2j*pi*k*drift_rate*dt/L)``, and ``irfft`` reconstructs
    ``physical(x)`` from term ``exp(2j*pi*k*x/L)``, so this phase is equivalent
    to evaluating the lab-frame field at ``x - cells`` grid points, i.e.
    ``np.roll(lab_physical, +cells)``.
    """
    resolution = 64
    cells = 3
    drift_rate = DOMAIN_SIZE * cells / resolution
    dt = 1.0

    modes = np.zeros((2, 17), dtype=np.complex128)
    modes[0, 1:16] = (rng.standard_normal(15) + 1j * rng.standard_normal(15)) * 0.1
    modes[1] = modes[0]
    trajectory = KSTrajectory(modes=modes, dt=dt, resolution=resolution)

    comoving = trajectory.to_comoving(drift_rate=drift_rate)
    lab_physical = trajectory.to_physical()
    comoving_physical = comoving.to_physical()

    expected = np.roll(lab_physical[1], +cells)
    np.testing.assert_allclose(comoving_physical[1], expected, rtol=1e-6, atol=1e-6)


def test_shift_distances_sq_matches_brute_force(rng: np.random.Generator) -> None:
    r"""``shift_distances_sq`` produces zero self-distance at shift 0 and
    matches a real-space ``np.roll`` reference for
    :math:`\| u_t - \mathrm{roll}(v_t, -s) \|^2` at every shift."""
    resolution = 32
    modes_a = np.zeros((4, 17), dtype=np.complex128)
    modes_b = np.zeros((4, 17), dtype=np.complex128)
    modes_a[:, 1:16] = (rng.standard_normal((4, 15)) + 1j * rng.standard_normal((4, 15))) * 0.1
    modes_b[:, 1:16] = (rng.standard_normal((4, 15)) + 1j * rng.standard_normal((4, 15))) * 0.1

    distances = shift_distances_sq(modes_a, modes_b, resolution)
    self_distances = shift_distances_sq(modes_a, modes_a, resolution)
    np.testing.assert_allclose(self_distances[:, 0], 0.0, atol=1e-6)

    physical_a = resolution * fft.irfft(modes_a, resolution, axis=-1)
    physical_b = resolution * fft.irfft(modes_b, resolution, axis=-1)
    expected = np.empty((4, resolution), dtype=np.float64)
    for shift in range(resolution):
        diff = physical_a - np.roll(physical_b, -shift, axis=-1)
        expected[:, shift] = np.sum(diff**2, axis=-1)

    np.testing.assert_allclose(distances, expected, rtol=1e-6, atol=1e-6)


def test_chunks_physical_reconstructs(random_trajectory: KSTrajectory) -> None:
    """Concatenating all chunks from ``chunks_physical`` reproduces
    ``to_physical`` and yields evenly-spaced ``start_index`` values."""
    chunks = list(random_trajectory.chunks_physical(chunk_size=7))
    assert [start for start, _ in chunks] == [0, 7, 14]
    reconstructed = np.vstack([chunk for _, chunk in chunks])
    expected = random_trajectory.to_physical()
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-6, atol=1e-6)


def test_trajectory_roundtrip_preserves_modes_and_dt(
    random_trajectory: KSTrajectory, tmp_path: Path
) -> None:
    """``KSTrajectory.save`` / ``load`` preserves ``modes`` and ``dt``;
    ``resolution`` is supplied at load time, not stored on disk."""
    path = tmp_path / "trajectory.h5"

    random_trajectory.save(path)
    loaded = KSTrajectory.load(path, resolution=128)

    np.testing.assert_array_equal(loaded.modes, random_trajectory.modes)
    assert loaded.dt == random_trajectory.dt
    assert loaded.resolution == 128


def test_from_rpo_reorders_when_native(rpo_data_path: Path) -> None:
    """``from_rpo(native=True, downsample=k)`` produces a trajectory whose
    comoving frame at row j matches the native trajectory's comoving frame
    at native phase ``(j * k) mod rpo.time_steps``, with length equal to
    the cycle length of that permutation and ``dt = rpo.dt * k``."""
    rpo = load_rpos(rpo_data_path)[0]
    downsample = 23

    native_only = KSTrajectory.from_rpo(rpo, resolution=64, downsample=1, native=False)
    reordered = KSTrajectory.from_rpo(rpo, resolution=64, downsample=downsample, native=True)

    assert reordered.dt == pytest.approx(rpo.dt * downsample)

    cycle_length = rpo.time_steps // math.gcd(downsample, rpo.time_steps)
    assert reordered.num_timesteps == cycle_length

    expected_indices = (np.arange(cycle_length) * downsample) % rpo.time_steps
    native_comoving = native_only.to_comoving(rpo.drift_rate)
    reordered_comoving = reordered.to_comoving(rpo.drift_rate)
    np.testing.assert_allclose(
        reordered_comoving.modes,
        native_comoving.modes[expected_indices],
        rtol=1e-12,
        atol=1e-12,
    )


def test_from_rpo_native_is_forward_evolution(rpo_data_path: Path) -> None:
    """``from_rpo(native=True)`` produces a row that is the forward KSE
    evolution of the previous row across a native-period wrap.
    """
    rpo = load_rpos(rpo_data_path)[1]
    downsample = 23
    assert math.gcd(downsample, rpo.time_steps) == 1

    reordered = KSTrajectory.from_rpo(rpo, resolution=64, downsample=downsample, native=True)

    wrap_row = next(
        r
        for r in range(1, reordered.num_timesteps)
        if (r * downsample) // rpo.time_steps != ((r - 1) * downsample) // rpo.time_steps
    )

    predicted = ksint(reordered.modes[wrap_row - 1], rpo.dt, downsample)[-1]
    np.testing.assert_allclose(reordered.modes[wrap_row], predicted, rtol=1e-4, atol=1e-3)


def test_from_rpo_slices_when_not_native(rpo_data_path: Path) -> None:
    """``from_rpo(native=False, downsample=k)`` returns every kth row of the
    one-period native integration, with ``dt = rpo.dt * k``."""
    rpo = load_rpos(rpo_data_path)[0]
    downsample = 23

    native_only = KSTrajectory.from_rpo(rpo, resolution=64, downsample=1, native=False)
    sliced = KSTrajectory.from_rpo(rpo, resolution=64, downsample=downsample, native=False)

    assert sliced.dt == pytest.approx(rpo.dt * downsample)
    np.testing.assert_allclose(
        sliced.modes, native_only.modes[::downsample], rtol=1e-12, atol=1e-12
    )
