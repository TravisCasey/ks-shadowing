"""Tests for KSTrajectory and shift_distances_sq."""

from pathlib import Path

import numpy as np
import pytest
from scipy import fft

from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq


def test_post_init_rejects_invalid_modes() -> None:
    """``__post_init__`` raises ``ValueError`` for ``modes`` arrays that are
    not 2D or do not have 17 columns."""
    with pytest.raises(ValueError, match="2-dimensional"):
        KSTrajectory(modes=np.zeros(17, dtype=np.complex128), dt=0.02, resolution=64)
    with pytest.raises(ValueError, match="17 columns"):
        KSTrajectory(modes=np.zeros((5, 10), dtype=np.complex128), dt=0.02, resolution=64)


def test_from_initial_state_length(sample_initial_state: np.ndarray) -> None:
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
    """``to_comoving(d, start_time=t)`` on ``traj[k:]`` (where ``t`` is the
    time at row ``k``) matches rows ``[k:]`` of ``to_comoving(d, start_time=0)``
    on the full trajectory."""
    drift = 0.27
    offset = 5
    offset_time = offset * random_trajectory.dt
    full = random_trajectory.to_comoving(drift_rate=drift, start_time=0.0)
    sliced = random_trajectory[offset:].to_comoving(drift_rate=drift, start_time=offset_time)
    np.testing.assert_allclose(sliced.modes, full.modes[offset:], atol=1e-12)


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
