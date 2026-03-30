"""Tests for KSTrajectory and shift_distances_sq."""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing.core.rpo import load_all_rpos
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq


@pytest.fixture
def rpo_trajectory(rpo_data_path: Path) -> KSTrajectory:
    """Short trajectory from the first RPO's initial condition."""
    rpos = load_all_rpos(rpo_data_path)
    rpo = rpos[0]
    steps = 50
    return KSTrajectory.from_initial_state(rpo.fourier_coeffs, dt=0.02, steps=steps, resolution=64)


@pytest.fixture
def random_trajectory(rng: np.random.Generator) -> KSTrajectory:
    """Random KSTrajectory for transform tests."""
    modes = np.zeros((20, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((20, 15)) + 1j * rng.standard_normal((20, 15))) * 0.1
    return KSTrajectory(modes=modes, dt=0.02, resolution=64)


class TestFromInitialState:
    def test_length_equals_steps(self, rpo_data_path: Path) -> None:
        rpos = load_all_rpos(rpo_data_path)
        rpo = rpos[0]
        steps = 30
        result = KSTrajectory.from_initial_state(
            rpo.fourier_coeffs, dt=0.02, steps=steps, resolution=64
        )
        assert len(result) == steps

    def test_modes_shape(self, rpo_trajectory: KSTrajectory) -> None:
        assert rpo_trajectory.modes.shape == (50, 17)

    def test_modes_dtype(self, rpo_trajectory: KSTrajectory) -> None:
        assert rpo_trajectory.modes.dtype == np.complex128


class TestToPhysical:
    def test_parseval(self, random_trajectory: KSTrajectory) -> None:
        """Parseval: sum(physical**2) == 2 * resolution * sum(|modes|**2)."""
        physical = random_trajectory.to_physical()
        physical_energy = np.sum(physical**2)
        spectral_energy = (
            2 * random_trajectory.resolution * np.sum(np.abs(random_trajectory.modes) ** 2)
        )
        np.testing.assert_allclose(physical_energy, spectral_energy, rtol=1e-6, atol=1e-6)

    def test_output_shape(self, random_trajectory: KSTrajectory) -> None:
        physical = random_trajectory.to_physical()
        assert physical.shape == (20, 64)


class TestToComoving:
    def test_zero_drift_is_identity(self, random_trajectory: KSTrajectory) -> None:
        comoving = random_trajectory.to_comoving(drift_rate=0.0)
        np.testing.assert_allclose(comoving.modes, random_trajectory.modes, atol=1e-6)

    def test_drift_then_negate_recovers_original(self, random_trajectory: KSTrajectory) -> None:
        drift = 0.35
        forward = random_trajectory.to_comoving(drift_rate=drift)
        recovered = forward.to_comoving(drift_rate=-drift)
        np.testing.assert_allclose(recovered.modes, random_trajectory.modes, atol=1e-6)

    def test_preserves_dt_and_resolution(self, random_trajectory: KSTrajectory) -> None:
        comoving = random_trajectory.to_comoving(drift_rate=0.1)
        assert comoving.dt == random_trajectory.dt
        assert comoving.resolution == random_trajectory.resolution


class TestTile:
    def test_length_at_least_target(self, random_trajectory: KSTrajectory) -> None:
        target = 75
        tiled = random_trajectory.tile(target)
        assert len(tiled) >= target

    def test_periodic_structure(self, random_trajectory: KSTrajectory) -> None:
        tiled = random_trajectory.tile(75)
        period = len(random_trajectory)
        for index in range(len(tiled)):
            np.testing.assert_array_equal(
                tiled.modes[index], random_trajectory.modes[index % period]
            )

    def test_no_op_when_already_long_enough(self, random_trajectory: KSTrajectory) -> None:
        tiled = random_trajectory.tile(10)
        assert len(tiled) == len(random_trajectory)


class TestGetitem:
    def test_slice_preserves_dt_and_resolution(self, random_trajectory: KSTrajectory) -> None:
        sliced = random_trajectory[2:5]
        assert sliced.dt == random_trajectory.dt
        assert sliced.resolution == random_trajectory.resolution

    def test_slice_length(self, random_trajectory: KSTrajectory) -> None:
        assert len(random_trajectory[2:5]) == 3

    def test_integer_returns_single_timestep(self, random_trajectory: KSTrajectory) -> None:
        single = random_trajectory[3]
        assert single.modes.shape == (1, 17)

    def test_integer_preserves_values(self, random_trajectory: KSTrajectory) -> None:
        single = random_trajectory[3]
        np.testing.assert_array_equal(single.modes[0], random_trajectory.modes[3])


class TestChunksPhysical:
    def test_reconstructs_full_physical(self, random_trajectory: KSTrajectory) -> None:
        chunks = list(random_trajectory.chunks_physical(chunk_size=7))
        reconstructed = np.vstack([chunk for _, chunk in chunks])
        expected = random_trajectory.to_physical()
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-6, atol=1e-6)

    def test_start_indices(self, random_trajectory: KSTrajectory) -> None:
        starts = [start for start, _ in random_trajectory.chunks_physical(chunk_size=7)]
        assert starts == [0, 7, 14]


class TestChunksFourier:
    def test_reconstructs_full_modes(self, random_trajectory: KSTrajectory) -> None:
        chunks = list(random_trajectory.chunks_fourier(chunk_size=7))
        reconstructed = np.vstack([chunk for _, chunk in chunks])
        np.testing.assert_array_equal(reconstructed, random_trajectory.modes)

    def test_start_indices(self, random_trajectory: KSTrajectory) -> None:
        starts = [start for start, _ in random_trajectory.chunks_fourier(chunk_size=7)]
        assert starts == [0, 7, 14]


class TestShiftDistancesSq:
    def test_self_distance_zero_at_shift_zero(self, random_trajectory: KSTrajectory) -> None:
        distances = shift_distances_sq(
            random_trajectory.modes, random_trajectory.modes, random_trajectory.resolution
        )
        np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-6)

    def test_all_non_negative(self, random_trajectory: KSTrajectory) -> None:
        distances = shift_distances_sq(
            random_trajectory.modes, random_trajectory.modes, random_trajectory.resolution
        )
        assert np.all(distances >= -1e-6)

    def test_output_shape(self, random_trajectory: KSTrajectory) -> None:
        distances = shift_distances_sq(
            random_trajectory.modes, random_trajectory.modes, random_trajectory.resolution
        )
        assert distances.shape == (20, 64)


class TestPostInit:
    def test_rejects_1d_modes(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            KSTrajectory(modes=np.zeros(17, dtype=np.complex128), dt=0.02, resolution=64)

    def test_rejects_wrong_column_count(self) -> None:
        with pytest.raises(ValueError, match="17 columns"):
            KSTrajectory(modes=np.zeros((5, 10), dtype=np.complex128), dt=0.02, resolution=64)
