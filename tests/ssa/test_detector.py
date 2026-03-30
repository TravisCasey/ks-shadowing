"""Tests for SSA detector."""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing import RPO, load_all_rpos
from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.ssa import SSADetector
from ks_shadowing.ssa.detector import _compute_distances_sq
from ks_shadowing.ssa.rpo import _RPOStateSpace


def _make_rpo_trajectory(rpo: RPO, resolution: int) -> KSTrajectory:
    """Integrate RPO over one period and return as KSTrajectory."""
    rpo_dt = rpo.period / rpo.time_steps
    return KSTrajectory.from_initial_state(
        rpo.fourier_coeffs, rpo_dt, rpo.time_steps + 1, resolution
    )[:-1]


class TestSSADetector:
    def test_rpo_shadows_itself(self, rpo_data_path: Path):
        """RPO trajectory shadows itself with high threshold."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps
        detector = SSADetector([rpo], dt, resolution=64)

        trajectory = KSTrajectory.from_initial_state(
            rpo.fourier_coeffs, dt, rpo.time_steps * 2 + 1, resolution=64
        )
        events = detector.detect(trajectory, threshold=1.0)

        assert len(events) > 0
        total_shadowed = sum(e.end_timestep - e.start_timestep for e in events)
        assert total_shadowed > len(trajectory) * 0.8

    def test_zero_threshold_no_events(self, rpo_data_path: Path, rng: np.random.Generator):
        """Zero threshold finds no events."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps
        detector = SSADetector([rpo], dt, resolution=128)
        fake_modes = np.zeros((20, 17), dtype=np.complex128)
        fake_modes[:, 1:16] = (
            rng.standard_normal((20, 15)) + 1j * rng.standard_normal((20, 15))
        ) * 0.1
        fake_trajectory = KSTrajectory(modes=fake_modes, dt=dt, resolution=128)

        events = detector.detect(fake_trajectory, threshold=0.0)
        assert len(events) == 0

    def test_compute_min_distances_shape(self, rpo_data_path: Path, rng: np.random.Generator):
        """compute_min_distances returns correct shape."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps
        detector = SSADetector([rpo], dt, resolution=128)
        fake_modes = np.zeros((15, 17), dtype=np.complex128)
        fake_modes[:, 1:16] = (
            rng.standard_normal((15, 15)) + 1j * rng.standard_normal((15, 15))
        ) * 0.1
        fake_trajectory = KSTrajectory(modes=fake_modes, dt=dt, resolution=128)

        min_dists = detector.compute_min_distances(fake_trajectory)
        assert min_dists.shape == (15,)
        assert np.all(min_dists >= 0)


class TestParallelExecution:
    def test_parallel_matches_sequential(self, rpo_data_path: Path):
        """Parallel detection produces same results as sequential."""
        rpos = load_all_rpos(rpo_data_path)[:4]
        dt = rpos[0].period / rpos[0].time_steps
        detector = SSADetector(rpos, dt, resolution=64)
        trajectory = KSTrajectory.from_initial_state(rpos[0].fourier_coeffs, dt, 201, resolution=64)

        events_seq = detector.detect(trajectory, threshold=0.8, n_jobs=1)
        events_par = detector.detect(trajectory, threshold=0.8, n_jobs=2)

        assert len(events_seq) == len(events_par)
        for e_seq, e_par in zip(events_seq, events_par, strict=True):
            assert e_seq.rpo_index == e_par.rpo_index
            assert e_seq.start_timestep == e_par.start_timestep
            assert e_seq.mean_distance == pytest.approx(e_par.mean_distance)


class TestComputeDistancesSq:
    def test_yields_correct_shape(self, rpo_data_path: Path):
        """Yields one squared distance array per phase with shape (timesteps, resolution)."""
        rpo = RPO.load(rpo_data_path, 0)
        resolution = 128
        rpo_trajectory = _make_rpo_trajectory(rpo, resolution)
        rpo_data = _RPOStateSpace(rpo=rpo, trajectory=rpo_trajectory)

        short_trajectory = rpo_trajectory[:5]
        results = list(_compute_distances_sq(short_trajectory, rpo_data))

        assert len(results) == rpo_data.time_steps
        for _, chunk_start, dist_sq in results:
            assert chunk_start == 0
            assert dist_sq.shape == (5, resolution)

    def test_self_distance_near_zero(self, rpo_data_path: Path):
        """RPO trajectory has near-zero squared distance to itself at shift=0."""
        rpo = RPO.load(rpo_data_path, 0)
        resolution = 128
        rpo_trajectory = _make_rpo_trajectory(rpo, resolution)
        rpo_data = _RPOStateSpace(rpo=rpo, trajectory=rpo_trajectory)

        short_trajectory = rpo_trajectory[:10]
        phase, chunk_start, dist_sq = next(_compute_distances_sq(short_trajectory, rpo_data))
        assert phase == 0
        assert chunk_start == 0

        for timestep in range(10):
            assert dist_sq[timestep, 0] < 1e-10


class TestChunkedComputation:
    def test_chunked_matches_unchunked(self, rpo_data_path: Path):
        """Chunked detection produces identical events to unchunked."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps
        detector_default = SSADetector([rpo], dt, resolution=32)
        detector_chunked = SSADetector([rpo], dt, resolution=32, chunk_size=100)

        trajectory = KSTrajectory.from_initial_state(
            rpo.fourier_coeffs, dt, rpo.time_steps * 2 + 1, resolution=32
        )
        events_default = detector_default.detect(trajectory, threshold=0.5, min_duration=1)
        events_chunked = detector_chunked.detect(trajectory, threshold=0.5, min_duration=1)

        assert len(events_default) == len(events_chunked)
        for e1, e2 in zip(events_default, events_chunked, strict=True):
            assert e1.start_timestep == e2.start_timestep
            assert e1.end_timestep == e2.end_timestep
            assert e1.rpo_index == e2.rpo_index
            np.testing.assert_allclose(e1.shifts, e2.shifts)
