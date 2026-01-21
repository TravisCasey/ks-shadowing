"""Tests for SSA detector."""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing import RPO, load_all_rpos
from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.transforms import interleaved_to_complex, to_physical
from ks_shadowing.ssa import SSADetector
from ks_shadowing.ssa.detector import compute_distances_sq
from ks_shadowing.ssa.rpo import RPOStateSpace


def make_rpo_physical_trajectory(rpo, resolution: int) -> np.ndarray:
    """Integrate RPO and convert to physical space for testing."""
    rpo_dt = rpo.period / rpo.time_steps
    fourier_trajectory = ksint(rpo.fourier_coeffs, rpo_dt, rpo.time_steps)[:-1]
    return to_physical(interleaved_to_complex(fourier_trajectory), resolution)


class TestSSADetector:
    def test_rpo_shadows_itself(self, rpo_data_path: Path):
        """RPO trajectory shadows itself with high threshold."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps
        detector = SSADetector([rpo], dt, resolution=64)

        trajectory = ksint(rpo.fourier_coeffs, dt, rpo.time_steps * 2)
        events = detector.detect(trajectory, threshold=1.0)

        assert len(events) > 0
        total_shadowed = sum(e.end_timestep - e.start_timestep for e in events)
        assert total_shadowed > len(trajectory) * 0.8

    def test_zero_threshold_no_events(self, rpo_data_path: Path, rng: np.random.Generator):
        """Zero threshold finds no events."""
        rpo = RPO.load(rpo_data_path, 0)
        detector = SSADetector([rpo], rpo.period / rpo.time_steps, resolution=128)
        trajectory = rng.standard_normal((20, 30)) * 0.1

        events = detector.detect(trajectory, threshold=0.0)
        assert len(events) == 0

    def test_compute_min_distances_shape(self, rpo_data_path: Path, rng: np.random.Generator):
        """compute_min_distances returns correct shape."""
        rpo = RPO.load(rpo_data_path, 0)
        detector = SSADetector([rpo], rpo.period / rpo.time_steps, resolution=128)
        trajectory = rng.standard_normal((15, 30)) * 0.1

        min_dists = detector.compute_min_distances(trajectory)
        assert min_dists.shape == (15,)
        assert np.all(min_dists >= 0)


class TestParallelExecution:
    def test_parallel_matches_sequential(self, rpo_data_path: Path):
        """Parallel detection produces same results as sequential."""
        rpos = load_all_rpos(rpo_data_path)[:4]
        dt = rpos[0].period / rpos[0].time_steps
        detector = SSADetector(rpos, dt, resolution=64)
        trajectory = ksint(rpos[0].fourier_coeffs, dt, 200)

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
        rpo_physical = make_rpo_physical_trajectory(rpo, resolution)
        rpo_data = RPOStateSpace(rpo=rpo, trajectory=rpo_physical)

        phases = list(compute_distances_sq(rpo_physical[:5], rpo_data))

        assert len(phases) == rpo_data.time_steps
        for _, dist_sq in phases:
            assert dist_sq.shape == (5, resolution)

    def test_self_distance_near_zero(self, rpo_data_path: Path):
        """RPO trajectory has near-zero squared distance to itself at shift=0."""
        rpo = RPO.load(rpo_data_path, 0)
        resolution = 128
        rpo_physical = make_rpo_physical_trajectory(rpo, resolution)
        rpo_data = RPOStateSpace(rpo=rpo, trajectory=rpo_physical)

        phase, dist_sq = next(compute_distances_sq(rpo_physical[:10], rpo_data))
        assert phase == 0

        for timestep in range(10):
            assert dist_sq[timestep, 0] < 1e-10
