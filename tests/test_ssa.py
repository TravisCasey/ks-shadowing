"""Tests for State Space Approach (SSA) shadowing detection."""

from pathlib import Path

import numpy as np

from ks_shadowing import load_rpo
from ks_shadowing.integrator import ksint
from ks_shadowing.ssa import SSADetector, compute_distances_to_rpo
from ks_shadowing.transforms import interleaved_to_complex, to_physical


def make_rpo_trajectory(rpo, dt: float, resolution: int) -> np.ndarray:
    """Helper to create RPO physical trajectory for tests."""
    fourier_traj = ksint(rpo.fourier_coeffs, dt, rpo.time_steps)
    fourier_traj = fourier_traj[:-1]  # Exclude endpoint (periodic)
    complex_coeffs = interleaved_to_complex(fourier_traj)
    return to_physical(complex_coeffs, resolution)


class TestSSADetector:
    def test_rpo_shadows_itself(self, rpo_mat_path: Path):
        """RPO trajectory shadows itself with high threshold."""
        rpo = load_rpo(rpo_mat_path, 0)
        dt = rpo.period / rpo.time_steps

        detector = SSADetector([rpo], dt, resolution=64)

        # Integrate RPO as trajectory
        trajectory = ksint(rpo.fourier_coeffs, dt, rpo.time_steps * 2)

        # With high threshold, should find shadowing
        events = detector.detect(trajectory, threshold=1.0)
        assert len(events) > 0

        # The events should cover most of the trajectory
        total_shadowed = sum(e.duration for e in events)
        assert total_shadowed > len(trajectory) * 0.9

    def test_zero_threshold_no_events(self, rpo_mat_path: Path, rng: np.random.Generator):
        """Zero threshold finds no events (distances are positive)."""
        rpo = load_rpo(rpo_mat_path, 0)
        dt = rpo.period / rpo.time_steps

        detector = SSADetector([rpo], dt, resolution=128)

        # Random trajectory
        trajectory = rng.standard_normal((20, 30)) * 0.1

        events = detector.detect(trajectory, threshold=0.0)
        assert len(events) == 0

    def test_compute_min_distances_shape(self, rpo_mat_path: Path, rng: np.random.Generator):
        """compute_min_distances returns correct shape."""
        rpo = load_rpo(rpo_mat_path, 0)
        dt = rpo.period / rpo.time_steps

        detector = SSADetector([rpo], dt, resolution=128)
        trajectory = rng.standard_normal((15, 30)) * 0.1

        min_dists = detector.compute_min_distances(trajectory)
        assert min_dists.shape == (15,)
        assert np.all(min_dists >= 0)


class TestComputeDistancesToRpo:
    def test_yields_correct_shape(self, rpo_mat_path: Path):
        """compute_distances_to_rpo yields arrays with correct shape."""
        rpo = load_rpo(rpo_mat_path, 0)
        dt = rpo.period / rpo.time_steps
        resolution = 128

        rpo_traj = make_rpo_trajectory(rpo, dt, resolution)
        trajectory = rpo_traj[:5]  # Just a few timesteps

        distances_list = list(compute_distances_to_rpo(trajectory, rpo_traj))

        assert len(distances_list) == 5
        for dist in distances_list:
            assert dist.shape == (rpo_traj.shape[0], resolution)

    def test_self_distance_near_zero(self, rpo_mat_path: Path):
        """RPO snapshot has near-zero distance to itself at correct phase/shift."""
        rpo = load_rpo(rpo_mat_path, 0)
        dt = rpo.period / rpo.time_steps
        resolution = 128

        rpo_traj = make_rpo_trajectory(rpo, dt, resolution)

        # Use first snapshot of RPO as trajectory
        trajectory = rpo_traj[:1]

        distances = next(compute_distances_to_rpo(trajectory, rpo_traj))

        # Distance at phase=0, shift=0 should be near zero
        assert distances[0, 0] < 1e-10
