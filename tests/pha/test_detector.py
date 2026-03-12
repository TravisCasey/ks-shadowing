"""Tests for PHA detector."""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing import RPO, load_all_rpos
from ks_shadowing.core.integrator import ksint
from ks_shadowing.pha import PHADetector


class TestPHADetector:
    def test_rpo_shadows_itself(self, rpo_data_path: Path):
        """RPO trajectory shadows itself with high threshold."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps
        detector = PHADetector([rpo], dt, resolution=32, delay=3)

        trajectory = ksint(rpo.fourier_coeffs, dt, rpo.time_steps * 2)
        events = detector.detect(trajectory, threshold=1.0)

        assert len(events) > 0
        total_shadowed = sum(e.end_timestep - e.start_timestep for e in events)
        assert total_shadowed > len(trajectory) * 0.8

    def test_zero_threshold_no_events(self, rpo_data_path: Path, rng: np.random.Generator):
        """Zero threshold finds no events."""
        rpo = RPO.load(rpo_data_path, 0)
        detector = PHADetector([rpo], dt=rpo.period / rpo.time_steps, resolution=32, delay=3)

        trajectory = rng.standard_normal((20, 30)) * 0.1

        events = detector.detect(trajectory, threshold=0.0)
        assert len(events) == 0

    def test_compute_min_distances(self, rpo_data_path: Path, rng: np.random.Generator):
        """compute_min_distances returns correct shape with proper edge effects."""
        rpo = RPO.load(rpo_data_path, 0)
        delay = 5
        detector = PHADetector([rpo], dt=rpo.period / rpo.time_steps, resolution=32, delay=delay)

        trajectory = rng.standard_normal((20, 30)) * 0.1

        min_dists = detector.compute_min_distances(trajectory)

        assert min_dists.shape == (20,)
        assert np.all(min_dists >= 0)
        # Last (delay - 1) entries should be infinite
        assert np.all(np.isinf(min_dists[-(delay - 1) :]))
        # Earlier entries should be finite
        assert np.all(np.isfinite(min_dists[: -(delay - 1)]))

    def test_auto_detect_returns_threshold(self, rpo_data_path: Path, rng: np.random.Generator):
        """auto_detect returns events and the computed threshold."""
        rpo = RPO.load(rpo_data_path, 0)
        detector = PHADetector([rpo], dt=rpo.period / rpo.time_steps, resolution=32, delay=3)

        trajectory = rng.standard_normal((20, 30)) * 0.1

        _, threshold = detector.auto_detect(trajectory, threshold_quantile=0.4)

        assert isinstance(threshold, float)
        assert threshold > 0


class TestParallelExecution:
    def test_parallel_matches_sequential(self, rpo_data_path: Path):
        """Parallel detection produces same results as sequential."""
        rpos = load_all_rpos(rpo_data_path)[:3]
        dt = rpos[0].period / rpos[0].time_steps
        detector = PHADetector(rpos, dt, resolution=32, delay=3)

        trajectory = ksint(rpos[0].fourier_coeffs, dt, 25)

        events_seq = detector.detect(trajectory, threshold=5.0, n_jobs=1)
        events_par = detector.detect(trajectory, threshold=5.0, n_jobs=2)

        assert len(events_seq) == len(events_par)
        for e_seq, e_par in zip(events_seq, events_par, strict=True):
            assert e_seq.rpo_index == e_par.rpo_index
            assert e_seq.start_timestep == e_par.start_timestep
            assert e_seq.mean_distance == pytest.approx(e_par.mean_distance)


class TestDelayEmbedding:
    def test_delay_behavior(self, rpo_data_path: Path, rng: np.random.Generator):
        """Different delay values produce different results."""
        rpo = RPO.load(rpo_data_path, 0)
        dt = rpo.period / rpo.time_steps

        trajectory = rng.standard_normal((25, 30)) * 0.1

        detector_delay1 = PHADetector([rpo], dt, resolution=32, delay=1)
        detector_delay7 = PHADetector([rpo], dt, resolution=32, delay=7)

        min_dists_1 = detector_delay1.compute_min_distances(trajectory)
        min_dists_7 = detector_delay7.compute_min_distances(trajectory)

        # delay=1 has no edge effects - all entries finite
        assert np.all(np.isfinite(min_dists_1))

        # Different delays should produce different min distances
        assert not np.allclose(min_dists_1, min_dists_7, equal_nan=True)
