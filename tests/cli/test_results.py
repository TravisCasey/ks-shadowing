"""Tests for CLI result serialization."""

from pathlib import Path

import numpy as np

from ks_shadowing.cli.results import DetectionMetadata, load_results, save_results
from ks_shadowing.core.event import ShadowingEvent


def build_metadata() -> DetectionMetadata:
    """Build minimal metadata block for result files."""
    return DetectionMetadata(
        detector_type="SSA",
        seed=123,
        spatial_resolution=64,
        trajectory_steps=5,
        initial_amplitude=0.1,
        min_duration=5,
        threshold=1.5,
        rpo_file="data/rpos_selected.npz",
        threshold_quantile=0.4,
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Save and load should preserve metadata and event arrays."""
    output_path = tmp_path / "results.h5"

    initial_state = np.random.default_rng(42).standard_normal(30) * 0.1
    events = [
        ShadowingEvent(
            rpo_index=2,
            start_timestep=3,
            end_timestep=7,
            mean_distance=0.12,
            min_distance=0.04,
            start_phase=5,
            shifts=np.array([1, 2, 2, 3], dtype=np.int32),
        )
    ]

    save_results(output_path, build_metadata(), initial_state, events)
    metadata, loaded_state, loaded_events = load_results(output_path)

    assert metadata.detector_type == "SSA"
    assert metadata.trajectory_steps == 5
    assert metadata.threshold_quantile == 0.4
    assert metadata.delay is None

    np.testing.assert_array_equal(loaded_state, initial_state)

    assert len(loaded_events) == 1
    loaded_event = loaded_events[0]
    assert loaded_event.rpo_index == 2
    assert loaded_event.start_timestep == 3
    assert loaded_event.end_timestep == 7
    np.testing.assert_array_equal(loaded_event.shifts, np.array([1, 2, 2, 3], dtype=np.int32))


def test_save_load_zero_events(tmp_path: Path) -> None:
    """Save and load should support files with zero events."""
    output_path = tmp_path / "results_empty.h5"
    initial_state = np.random.default_rng(0).standard_normal(30) * 0.1

    save_results(output_path, build_metadata(), initial_state, [])
    metadata, loaded_state, loaded_events = load_results(output_path)

    assert metadata.detector_type == "SSA"
    np.testing.assert_array_equal(loaded_state, initial_state)
    assert loaded_events == []
