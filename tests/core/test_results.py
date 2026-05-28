"""Round-trip tests for the results module."""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing.core.results import DetectionMetadata, load_results, save_results
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.trajectory import KSTrajectory


def _make_event() -> ShadowingEvent:
    return ShadowingEvent(
        rpo_index=0,
        start_timestep=10,
        end_timestep=15,
        mean_distance=0.5,
        min_distance=0.4,
        start_phase=3,
        shifts=np.array([0, 1, 0, -1, 0], dtype=np.int32),
    )


def _make_metadata(spatial_resolution: int = 32) -> DetectionMetadata:
    return DetectionMetadata(
        detector_type="SSA",
        min_duration=4,
        threshold=1.0,
        rpo_file="data/rpos_selected.npz",
        spatial_resolution=spatial_resolution,
    )


def test_result_roundtrip_returns_trajectory(
    random_trajectory: KSTrajectory, tmp_path: Path
) -> None:
    """``save_results`` / ``load_results`` round-trips events, metadata,
    and trajectory; ``loaded_trajectory.resolution`` comes from
    ``metadata.spatial_resolution``, not from the saved trajectory file."""
    trajectory_path = tmp_path / "trajectory.h5"
    result_path = tmp_path / "result.h5"

    random_trajectory.save(trajectory_path)

    metadata = _make_metadata(spatial_resolution=128)
    events = [_make_event()]

    save_results(result_path, metadata, events, trajectory_path=trajectory_path)
    loaded_metadata, loaded_trajectory, loaded_events = load_results(result_path)

    np.testing.assert_array_equal(loaded_trajectory.modes, random_trajectory.modes)
    assert loaded_trajectory.resolution == 128
    assert loaded_metadata.detector_type == "SSA"
    assert loaded_metadata.threshold == 1.0
    assert loaded_metadata.min_duration == 4
    assert loaded_metadata.spatial_resolution == 128
    assert len(loaded_events) == 1
    np.testing.assert_array_equal(loaded_events[0].shifts, events[0].shifts)


def test_save_results_rejects_non_sibling_trajectory(
    random_trajectory: KSTrajectory, tmp_path: Path
) -> None:
    """``save_results`` raises ``ValueError`` when ``trajectory_path``
    is not a sibling of ``path``."""
    trajectory_dir = tmp_path / "traj_dir"
    trajectory_dir.mkdir()
    trajectory_path = trajectory_dir / "trajectory.h5"
    result_path = tmp_path / "result.h5"
    random_trajectory.save(trajectory_path)

    with pytest.raises(ValueError, match="sibling"):
        save_results(result_path, _make_metadata(), [], trajectory_path=trajectory_path)
