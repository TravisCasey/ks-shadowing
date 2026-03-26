"""Tests for ks-match CLI validation."""

import numpy as np
import pytest

from ks_shadowing.cli.match import _validate_result_pair
from ks_shadowing.cli.results import DetectionMetadata


def _make_metadata(
    detector_type: str,
    trajectory_steps: int = 1000,
) -> DetectionMetadata:
    """Build minimal metadata for testing."""
    return DetectionMetadata(
        detector_type=detector_type,
        seed=42,
        spatial_resolution=64,
        trajectory_steps=trajectory_steps,
        initial_amplitude=0.1,
        min_duration=5,
        threshold=1.5,
        rpo_file="data/rpos_selected.npz",
    )


class TestValidateResultPair:
    """Tests for _validate_result_pair."""

    def test_valid_pair(self) -> None:
        """Accepts SSA + PHA with matching initial states."""
        state = np.ones(30)
        _validate_result_pair(
            _make_metadata("SSA"),
            state,
            _make_metadata("PHA"),
            state,
        )

    def test_same_detector_type_raises(self) -> None:
        """Rejects two results from the same detector type."""
        state = np.ones(30)
        with pytest.raises(ValueError, match=r"one SSA.*one PHA"):
            _validate_result_pair(
                _make_metadata("SSA"),
                state,
                _make_metadata("SSA"),
                state,
            )

    def test_different_initial_state_raises(self) -> None:
        """Rejects results from different trajectories."""
        with pytest.raises(ValueError, match="initial_state"):
            _validate_result_pair(
                _make_metadata("SSA"),
                np.ones(30),
                _make_metadata("PHA"),
                np.zeros(30),
            )

    def test_different_trajectory_steps_raises(self) -> None:
        """Rejects results with different trajectory lengths."""
        state = np.ones(30)
        with pytest.raises(ValueError, match="trajectory_steps"):
            _validate_result_pair(
                _make_metadata("SSA", trajectory_steps=1000),
                state,
                _make_metadata("PHA", trajectory_steps=2000),
                state,
            )
