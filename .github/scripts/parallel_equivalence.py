"""Verify that parallel detection reproduces sequential detection exactly.

This runs as a standalone script for CI rather than a pytest case, as running a
process pool inside pytest is fragile.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ks_shadowing import DetectionResult, ShadowingEvent, load_rpos, pha, ssa
from ks_shadowing.core import INTEGRATION_DT, KSTrajectory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RPO_FILE = _REPO_ROOT / "data" / "rpos_selected.npz"

_NUM_RPOS = 2
_NUM_TIMESTEPS = 200
_RESOLUTION = 32


def _build_trajectory() -> KSTrajectory:
    """Integrate a short trajectory from a fixed pseudo-random initial state."""
    rng = np.random.default_rng(0)
    modes = np.zeros(17, dtype=np.complex128)
    modes[1:16] = (rng.standard_normal(15) + 1j * rng.standard_normal(15)) * 0.1
    return KSTrajectory.from_initial_state(modes, INTEGRATION_DT, _NUM_TIMESTEPS, _RESOLUTION)


def _assert_event_matches(label: str, sequential: ShadowingEvent, parallel: ShadowingEvent) -> None:
    """Assert two events agree on every field."""
    sequential_key = (
        sequential.rpo_index,
        sequential.start_timestep,
        sequential.end_timestep,
        sequential.start_phase,
    )
    parallel_key = (
        parallel.rpo_index,
        parallel.start_timestep,
        parallel.end_timestep,
        parallel.start_phase,
    )
    assert sequential_key == parallel_key, f"{label}: {sequential_key} != {parallel_key}"
    np.testing.assert_array_equal(sequential.shifts, parallel.shifts, err_msg=f"{label}: shifts")
    np.testing.assert_allclose(sequential.mean_distance, parallel.mean_distance)
    np.testing.assert_allclose(sequential.min_distance, parallel.min_distance)


def _assert_results_match(
    label: str, sequential: DetectionResult, parallel: DetectionResult
) -> None:
    """Assert two detection results agree on threshold and every event."""
    np.testing.assert_allclose(
        sequential.threshold, parallel.threshold, err_msg=f"{label}: threshold"
    )
    assert len(sequential.events) == len(parallel.events), (
        f"{label}: event count {len(sequential.events)} != {len(parallel.events)}"
    )
    for sequential_event, parallel_event in zip(sequential.events, parallel.events, strict=True):
        _assert_event_matches(label, sequential_event, parallel_event)
    print(f"  {label}: {len(sequential.events)} events, parallel matches sequential")


def main() -> int:
    """Run every parallel/sequential comparison, returning a process exit code."""
    print(f"python {sys.version}")
    if not _RPO_FILE.exists():
        print(f"RPO data file not found: {_RPO_FILE}", file=sys.stderr)
        return 1

    rpos = load_rpos(_RPO_FILE)[:_NUM_RPOS]
    trajectory = _build_trajectory()

    cases: list[tuple[str, Any, int, dict[str, Any]]] = [
        ("ssa", ssa, 2, {}),
        ("ssa n_jobs=-1", ssa, -1, {}),
        ("pha", pha, 2, {}),
        ("pha delay=2 order=1", pha, 2, {"delay": 2, "max_derivative_order": 1}),
        ("pha rescaled", pha, 2, {"max_derivative_order": 1, "rescale_orders": True}),
        ("pha n_jobs=-1", pha, -1, {}),
    ]

    for label, module in [("ssa", ssa), ("pha", pha)]:
        sequential_distances = module.compute_min_distances(
            trajectory, rpos, n_jobs=1, show_progress=False
        )
        parallel_distances = module.compute_min_distances(
            trajectory, rpos, n_jobs=2, show_progress=False
        )
        np.testing.assert_allclose(
            sequential_distances,
            parallel_distances,
            err_msg=f"{label}.compute_min_distances",
        )
        print(f"  {label}.compute_min_distances: parallel matches sequential")

    for label, module, n_jobs, kwargs in cases:
        sequential = module.auto_detect(trajectory, rpos, n_jobs=1, show_progress=False, **kwargs)
        parallel = module.auto_detect(
            trajectory, rpos, n_jobs=n_jobs, show_progress=False, **kwargs
        )
        _assert_results_match(label, sequential, parallel)

    print("parallel and sequential detection agree on all cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
