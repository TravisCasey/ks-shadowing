"""CLI entry point for shadowing event detection."""

import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ks_shadowing import PHADetector, SSADetector, load_all_rpos
from ks_shadowing.cli.results import DetectionMetadata, save_results
from ks_shadowing.core import DEFAULT_CHUNK_SIZE, TRAJECTORY_DT
from ks_shadowing.core.trajectory import KSTrajectory

DEFAULT_INITIAL_AMPLITUDE = 0.1
DEFAULT_THRESHOLD_QUANTILE = 0.4
DEFAULT_MIN_DURATION = 600
DEFAULT_DELAY = 4
DEFAULT_N_JOBS = -1
DEFAULT_RPO_FILE = Path("data/rpos_selected.npz")
DEFAULT_OUTPUT_BY_METHOD = {
    "ssa": Path("results/shadowing_results_ssa.h5"),
    "pha": Path("results/shadowing_results_pha.h5"),
}


def build_parser() -> ArgumentParser:
    """Build CLI parser for ``ks-detect``."""
    parser = ArgumentParser(description="Detect shadowing events with SSA or PHA.")
    parser.add_argument("--method", choices=["ssa", "pha"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rpo-file", type=Path, default=DEFAULT_RPO_FILE)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trajectory-steps", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--initial-amplitude", type=float, default=DEFAULT_INITIAL_AMPLITUDE)

    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--min-duration", type=int, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY)

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--show-progress", action="store_true", default=False)
    return parser


def main() -> None:
    """Run CLI detection and save events."""
    parser = build_parser()
    arguments = parser.parse_args()

    method = arguments.method
    output_path = arguments.output or DEFAULT_OUTPUT_BY_METHOD[method]

    print("Loading RPOs...")
    rpos = load_all_rpos(arguments.rpo_file)
    print(f"  Loaded {len(rpos)} RPOs from {arguments.rpo_file}")

    rng = np.random.default_rng(arguments.seed)

    print("Generating trajectory...")
    initial_state = np.zeros(17, dtype=np.complex128)
    initial_state[1:16] = (
        rng.standard_normal(15) + 1j * rng.standard_normal(15)
    ) * arguments.initial_amplitude
    resolution = arguments.resolution
    trajectory = KSTrajectory.from_initial_state(
        initial_state, TRAJECTORY_DT, arguments.trajectory_steps + 1, resolution
    )
    print(
        f"  Shape: {trajectory.modes.shape} "
        f"({arguments.trajectory_steps * TRAJECTORY_DT:.0f} time units, dt={TRAJECTORY_DT})"
    )

    if method == "ssa":
        detector = SSADetector(
            rpos, TRAJECTORY_DT, resolution=resolution, chunk_size=arguments.chunk_size
        )
    else:
        detector = PHADetector(
            rpos,
            TRAJECTORY_DT,
            resolution=resolution,
            delay=arguments.delay,
            chunk_size=arguments.chunk_size,
        )

    print(f"Detecting events with {method.upper()}...")
    t0 = time.perf_counter()

    if arguments.threshold is not None:
        events = detector.detect(
            trajectory,
            threshold=arguments.threshold,
            min_duration=arguments.min_duration,
            show_progress=arguments.show_progress,
            n_jobs=arguments.n_jobs,
        )
        threshold = arguments.threshold
        threshold_quantile = None
    else:
        events, threshold = detector.auto_detect(
            trajectory,
            threshold_quantile=arguments.threshold_quantile,
            min_duration=arguments.min_duration,
            show_progress=arguments.show_progress,
            n_jobs=arguments.n_jobs,
            downsample=arguments.downsample,
        )
        threshold_quantile = arguments.threshold_quantile
    elapsed_seconds = time.perf_counter() - t0

    auto_label = "auto" if threshold_quantile is not None else "manual"
    print(f"  Threshold ({auto_label}): {threshold:.4f}")
    print(f"  Found {len(events)} events")
    print(f"  Elapsed: {elapsed_seconds:.2f}s")

    metadata = DetectionMetadata(
        detector_type=method.upper(),
        seed=arguments.seed if arguments.seed is not None else -1,
        spatial_resolution=arguments.resolution,
        trajectory_steps=arguments.trajectory_steps,
        initial_amplitude=arguments.initial_amplitude,
        min_duration=arguments.min_duration,
        threshold=threshold,
        rpo_file=str(arguments.rpo_file),
        threshold_quantile=threshold_quantile,
        delay=arguments.delay if method == "pha" else None,
        elapsed_seconds=elapsed_seconds,
    )

    print(f"Saving results to {output_path}...")
    save_results(output_path, metadata, trajectory.modes[0], events)

    if events:
        best_event = min(events, key=lambda event: event.mean_distance)
        duration_timestep = best_event.end_timestep - best_event.start_timestep
        print(
            f"Best event: RPO {best_event.rpo_index}, duration={duration_timestep}, "
            f"mean_dist={best_event.mean_distance:.4f}"
        )


if __name__ == "__main__":
    main()
