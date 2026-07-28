"""CLI entry point for shadowing event detection."""

import time
from argparse import ArgumentParser, Namespace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ks_shadowing import RPO, DetectionResult, load_rpos, pha, ssa
from ks_shadowing.core import DEFAULT_CHUNK_SIZE, INTEGRATION_DT
from ks_shadowing.core.results import DetectionMetadata, save_results
from ks_shadowing.core.trajectory import KSTrajectory

_DEFAULT_INITIAL_AMPLITUDE = 0.1
_DEFAULT_THRESHOLD_QUANTILE = 0.4
_DEFAULT_MIN_DURATION = 600
_DEFAULT_DELAY = 1
_DEFAULT_MAX_DERIVATIVE_ORDER = 0
_DEFAULT_N_JOBS = -1
_DEFAULT_RPO_FILE = Path("data/rpos_selected.npz")
_MIN_TRAJECTORY_STEPS = 2
_DEFAULT_OUTPUT_BY_METHOD = {
    "ssa": Path("results/shadowing_results_ssa.h5"),
    "pha": Path("results/shadowing_results_pha.h5"),
}


def _build_parser() -> ArgumentParser:
    """Build CLI parser for ``ks-detect``."""
    parser = ArgumentParser(description="Detect shadowing events with SSA or PHA.")
    parser.add_argument("--version", action="version", version=version("ks-shadowing"))
    parser.add_argument(
        "--method", choices=["ssa", "pha"], required=True, help="Detection algorithm to run."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result HDF5 path. Defaults to results/shadowing_results_<method>.h5.",
    )
    parser.add_argument(
        "--rpo-file",
        type=Path,
        default=_DEFAULT_RPO_FILE,
        help="RPO data file (.npz). Default path is relative to the repository root.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for the random initial condition."
    )
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=None,
        help=(
            "Number of integration steps; the saved trajectory has one additional row for the "
            "initial condition. Total integration cost scales with --downsample. Required "
            "unless --trajectory is given."
        ),
    )
    parser.add_argument(
        "--resolution", type=int, required=True, help="Spatial grid resolution for detection."
    )
    parser.add_argument(
        "--initial-amplitude",
        type=float,
        default=None,
        help="Amplitude of the random initial Fourier modes. Default 0.1.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help=(
            "Reuse an existing trajectory file instead of generating one; must be a sibling "
            "of --output. Sampling is inferred from the file, so it is mutually exclusive "
            "with --seed, --trajectory-steps, --initial-amplitude, --downsample."
        ),
    )

    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=None,
        help="Quantile of the per-timestep min-distance distribution used as the "
        "detection threshold. Mutually exclusive with --threshold. Default 0.4.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual distance threshold for detection. Mutually exclusive with "
        "--threshold-quantile.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=_DEFAULT_MIN_DURATION,
        help="Minimum event duration in timesteps for a detected path to be kept.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=_DEFAULT_DELAY,
        help=(
            "PHA time-delay embedding window. 1 (default) means no temporal "
            "embedding; the per-RPO Wasserstein matrix is used directly. "
            "Higher values average over delay consecutive timesteps. "
            "Ignored for SSA."
        ),
    )
    parser.add_argument(
        "--max-derivative-order",
        type=int,
        default=_DEFAULT_MAX_DERIVATIVE_ORDER,
        help=(
            "Highest spatial-derivative order included in PHA persistence "
            "diagrams. 0 (default) means only the field itself; k > 0 "
            "computes diagrams of orders 0..k and averages their "
            "Wasserstein distances. Ignored for SSA."
        ),
    )
    parser.add_argument(
        "--rescale-orders",
        action="store_true",
        help="Divide each PHA derivative order's Wasserstein matrix by a "
        "per-order median scale before averaging across orders. Ignored for SSA.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Sampling stride for a newly generated trajectory: sets the saved "
        "sampling stride, so stored dt = 0.02 * N. Generation-only; mutually "
        "exclusive with --trajectory, which infers sampling from the file.",
    )
    parser.add_argument(
        "--native-rpos",
        action="store_true",
        help="Build per-RPO trajectories by visiting every native RPO timestep "
        "through the stride permutation instead of subsampling, for fuller "
        "per-orbit coverage at higher per-RPO cost. No effect unless the "
        "sampling stride exceeds 1 (the two variants coincide at stride 1).",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows of trajectory processed per memory-bounded chunk.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_DEFAULT_N_JOBS,
        help="Worker processes for per-RPO detection. -1 (default) uses all CPUs; "
        "1 runs in-process with no multiprocessing.",
    )
    parser.add_argument(
        "--show-progress", action="store_true", help="Display a cost-weighted progress bar."
    )
    return parser


def _validate_arguments(parser: ArgumentParser, arguments: Namespace) -> None:
    """Reject out-of-range values and conflicting flag combinations."""
    if arguments.downsample is not None and arguments.downsample < 1:
        parser.error("--downsample must be >= 1.")
    if arguments.delay < 1:
        parser.error("--delay must be >= 1.")
    if arguments.max_derivative_order < 0:
        parser.error("--max-derivative-order must be >= 0.")
    if (
        arguments.trajectory_steps is not None
        and arguments.trajectory_steps < _MIN_TRAJECTORY_STEPS
    ):
        parser.error(f"--trajectory-steps must be >= {_MIN_TRAJECTORY_STEPS}.")
    if arguments.threshold is not None and arguments.threshold < 0:
        parser.error("--threshold must be >= 0.")
    if arguments.threshold_quantile is not None and not (0 < arguments.threshold_quantile < 1):
        parser.error("--threshold-quantile must be in (0, 1).")
    if arguments.threshold is not None and arguments.threshold_quantile is not None:
        parser.error("--threshold and --threshold-quantile are mutually exclusive.")


def _resolve_trajectory(arguments: Namespace, output_path: Path) -> tuple[KSTrajectory, Path]:
    """Return ``(trajectory, trajectory_path)``, integrating if needed.

    Validates that ``--trajectory`` is not combined with trajectory-
    generation flags, that any user-supplied ``--trajectory`` lives
    next to ``--output`` (the sibling-co-location rule enforced by
    :func:`save_results`), and that an integrating run does not
    overwrite an existing ``trajectory.h5`` next to ``--output``.
    """
    generation_flags_set = (
        arguments.trajectory_steps is not None
        or arguments.seed is not None
        or arguments.initial_amplitude is not None
        or arguments.downsample is not None
    )

    if arguments.trajectory is not None:
        if generation_flags_set:
            raise SystemExit(
                "ks-detect: --trajectory is mutually exclusive with "
                "--trajectory-steps, --seed, --initial-amplitude, --downsample."
            )
        if arguments.trajectory.parent.resolve() != output_path.parent.resolve():
            raise SystemExit(
                f"ks-detect: --trajectory must live in the same directory as "
                f"--output. Got --trajectory {arguments.trajectory}, "
                f"--output {output_path}."
            )
        print(f"Loading trajectory from {arguments.trajectory}...")
        trajectory = KSTrajectory.load(arguments.trajectory, resolution=arguments.resolution)
        print(f"  Shape: {trajectory.modes.shape} (dt={trajectory.dt})")
        return trajectory, arguments.trajectory

    if arguments.trajectory_steps is None:
        raise SystemExit(
            "ks-detect: --trajectory-steps is required when --trajectory is not given."
        )

    downsample = arguments.downsample if arguments.downsample is not None else 1
    initial_amplitude = (
        arguments.initial_amplitude
        if arguments.initial_amplitude is not None
        else _DEFAULT_INITIAL_AMPLITUDE
    )
    rng = np.random.default_rng(arguments.seed)
    print("Generating trajectory...")
    initial_state = np.zeros(17, dtype=np.complex128)
    initial_state[1:16] = (
        rng.standard_normal(15) + 1j * rng.standard_normal(15)
    ) * initial_amplitude
    trajectory = KSTrajectory.from_initial_state(
        initial_state,
        INTEGRATION_DT,
        arguments.trajectory_steps + 1,
        arguments.resolution,
        save_interval=downsample,
    )
    print(
        f"  Shape: {trajectory.modes.shape} "
        f"({(len(trajectory) - 1) * trajectory.dt:.0f} time units, "
        f"dt={trajectory.dt:.4f})"
    )

    trajectory_path = output_path.parent / "trajectory.h5"
    if trajectory_path.exists():
        raise SystemExit(
            f"ks-detect: refusing to overwrite existing trajectory at "
            f"{trajectory_path}. Pass --trajectory {trajectory_path} to "
            f"reuse it, or remove it first."
        )
    print(f"Saving trajectory to {trajectory_path}...")
    trajectory.save(trajectory_path)
    return trajectory, trajectory_path


def main() -> None:
    """Run CLI detection and save events."""
    parser = _build_parser()
    arguments = parser.parse_args()
    _validate_arguments(parser, arguments)
    if arguments.threshold_quantile is None:
        arguments.threshold_quantile = _DEFAULT_THRESHOLD_QUANTILE

    method = arguments.method
    output_path = arguments.output or _DEFAULT_OUTPUT_BY_METHOD[method]

    trajectory, trajectory_path = _resolve_trajectory(arguments, output_path)
    downsample = round(trajectory.dt / INTEGRATION_DT)
    print(f"  Downsample (inferred from trajectory dt): {downsample}")

    print("Loading RPOs...")
    rpos = load_rpos(arguments.rpo_file)
    print(f"  Loaded {len(rpos)} RPOs from {arguments.rpo_file}")

    print(f"Detecting events with {method.upper()}...")
    start_time = time.perf_counter()

    if arguments.threshold is not None:
        result = _detect_with_threshold(method, trajectory, rpos, arguments)
        events = result.events
        threshold = arguments.threshold
        threshold_quantile = None
    else:
        result = _detect_with_auto_threshold(method, trajectory, rpos, arguments)
        events = result.events
        threshold = result.threshold
        threshold_quantile = arguments.threshold_quantile
    elapsed_seconds = time.perf_counter() - start_time

    auto_label = "auto" if threshold_quantile is not None else "manual"
    print(f"  Threshold ({auto_label}): {threshold:.4f}")
    print(f"  Found {len(events)} events")
    print(f"  Elapsed: {elapsed_seconds:.2f}s")

    metadata = DetectionMetadata(
        detector_type=method.upper(),
        min_duration=arguments.min_duration,
        threshold=threshold,
        rpo_file=str(arguments.rpo_file),
        spatial_resolution=arguments.resolution,
        elapsed_seconds=elapsed_seconds,
        downsample=downsample,
        native=arguments.native_rpos,
        threshold_quantile=threshold_quantile,
        delay=arguments.delay if method == "pha" else 1,
        max_derivative_order=arguments.max_derivative_order if method == "pha" else 0,
        rescale_orders=arguments.rescale_orders if method == "pha" else False,
        order_scales=(
            tuple(float(x) for x in result.order_scales)
            if result.order_scales is not None
            else None
        ),
    )

    print(f"Saving results to {output_path}...")
    save_results(output_path, metadata, events, trajectory_path=trajectory_path)

    if events:
        best_event = min(events, key=lambda event: event.mean_distance)
        duration_timesteps = best_event.end_timestep - best_event.start_timestep
        print(
            f"Best event: RPO {best_event.rpo_index}, duration={duration_timesteps}, "
            f"mean_dist={best_event.mean_distance:.4f}"
        )


def _detect_with_threshold(
    method: str, trajectory: KSTrajectory, rpos: list[RPO], arguments: Namespace
) -> DetectionResult:
    """Dispatch :func:`ssa.detect` or :func:`pha.detect` for a manual threshold."""
    common_kwargs = {
        "min_duration": arguments.min_duration,
        "show_progress": arguments.show_progress,
        "n_jobs": arguments.n_jobs,
        "chunk_size": arguments.chunk_size,
        "native": arguments.native_rpos,
    }
    if method == "ssa":
        return ssa.detect(trajectory, rpos, threshold=arguments.threshold, **common_kwargs)
    return pha.detect(
        trajectory,
        rpos,
        threshold=arguments.threshold,
        delay=arguments.delay,
        max_derivative_order=arguments.max_derivative_order,
        rescale_orders=arguments.rescale_orders,
        **common_kwargs,
    )


def _detect_with_auto_threshold(
    method: str, trajectory: KSTrajectory, rpos: list[RPO], arguments: Namespace
) -> DetectionResult:
    """Dispatch :func:`ssa.auto_detect` or :func:`pha.auto_detect`."""
    common_kwargs = {
        "threshold_quantile": arguments.threshold_quantile,
        "min_duration": arguments.min_duration,
        "show_progress": arguments.show_progress,
        "n_jobs": arguments.n_jobs,
        "chunk_size": arguments.chunk_size,
        "native": arguments.native_rpos,
    }
    if method == "ssa":
        return ssa.auto_detect(trajectory, rpos, **common_kwargs)
    return pha.auto_detect(
        trajectory,
        rpos,
        delay=arguments.delay,
        max_derivative_order=arguments.max_derivative_order,
        rescale_orders=arguments.rescale_orders,
        **common_kwargs,
    )


if __name__ == "__main__":
    main()
