"""CLI entry point for matched shadowing event analysis."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing.cli.results import DetectionMetadata, load_results
from ks_shadowing.core.matching import MatchedEvent, find_matched_events

DEFAULT_OUTPUT = Path("plots/matched_events.png")
DEFAULT_DPI = 150


def build_parser() -> ArgumentParser:
    """Build CLI parser for ``ks-match``."""
    parser = ArgumentParser(
        description=("Match overlapping SSA and PHA shadowing events and plot results."),
    )
    parser.add_argument(
        "--ssa-input",
        type=Path,
        required=True,
        help="SSA results HDF5 file.",
    )
    parser.add_argument(
        "--pha-input",
        type=Path,
        required=True,
        help="PHA results HDF5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show figure interactively.",
    )
    return parser


def _validate_result_pair(
    metadata_a: DetectionMetadata,
    initial_state_a: np.ndarray,
    metadata_b: DetectionMetadata,
    initial_state_b: np.ndarray,
) -> None:
    """Validate that two result sets are from different methods on the same trajectory.

    Parameters
    ----------
    metadata_a : DetectionMetadata
        Metadata from the first result file.
    initial_state_a : NDArray[np.float64], shape (30,)
        Initial state from the first result file.
    metadata_b : DetectionMetadata
        Metadata from the second result file.
    initial_state_b : NDArray[np.float64], shape (30,)
        Initial state from the second result file.

    Raises
    ------
    ValueError
        If both results use the same detector type, or if the
        trajectories differ (mismatched initial state or trajectory
        steps).
    """
    types = {metadata_a.detector_type, metadata_b.detector_type}
    if types != {"SSA", "PHA"}:
        raise ValueError(
            f"Expected one SSA and one PHA result file, got "
            f"{metadata_a.detector_type} and {metadata_b.detector_type}"
        )
    if not np.array_equal(initial_state_a, initial_state_b):
        raise ValueError(
            "Result files have different initial_state arrays "
            "and do not describe the same trajectory"
        )
    if metadata_a.trajectory_steps != metadata_b.trajectory_steps:
        raise ValueError(
            f"Result files have different trajectory_steps: "
            f"{metadata_a.trajectory_steps} vs "
            f"{metadata_b.trajectory_steps}"
        )


def _plot_matches(matches: list[MatchedEvent]) -> plt.Figure:
    """Create a scatter plot of matched events colored by overlap ratio.

    Parameters
    ----------
    matches : list[MatchedEvent]
        Matched event pairs to plot.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    figure, ax = plt.subplots(figsize=(8, 6))

    if not matches:
        ax.text(
            0.5,
            0.5,
            "No matched events found",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_xlabel("SSA Event Length (timesteps)")
        ax.set_ylabel("PHA Event Length (timesteps)")
        ax.set_title("Matched Shadowing Events (SSA vs PHA)")
        return figure

    ssa_lengths = np.array([m.ssa_event.end_timestep - m.ssa_event.start_timestep for m in matches])
    pha_lengths = np.array([m.pha_event.end_timestep - m.pha_event.start_timestep for m in matches])
    iou = np.array([m.intersection_length / m.union_length for m in matches])

    scatter = ax.scatter(
        ssa_lengths,
        pha_lengths,
        c=iou,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(scatter, ax=ax, label="Overlap (IoU)")
    ax.set_xlabel("SSA Event Length (timesteps)")
    ax.set_ylabel("PHA Event Length (timesteps)")
    ax.set_title("Matched Shadowing Events (SSA vs PHA)")
    plt.tight_layout()
    return figure


def main() -> None:
    """Run match command."""
    parser = build_parser()
    arguments = parser.parse_args()

    print(f"Loading SSA results from {arguments.ssa_input}...")
    ssa_metadata, ssa_state, ssa_events = load_results(
        arguments.ssa_input,
    )
    print(f"  Events: {len(ssa_events)}")

    print(f"Loading PHA results from {arguments.pha_input}...")
    pha_metadata, pha_state, pha_events = load_results(
        arguments.pha_input,
    )
    print(f"  Events: {len(pha_events)}")

    _validate_result_pair(
        ssa_metadata,
        ssa_state,
        pha_metadata,
        pha_state,
    )

    matches = find_matched_events(ssa_events, pha_events)

    matched_ssa = {id(m.ssa_event) for m in matches}
    matched_pha = {id(m.pha_event) for m in matches}
    unmatched_ssa = sum(1 for e in ssa_events if id(e) not in matched_ssa)
    unmatched_pha = sum(1 for e in pha_events if id(e) not in matched_pha)

    print(f"\nMatched pairs: {len(matches)}")
    print(f"Unmatched SSA events: {unmatched_ssa}")
    print(f"Unmatched PHA events: {unmatched_pha}")

    figure = _plot_matches(matches)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        arguments.output,
        dpi=arguments.dpi,
        bbox_inches="tight",
    )

    if arguments.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"Saved plot to {arguments.output}")


if __name__ == "__main__":
    main()
