"""CLI entry point for matched shadowing event analysis."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing.core.results import load_results
from ks_shadowing.core.matching import MatchedEvent, match_events
from ks_shadowing.core.trajectory import KSTrajectory

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
    trajectory_a: KSTrajectory,
    trajectory_b: KSTrajectory,
) -> None:
    """Validate two result files were detected against the same trajectory.

    Compares trajectory length, resolution, and modes element-wise.

    Parameters
    ----------
    trajectory_a, trajectory_b : KSTrajectory
        Trajectories loaded alongside each result.

    Raises
    ------
    ValueError
        If the trajectories differ in length, resolution, or content.
    """
    if trajectory_a.num_timesteps != trajectory_b.num_timesteps:
        raise ValueError(
            f"Result files have different trajectory lengths: "
            f"{trajectory_a.num_timesteps} vs {trajectory_b.num_timesteps}"
        )
    if trajectory_a.resolution != trajectory_b.resolution:
        raise ValueError(
            f"Result files have different resolutions: "
            f"{trajectory_a.resolution} vs {trajectory_b.resolution}"
        )
    if not np.array_equal(trajectory_a.modes, trajectory_b.modes):
        raise ValueError("Result files reference different trajectories (modes arrays differ).")


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

    ssa_lengths = np.array(
        [match.ssa_event.end_timestep - match.ssa_event.start_timestep for match in matches]
    )
    pha_lengths = np.array(
        [match.pha_event.end_timestep - match.pha_event.start_timestep for match in matches]
    )
    iou = np.array([match.intersection_length / match.union_length for match in matches])

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
    _, ssa_trajectory, ssa_events = load_results(arguments.ssa_input)
    print(f"  Events: {len(ssa_events)}")

    print(f"Loading PHA results from {arguments.pha_input}...")
    _, pha_trajectory, pha_events = load_results(arguments.pha_input)
    print(f"  Events: {len(pha_events)}")

    _validate_result_pair(ssa_trajectory, pha_trajectory)

    matches = match_events(ssa_events, pha_events)

    matched_ssa_ids = {id(match.ssa_event) for match in matches}
    matched_pha_ids = {id(match.pha_event) for match in matches}
    unmatched_ssa = sum(1 for event in ssa_events if id(event) not in matched_ssa_ids)
    unmatched_pha = sum(1 for event in pha_events if id(event) not in matched_pha_ids)

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
