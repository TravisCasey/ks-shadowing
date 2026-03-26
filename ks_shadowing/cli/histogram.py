"""CLI entry point for shadowing event duration histogram."""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing.core import TRAJECTORY_DT

DEFAULT_OUTPUT = Path("plots/event_duration_histogram.png")
DEFAULT_BIN_WIDTH = 10
DEFAULT_DPI = 150


def build_parser() -> ArgumentParser:
    """Build CLI parser for ``ks-histogram``."""
    parser = ArgumentParser(description="Plot a histogram of shadowing event durations.")
    parser.add_argument("--input", type=Path, required=True, help="Input HDF5 results file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=DEFAULT_BIN_WIDTH,
        help=f"Bin width in number of timesteps (default: {DEFAULT_BIN_WIDTH}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help=f"Output image DPI (default: {DEFAULT_DPI})."
    )
    parser.add_argument(
        "--time-units",
        action="store_true",
        default=False,
        help="Label x-axis in time units instead of timesteps.",
    )
    parser.add_argument("--show", action="store_true", default=False, help="Show figure.")
    return parser


def load_event_durations(path: Path) -> np.ndarray:
    """Load event durations from an HDF5 results file without reconstructing the trajectory."""
    with h5py.File(path, "r") as f:
        events = f["events"][:]
    starts = events["start_timestep"]
    ends = events["end_timestep"]
    return (ends - starts).astype(np.int64)


def main() -> None:
    """Run histogram command."""
    parser = build_parser()
    arguments = parser.parse_args()

    print(f"Loading events from {arguments.input}...")
    durations = load_event_durations(arguments.input)

    if len(durations) == 0:
        print("No events found in this file.")
        return

    print(f"  Events: {len(durations)}")
    print(f"  Duration range: {durations.min()} - {durations.max()} timesteps")

    bin_width = arguments.bin_width
    bin_max = (durations.max() // bin_width + 1) * bin_width
    bins = np.arange(0, bin_max + bin_width, bin_width)

    if arguments.time_units:
        scale = TRAJECTORY_DT
        xlabel = "Event Duration (time units)"
        bin_edges = bins * scale
        plot_durations = durations * scale
    else:
        scale = 1
        xlabel = "Event Duration (timesteps)"
        bin_edges = bins.astype(float)
        plot_durations = durations.astype(float)

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.hist(plot_durations, bins=bin_edges, edgecolor="black", linewidth=0.5)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Events")
    ax.set_title(
        f"Shadowing Event Duration Distribution (n={len(durations)}, bin width={bin_width})"
    )
    plt.tight_layout()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(arguments.output, dpi=arguments.dpi, bbox_inches="tight")

    if arguments.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"Saved histogram to {arguments.output}")


if __name__ == "__main__":
    main()
