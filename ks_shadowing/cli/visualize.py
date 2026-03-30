"""CLI entry point for shadowing event visualization."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ks_shadowing import load_all_rpos
from ks_shadowing.cli.plotting import _align_rpo_to_window
from ks_shadowing.cli.results import load_results
from ks_shadowing.core import DOMAIN_SIZE, TRAJECTORY_DT
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.trajectory import KSTrajectory

DEFAULT_OUTPUT = Path("plots/shadowing_visualization.png")
DEFAULT_CONTEXT_FRACTION = 1.2
DEFAULT_DPI = 150


def build_parser() -> ArgumentParser:
    """Build CLI parser for ``ks-visualize``."""
    parser = ArgumentParser(description="Visualize saved shadowing events.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-fraction", type=float, default=DEFAULT_CONTEXT_FRACTION)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--show", action="store_true", default=False)

    event_group = parser.add_mutually_exclusive_group()
    event_group.add_argument("--event-rank", type=int, default=0)
    event_group.add_argument("--event-index", type=int, default=None)

    return parser


def select_event(
    events: list[ShadowingEvent],
    event_rank: int,
    event_index: int | None,
) -> ShadowingEvent:
    """Select an event either by absolute index or by rank by mean distance."""
    if event_index is not None:
        if event_index < 0 or event_index >= len(events):
            raise IndexError(f"event-index {event_index} is out of range for {len(events)} events")
        return events[event_index]

    sorted_events = sorted(events, key=lambda event: event.mean_distance)
    if event_rank < 0 or event_rank >= len(sorted_events):
        raise IndexError(f"event-rank {event_rank} is out of range for {len(sorted_events)} events")
    return sorted_events[event_rank]


def _plot_event(  # noqa: PLR0913
    trajectory_physical: np.ndarray,
    aligned_rpo: np.ndarray,
    event: ShadowingEvent,
    plot_start_timestep: int,
    resolution: int,
    cmap: str,
) -> plt.Figure:
    """Create a two-panel trajectory/RPO comparison figure."""
    plot_end_timestep = plot_start_timestep + trajectory_physical.shape[0]
    duration_timestep = event.end_timestep - event.start_timestep

    trajectory_time = np.arange(plot_start_timestep, plot_end_timestep) * TRAJECTORY_DT
    relative_time = (
        np.arange(plot_start_timestep, plot_end_timestep) - event.start_timestep
    ) * TRAJECTORY_DT
    space_axis = np.linspace(0, DOMAIN_SIZE, resolution, endpoint=False)

    vmin = min(trajectory_physical.min(), aligned_rpo.min())
    vmax = max(trajectory_physical.max(), aligned_rpo.max())

    figure, axes = plt.subplots(2, 1, figsize=(12, 7))

    image_top = axes[0].pcolormesh(
        trajectory_time,
        space_axis,
        trajectory_physical.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Space")
    axes[0].set_title("Chaotic Trajectory")
    event_start_time = event.start_timestep * TRAJECTORY_DT
    event_end_time = event.end_timestep * TRAJECTORY_DT
    axes[0].axvline(event_start_time, color="black", linestyle="--", linewidth=1.5)
    axes[0].axvline(event_end_time, color="black", linestyle="--", linewidth=1.5)
    figure.colorbar(image_top, ax=axes[0], label="u(x,t)")

    image_bottom = axes[1].pcolormesh(
        relative_time,
        space_axis,
        aligned_rpo.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_xlabel("Time Relative to Event Start")
    axes[1].set_ylabel("Space")
    axes[1].set_title(f"RPO {event.rpo_index} (aligned)")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].axvline(duration_timestep * TRAJECTORY_DT, color="black", linestyle="--", linewidth=1.5)
    figure.colorbar(image_bottom, ax=axes[1], label="u(x,t)")

    duration_time = duration_timestep * TRAJECTORY_DT
    figure.suptitle(
        f"Shadowing Event: RPO {event.rpo_index}, duration={duration_timestep} steps "
        f"({duration_time:.1f} time units), mean distance={event.mean_distance:.3f}",
        y=1.02,
    )
    plt.tight_layout()
    return figure


def main() -> None:
    """Run visualization command."""
    parser = build_parser()
    arguments = parser.parse_args()

    print(f"Loading results from {arguments.input}...")
    metadata, initial_state, events = load_results(arguments.input)
    resolution = metadata.spatial_resolution
    trajectory = KSTrajectory.from_initial_state(
        initial_state, TRAJECTORY_DT, metadata.trajectory_steps + 1, resolution
    )

    if not events:
        print("No events found in this file.")
        return

    detector_type = metadata.detector_type
    rpo_file = Path(metadata.rpo_file)

    print(f"  Detector type: {detector_type}")
    print(
        f"  Trajectory: {len(trajectory)} timesteps "
        f"({len(trajectory) * TRAJECTORY_DT:.0f} time units)"
    )
    print(f"  Events: {len(events)}")

    selected_event = select_event(events, arguments.event_rank, arguments.event_index)

    print(f"Loading RPOs from {rpo_file}...")
    rpos = load_all_rpos(rpo_file)
    rpo = rpos[selected_event.rpo_index]

    duration_timestep = selected_event.end_timestep - selected_event.start_timestep
    print(
        f"Selected event: RPO={selected_event.rpo_index}, start={selected_event.start_timestep}, "
        f"end={selected_event.end_timestep}, duration={duration_timestep}, "
        f"mean_distance={selected_event.mean_distance:.4f}"
    )

    context_timestep = int(duration_timestep * arguments.context_fraction)
    plot_start_timestep = max(0, selected_event.start_timestep - context_timestep)
    plot_end_timestep = min(len(trajectory), selected_event.end_timestep + context_timestep)

    minimum_window = duration_timestep + 2 * context_timestep
    if plot_end_timestep - plot_start_timestep < minimum_window:
        if plot_start_timestep > 0:
            plot_start_timestep = max(0, plot_end_timestep - minimum_window)
        elif plot_end_timestep < len(trajectory):
            plot_end_timestep = min(len(trajectory), plot_start_timestep + minimum_window)

    trajectory_slice = trajectory[plot_start_timestep:plot_end_timestep]
    trajectory_physical = trajectory_slice.to_physical()

    aligned_rpo = _align_rpo_to_window(
        rpo,
        selected_event,
        plot_start_timestep,
        plot_end_timestep,
        resolution,
    )

    figure = _plot_event(
        trajectory_physical,
        aligned_rpo,
        selected_event,
        plot_start_timestep,
        resolution,
        arguments.cmap,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(arguments.output, dpi=arguments.dpi, bbox_inches="tight")

    if arguments.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"Saved visualization to {arguments.output}")


if __name__ == "__main__":
    main()
