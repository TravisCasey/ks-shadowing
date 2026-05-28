"""CLI entry point for F_agree / F_disagree vs PHA delay (paper Figure 6).

At each trajectory timestep, each method reports whether the trajectory is
shadowing some RPO (union across RPOs). F_agree counts timesteps where both
methods agree on the shadowing status (both shadowing or both not shadowing),
and F_disagree is split into SSA-only and PHA-only contributions. All three
fractions sum to 1 and are normalized by the total trajectory length.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.results import load_results

DEFAULT_OUTPUT = Path("plots/coverage_vs_delay.png")
DEFAULT_DPI = 150


def build_parser() -> ArgumentParser:
    """Build CLI parser for ``ks-coverage``."""
    parser = ArgumentParser(
        description=(
            "Plot F_agree and F_disagree as a function of PHA delay. "
            "At each timestep, each method reports shadowing/not-shadowing "
            "(union across RPOs); agreement counts both cases of matching status."
        ),
    )
    parser.add_argument(
        "--ssa-input",
        type=Path,
        required=True,
        help="SSA results HDF5 file.",
    )
    parser.add_argument(
        "--pha-inputs",
        type=Path,
        nargs="+",
        required=True,
        help="PHA results HDF5 files (one per delay value).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output image DPI (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show figure interactively.",
    )
    return parser


def _events_to_union_mask(events: list[ShadowingEvent], num_timesteps: int) -> NDArray[np.bool_]:
    """Build 1D boolean mask of timesteps covered by any event (union across RPOs)."""
    mask = np.zeros(num_timesteps, dtype=np.bool_)
    for event in events:
        mask[event.start_timestep : event.end_timestep] = True
    return mask


def _plot_coverage(
    delays: NDArray[np.int64],
    f_agree: NDArray[np.float64],
    f_ssa_only: NDArray[np.float64],
    f_pha_only: NDArray[np.float64],
) -> plt.Figure:
    """Build the two-panel agreement/disagreement figure for ``ks-coverage``.

    Parameters
    ----------
    delays : NDArray[np.int64]
        PHA delay values, sorted ascending.
    f_agree : NDArray[np.float64]
        Per-delay fraction of timesteps where SSA and PHA agree on shadowing.
    f_ssa_only : NDArray[np.float64]
        Per-delay fraction of timesteps where only SSA reports shadowing.
    f_pha_only : NDArray[np.float64]
        Per-delay fraction of timesteps where only PHA reports shadowing.
    """
    figure, (ax_agree, ax_disagree) = plt.subplots(1, 2, figsize=(10, 4))

    ax_agree.plot(delays, f_agree, "o-", color="tab:blue")
    ax_agree.set_xlabel("Delay")
    ax_agree.set_ylabel(r"$F_{\mathrm{agree}}$")
    ax_agree.set_xticks(delays)
    ax_agree.set_title("Agreement")

    ax_disagree.plot(delays, f_ssa_only, "s-", color="tab:orange", label="SSA only")
    ax_disagree.plot(delays, f_pha_only, "^-", color="tab:green", label="PHA only")
    ax_disagree.set_xlabel("Delay")
    ax_disagree.set_ylabel(r"$F_{\mathrm{disagree}}$")
    ax_disagree.set_xticks(delays)
    ax_disagree.set_title("Disagreement")
    ax_disagree.legend()

    plt.tight_layout()
    return figure


def main() -> None:
    """Run coverage command."""
    parser = build_parser()
    arguments = parser.parse_args()

    _, ssa_trajectory, ssa_events = load_results(arguments.ssa_input)
    num_timesteps = ssa_trajectory.num_timesteps

    ssa_mask = _events_to_union_mask(ssa_events, num_timesteps)
    print(
        f"SSA: {len(ssa_events)} events, union covers {ssa_mask.sum()}/{num_timesteps} "
        f"timesteps ({ssa_mask.mean():.4f})"
    )

    delays: list[int] = []
    f_agree: list[float] = []
    f_ssa_only: list[float] = []
    f_pha_only: list[float] = []

    for pha_path in sorted(arguments.pha_inputs):
        pha_metadata, pha_trajectory, pha_events = load_results(pha_path)

        if pha_trajectory.num_timesteps != num_timesteps:
            raise ValueError(
                f"Trajectory lengths differ: {num_timesteps} vs "
                f"{pha_trajectory.num_timesteps} in {pha_path}"
            )
        if not np.array_equal(ssa_trajectory.modes, pha_trajectory.modes):
            raise ValueError(f"Trajectories differ between {arguments.ssa_input} and {pha_path}")

        delay = pha_metadata.delay
        if delay is None:
            raise ValueError(f"PHA file missing delay metadata: {pha_path}")

        pha_mask = _events_to_union_mask(pha_events, num_timesteps)

        agree = float((ssa_mask == pha_mask).mean())
        ssa_only = float((ssa_mask & ~pha_mask).mean())
        pha_only = float((~ssa_mask & pha_mask).mean())

        delays.append(delay)
        f_agree.append(agree)
        f_ssa_only.append(ssa_only)
        f_pha_only.append(pha_only)

        print(
            f"  delay={delay}: PHA {len(pha_events)} events, "
            f"PHA union {pha_mask.sum()}, "
            f"F_agree={agree:.4f}, F_ssa_only={ssa_only:.4f}, F_pha_only={pha_only:.4f}"
        )

    order = np.argsort(delays)
    figure = _plot_coverage(
        np.array(delays, dtype=np.int64)[order],
        np.array(f_agree, dtype=np.float64)[order],
        np.array(f_ssa_only, dtype=np.float64)[order],
        np.array(f_pha_only, dtype=np.float64)[order],
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=arguments.dpi, bbox_inches="tight")

    if arguments.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"Saved plot to {arguments.output}")


if __name__ == "__main__":
    main()
