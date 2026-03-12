"""Serialization helpers for CLI detection result files."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ks_shadowing.core import TRAJECTORY_DT
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import ksint

_EVENT_DTYPE = np.dtype(
    [
        ("rpo_index", np.int32),
        ("start_timestep", np.int32),
        ("end_timestep", np.int32),
        ("mean_distance", np.float64),
        ("min_distance", np.float64),
        ("start_phase", np.int32),
        ("shifts_end", np.int32),
    ]
)


@dataclass(frozen=True, slots=True)
class DetectionMetadata:
    """Metadata describing a detection run.

    Attributes
    ----------
    detector_type : str
        Detection method used: ``"SSA"`` or ``"PHA"``.
    seed : int
        RNG seed for trajectory generation. ``-1`` if not specified.
    spatial_resolution : int
        Number of spatial grid points.
    trajectory_steps : int
        Number of integration steps in the trajectory.
    initial_amplitude : float
        Scale factor for the random initial condition.
    min_duration : int
        Minimum event duration in timesteps.
    threshold : float
        Distance threshold used for detection.
    rpo_file : str
        Path to the RPO data file used for detection.
    threshold_quantile : float or None
        Quantile used for automatic threshold selection. ``None`` when
        ``threshold_mode`` is ``"manual"``.
    delay : int or None
        Time-delay embedding window size. ``None`` for SSA detection.
    """

    detector_type: str
    seed: int
    spatial_resolution: int
    trajectory_steps: int
    initial_amplitude: float
    min_duration: int
    threshold: float
    rpo_file: str
    threshold_quantile: float | None = None
    delay: int | None = None


def save_results(
    path: Path,
    metadata: DetectionMetadata,
    initial_state: np.ndarray,
    events: list[ShadowingEvent],
) -> None:
    """Save detection metadata, initial state, and events to an ``.h5`` file.

    The trajectory is not stored; it can be reproduced from the initial state
    and ``trajectory_steps`` metadata via
    :func:`~ks_shadowing.core.integrator.ksint`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("initial_state", data=initial_state)

        f.attrs["detector_type"] = metadata.detector_type
        f.attrs["seed"] = metadata.seed
        f.attrs["spatial_resolution"] = metadata.spatial_resolution
        f.attrs["trajectory_steps"] = metadata.trajectory_steps
        f.attrs["initial_amplitude"] = metadata.initial_amplitude
        f.attrs["min_duration"] = metadata.min_duration
        f.attrs["threshold"] = metadata.threshold
        f.attrs["rpo_file"] = metadata.rpo_file
        if metadata.threshold_quantile is not None:
            f.attrs["threshold_quantile"] = metadata.threshold_quantile
        if metadata.delay is not None:
            f.attrs["delay"] = metadata.delay

        shifts_list = [event.shifts for event in events]
        shifts_ends = (
            np.cumsum([len(s) for s in shifts_list], dtype=np.int32)
            if events
            else np.array([], dtype=np.int32)
        )

        event_records = np.array(
            [
                (
                    event.rpo_index,
                    event.start_timestep,
                    event.end_timestep,
                    event.mean_distance,
                    event.min_distance,
                    event.start_phase,
                    shifts_ends[i],
                )
                for i, event in enumerate(events)
            ],
            dtype=_EVENT_DTYPE,
        )
        f.create_dataset("events", data=event_records)

        shifts = np.concatenate(shifts_list) if shifts_list else np.array([], dtype=np.int32)
        f.create_dataset("shifts", data=shifts)


def load_results(path: Path) -> tuple[DetectionMetadata, np.ndarray, list[ShadowingEvent]]:
    """Load metadata, trajectory, and events from an ``.h5`` file.

    The trajectory is reproduced by integrating the saved initial state
    forward with :func:`~ks_shadowing.core.integrator.ksint`.
    """
    with h5py.File(path, "r") as f:
        attrs = f.attrs
        metadata = DetectionMetadata(
            detector_type=str(attrs["detector_type"]),
            seed=int(attrs["seed"]),
            spatial_resolution=int(attrs["spatial_resolution"]),
            trajectory_steps=int(attrs["trajectory_steps"]),
            initial_amplitude=float(attrs["initial_amplitude"]),
            min_duration=int(attrs["min_duration"]),
            threshold=float(attrs["threshold"]),
            rpo_file=str(attrs["rpo_file"]),
            threshold_quantile=(
                float(attrs["threshold_quantile"]) if "threshold_quantile" in attrs else None
            ),
            delay=int(attrs["delay"]) if "delay" in attrs else None,
        )

        initial_state = f["initial_state"][:].astype(np.float64, copy=False)
        event_records = f["events"][:]
        shifts = f["shifts"][:].astype(np.int32, copy=False)

    shifts_start = 0
    events: list[ShadowingEvent] = []
    for record in event_records:
        shifts_end = int(record["shifts_end"])
        events.append(
            ShadowingEvent(
                rpo_index=int(record["rpo_index"]),
                start_timestep=int(record["start_timestep"]),
                end_timestep=int(record["end_timestep"]),
                mean_distance=float(record["mean_distance"]),
                min_distance=float(record["min_distance"]),
                start_phase=int(record["start_phase"]),
                shifts=shifts[shifts_start:shifts_end],
            )
        )
        shifts_start = shifts_end

    trajectory = ksint(initial_state, TRAJECTORY_DT, metadata.trajectory_steps)
    return metadata, trajectory, events
