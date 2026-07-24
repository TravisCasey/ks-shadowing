"""Serialization helpers for CLI detection result files."""

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.trajectory import KSTrajectory

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
    """Run metadata serialized alongside detection events.

    Attributes
    ----------
    detector_type : str
        Either ``"SSA"`` or ``"PHA"``.
    min_duration : int
        Minimum event duration in timesteps.
    threshold : float
        Distance threshold used for detection.
    rpo_file : str
        Path to the RPO data file used for detection.
    spatial_resolution : int
        Number of physical-space grid points the trajectory was loaded at for
        detection. Required to interpret ``ShadowingEvent.shifts``, which are
        recorded in grid cells.
    elapsed_seconds : float
        Wall-clock seconds spent inside the detect/auto_detect call that
        produced this result.
    downsample : int
        Sampling stride used during trajectory integration. The trajectory was
        integrated at the native ``INTEGRATION_DT`` and every ``downsample``-th
        row was retained, yielding an effective timestep of
        ``downsample * INTEGRATION_DT``. Per-RPO trajectories used the same
        stride. Defaults to ``1``.
    native : bool
        When ``True``, RPO trajectories were built by reordering all native rows
        with the stride-downsample permutation rather than slicing every Nth
        row. Defaults to ``False``.
    threshold_quantile : float or None
        Quantile used for automatic threshold selection. ``None`` when
        ``threshold`` was supplied manually.
    delay : int
        Time-delay embedding window size for PHA. ``1`` (default) means no
        temporal embedding. Recorded as ``1`` for SSA results.
    max_derivative_order : int
        Highest spatial-derivative order used for PHA. ``0`` (default) means
        only the field itself. Recorded as ``0`` for SSA results.
    """

    detector_type: str
    min_duration: int
    threshold: float
    rpo_file: str
    spatial_resolution: int
    elapsed_seconds: float
    downsample: int = 1
    native: bool = False
    threshold_quantile: float | None = None
    delay: int = 1
    max_derivative_order: int = 0


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Return value of the SSA and PHA ``detect`` and ``auto_detect`` functions.

    Attributes
    ----------
    events : list[ShadowingEvent]
        Detected events sorted by ``(start_timestep, rpo_index)``.
    threshold : float
        Distance threshold used for detection: the value passed to ``detect``,
        or the quantile-selected value chosen by ``auto_detect``.
    order_scales : NDArray[np.float64] or None
        Per-derivative-order scale factors applied during PHA detection.
        ``None`` unless PHA detection ran with per-order rescaling enabled.
    """

    events: list[ShadowingEvent]
    threshold: float
    order_scales: NDArray[np.float64] | None = None


def save_results(
    path: str | os.PathLike[str],
    metadata: DetectionMetadata,
    events: list[ShadowingEvent],
    *,
    trajectory_path: str | os.PathLike[str],
) -> None:
    """Save detection metadata and events to an ``.h5`` file.

    The trajectory itself is stored separately via
    :meth:`~ks_shadowing.core.trajectory.KSTrajectory.save` and must be a
    sibling of ``path`` (same parent directory). Only its filename is recorded
    in ``attrs["trajectory_path"]``; on load the trajectory is read from
    ``path.parent / attrs["trajectory_path"]``.

    Parameters
    ----------
    path : str or os.PathLike
        Destination ``.h5`` path. Parent directories are created if missing.
    metadata : DetectionMetadata
        Run metadata serialized to file-level attributes.
    events : list[ShadowingEvent]
        Detected events. Variable-length ``shifts`` are concatenated into a
        single dataset and indexed by per-event ``shifts_end`` offsets in the
        events table.
    trajectory_path : str or os.PathLike
        Path to the trajectory file. Must live in the same directory as
        ``path``.

    Raises
    ------
    ValueError
        If ``trajectory_path`` is not a sibling of ``path``.
    """
    path = Path(path)
    trajectory_path = Path(trajectory_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory_path.parent.resolve() != path.parent.resolve():
        raise ValueError(
            f"trajectory_path must be a sibling of path; got "
            f"trajectory={trajectory_path}, result={path}"
        )

    with h5py.File(path, "w") as f:
        f.attrs["detector_type"] = metadata.detector_type
        f.attrs["min_duration"] = metadata.min_duration
        f.attrs["threshold"] = metadata.threshold
        f.attrs["rpo_file"] = metadata.rpo_file
        f.attrs["spatial_resolution"] = metadata.spatial_resolution
        f.attrs["elapsed_seconds"] = metadata.elapsed_seconds
        f.attrs["downsample"] = metadata.downsample
        f.attrs["native"] = metadata.native
        f.attrs["delay"] = metadata.delay
        f.attrs["max_derivative_order"] = metadata.max_derivative_order
        f.attrs["trajectory_path"] = trajectory_path.name
        if metadata.threshold_quantile is not None:
            f.attrs["threshold_quantile"] = metadata.threshold_quantile

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


def load_results(
    path: str | os.PathLike[str],
) -> tuple[DetectionMetadata, KSTrajectory, list[ShadowingEvent]]:
    """Load metadata, trajectory, and events for a result file.

    The trajectory is loaded from ``path.parent / attrs["trajectory_path"]``
    (the attribute stores just the filename; see :func:`save_results`) via
    :meth:`~ks_shadowing.core.trajectory.KSTrajectory.load`, using
    ``metadata.spatial_resolution`` for the grid.

    Parameters
    ----------
    path : str or os.PathLike
        HDF5 file produced by :func:`save_results`.

    Returns
    -------
    metadata : DetectionMetadata
        Run metadata reconstructed from file-level attributes.
    trajectory : KSTrajectory
        Trajectory associated with the run, at the resolution recorded
        in ``metadata.spatial_resolution``.
    events : list[ShadowingEvent]
        Detected events with their ``shifts`` arrays sliced from the
        concatenated dataset.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = f.attrs
        metadata = DetectionMetadata(
            detector_type=str(attrs["detector_type"]),
            min_duration=int(attrs["min_duration"]),
            threshold=float(attrs["threshold"]),
            rpo_file=str(attrs["rpo_file"]),
            spatial_resolution=int(attrs["spatial_resolution"]),
            elapsed_seconds=float(attrs["elapsed_seconds"]),
            downsample=int(attrs["downsample"]) if "downsample" in attrs else 1,
            native=bool(attrs["native"]) if "native" in attrs else False,
            threshold_quantile=(
                float(attrs["threshold_quantile"]) if "threshold_quantile" in attrs else None
            ),
            delay=int(attrs["delay"]) if "delay" in attrs else 1,
            max_derivative_order=(
                int(attrs["max_derivative_order"]) if "max_derivative_order" in attrs else 0
            ),
        )
        trajectory_filename = str(attrs["trajectory_path"])
        event_records = f["events"][:]
        shifts = f["shifts"][:].astype(np.int32, copy=False)

    trajectory_path = path.parent / trajectory_filename
    trajectory = KSTrajectory.load(trajectory_path, resolution=metadata.spatial_resolution)

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

    return metadata, trajectory, events
