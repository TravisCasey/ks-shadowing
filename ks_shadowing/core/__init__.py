"""Core infrastructure shared by SSA and PHA shadowing detection algorithms."""

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import DOMAIN_SIZE, ksint
from ks_shadowing.core.rpo import RPO, load_all_rpos
from ks_shadowing.core.trajectory import (
    DEFAULT_CHUNK_SIZE,
    KSTrajectory,
    shift_distances_sq,
)

TRAJECTORY_DT: float = 0.02
"""Fixed integration timestep for chaotic trajectories."""

__all__: list[str] = [
    "DEFAULT_CHUNK_SIZE",
    "DOMAIN_SIZE",
    "RPO",
    "TRAJECTORY_DT",
    "KSTrajectory",
    "ShadowingEvent",
    "ksint",
    "load_all_rpos",
    "shift_distances_sq",
]
