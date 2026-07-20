"""Persistent Homology Approach (PHA) for shadowing detection."""

from ks_shadowing.pha.detection import auto_detect, compute_min_distances, detect
from ks_shadowing.pha.persistence import KSPersistenceTrajectory

__all__: list[str] = [
    "KSPersistenceTrajectory",
    "auto_detect",
    "compute_min_distances",
    "detect",
]
