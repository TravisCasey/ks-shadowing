"""Persistent Homology Approach (PHA) for shadowing detection."""

from ks_shadowing.pha.detection import auto_detect, compute_min_distances, detect
from ks_shadowing.pha.pathfinding import connected_components
from ks_shadowing.pha.persistence import KSPersistenceTrajectory
from ks_shadowing.pha.wasserstein import wasserstein_matrix

__all__: list[str] = [
    "KSPersistenceTrajectory",
    "auto_detect",
    "compute_min_distances",
    "connected_components",
    "detect",
    "wasserstein_matrix",
]
