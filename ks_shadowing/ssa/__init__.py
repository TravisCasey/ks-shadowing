"""State Space Approach (SSA) for shadowing detection."""

from ks_shadowing.ssa.detection import auto_detect, compute_min_distances, detect

__all__: list[str] = ["auto_detect", "compute_min_distances", "detect"]
