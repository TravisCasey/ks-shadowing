"""State Space Approach (SSA) for shadowing detection.

Detects shadowing by computing L2 distances between trajectory snapshots and
RPO phases in physical space, using FFT cross-correlation to optimize over
spatial shifts. Events are extracted via connected components analysis.
"""

from ks_shadowing.ssa.detector import SSADetector

__all__: list[str] = ["SSADetector"]
