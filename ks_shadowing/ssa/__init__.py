"""State Space Approach (SSA) for shadowing detection.

SSA detects shadowing by computing L2 distances in physical space between
trajectory snapshots and RPO phases, with spatial shift optimization via
FFT cross-correlation. The algorithm enforces spatial continuity: consecutive
spatial shifts can differ by at most 1 position.
"""

from ks_shadowing.ssa.detector import SSADetector

__all__: list[str] = ["SSADetector"]
