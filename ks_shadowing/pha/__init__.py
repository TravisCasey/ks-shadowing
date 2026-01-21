"""Persistent Homology Approach (PHA) for shadowing detection.

PHA detects shadowing using Wasserstein distances between persistence diagrams
of trajectory and RPO snapshots. Unlike SSA, PHA quotients out the continuous
spatial symmetry via persistence diagrams, eliminating the need for explicit
shift optimization.
"""

from ks_shadowing.pha.detector import PHADetector

__all__: list[str] = ["PHADetector"]
