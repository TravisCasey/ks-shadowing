"""Core infrastructure shared by SSA and PHA shadowing detection algorithms."""

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import DOMAIN_SIZE, INTERLEAVED_COEFFS, ksint
from ks_shadowing.core.rpo import RPO, load_all_rpos
from ks_shadowing.core.transforms import (
    interleaved_to_complex,
    interleaved_to_physical,
    to_comoving_frame,
    to_physical,
)

TRAJECTORY_DT: float = 0.02
"""Fixed integration timestep for chaotic trajectories."""

DEFAULT_CHUNK_SIZE: int = 50000
"""Default number of trajectory timesteps to process at once.

Controls the memory-vectorization tradeoff for physical-space computations.
At resolution 2048, each chunk of 50000 steps uses approximately 780 MiB.
"""

__all__: list[str] = [
    "DOMAIN_SIZE",
    "INTERLEAVED_COEFFS",
    "RPO",
    "TRAJECTORY_DT",
    "ShadowingEvent",
    "interleaved_to_complex",
    "interleaved_to_physical",
    "ksint",
    "load_all_rpos",
    "to_comoving_frame",
    "to_physical",
]
