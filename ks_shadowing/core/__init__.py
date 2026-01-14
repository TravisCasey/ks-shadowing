"""Core infrastructure shared by SSA and PHA shadowing detection approaches."""

from ks_shadowing.core.constants import DEFAULT_RESOLUTION, N_COEFFS
from ks_shadowing.core.detection import ShadowingEvent
from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO, load_all_rpos, load_rpo
from ks_shadowing.core.transforms import (
    interleaved_to_complex,
    l2_distance_all_shifts,
    to_physical,
)

__all__: list[str] = [
    "DEFAULT_RESOLUTION",
    "N_COEFFS",
    "RPO",
    "ShadowingEvent",
    "interleaved_to_complex",
    "ksint",
    "l2_distance_all_shifts",
    "load_all_rpos",
    "load_rpo",
    "to_physical",
]
