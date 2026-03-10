"""Core infrastructure shared by SSA and PHA shadowing detection algorithms."""

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import DOMAIN_SIZE, INTERLEAVED_COEFFS, ksint
from ks_shadowing.core.rpo import RPO, load_all_rpos
from ks_shadowing.core.transforms import (
    interleaved_to_complex,
    l2_distance_all_shifts,
    to_physical,
)

__all__: list[str] = [
    "DOMAIN_SIZE",
    "INTERLEAVED_COEFFS",
    "RPO",
    "ShadowingEvent",
    "interleaved_to_complex",
    "ksint",
    "l2_distance_all_shifts",
    "load_all_rpos",
    "to_physical",
]
