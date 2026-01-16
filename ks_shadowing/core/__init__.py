"""Core infrastructure shared by SSA and PHA shadowing detection approaches."""

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO, load_all_rpos
from ks_shadowing.core.transforms import (
    interleaved_to_complex,
    l2_distance_all_shifts,
    to_physical,
)

# Spatial domain size for the Kuramoto-Sivashinsky equation.
DOMAIN_SIZE = 22.0

__all__: list[str] = [
    "DOMAIN_SIZE",
    "RPO",
    "ShadowingEvent",
    "interleaved_to_complex",
    "ksint",
    "l2_distance_all_shifts",
    "load_all_rpos",
    "to_physical",
]
