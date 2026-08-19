"""RPO Shadowing detection for Kuramoto-Sivashinsky system."""

from ks_shadowing import pha, ssa
from ks_shadowing.core import INTEGRATION_DT
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import DOMAIN_SIZE
from ks_shadowing.core.matching import EventMatch, match_events
from ks_shadowing.core.results import (
    DetectionMetadata,
    DetectionResult,
    load_results,
    save_results,
)
from ks_shadowing.core.rpo import RPO, load_rpos
from ks_shadowing.core.trajectory import KSTrajectory, shift_distances_sq
from ks_shadowing.plotting import (
    align_rpo_to_window,
    assert_same_trajectory,
    events_to_union_mask,
)

__all__: list[str] = [
    "DOMAIN_SIZE",
    "INTEGRATION_DT",
    "RPO",
    "DetectionMetadata",
    "DetectionResult",
    "EventMatch",
    "KSTrajectory",
    "ShadowingEvent",
    "align_rpo_to_window",
    "assert_same_trajectory",
    "events_to_union_mask",
    "load_results",
    "load_rpos",
    "match_events",
    "pha",
    "save_results",
    "shift_distances_sq",
    "ssa",
]
