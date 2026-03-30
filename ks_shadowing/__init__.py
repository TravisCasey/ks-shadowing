"""RPO Shadowing detection for Kuramoto-Sivashinsky system."""

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.matching import MatchedEvent, find_matched_events
from ks_shadowing.core.rpo import RPO, load_all_rpos
from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha import PHADetector
from ks_shadowing.ssa import SSADetector

__all__: list[str] = [
    "RPO",
    "KSTrajectory",
    "MatchedEvent",
    "PHADetector",
    "SSADetector",
    "ShadowingEvent",
    "find_matched_events",
    "load_all_rpos",
]
