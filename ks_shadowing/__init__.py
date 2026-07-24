"""RPO Shadowing detection for Kuramoto-Sivashinsky system."""

from ks_shadowing import pha, ssa
from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import DOMAIN_SIZE
from ks_shadowing.core.matching import MatchedEvent, match_events
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
    select_event_by_rank,
)

# Keep in sync with ks_shadowing.core.INTEGRATION_DT.
INTEGRATION_DT: float = 0.02
"""Native ETDRK4 integration timestep. See
:data:`ks_shadowing.core.INTEGRATION_DT` for full documentation."""

__all__: list[str] = [
    "DOMAIN_SIZE",
    "INTEGRATION_DT",
    "RPO",
    "DetectionMetadata",
    "DetectionResult",
    "KSTrajectory",
    "MatchedEvent",
    "ShadowingEvent",
    "align_rpo_to_window",
    "assert_same_trajectory",
    "events_to_union_mask",
    "load_results",
    "load_rpos",
    "match_events",
    "pha",
    "save_results",
    "select_event_by_rank",
    "shift_distances_sq",
    "ssa",
]
