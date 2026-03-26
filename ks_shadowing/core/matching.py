"""Matched event computation for comparing SSA and PHA detection results."""

from collections import defaultdict
from dataclasses import dataclass

from ks_shadowing.core.event import ShadowingEvent


@dataclass(frozen=True, slots=True)
class MatchedEvent:
    """A pair of overlapping SSA and PHA shadowing events on the same RPO.

    Attributes
    ----------
    ssa_event : ShadowingEvent
        The SSA detection event.
    pha_event : ShadowingEvent
        The PHA detection event.
    intersection_length : int
        Number of timesteps in the overlap of the two event ranges.
    union_length : int
        Number of timesteps in the union of the two event ranges.
    """

    ssa_event: ShadowingEvent
    pha_event: ShadowingEvent
    intersection_length: int
    union_length: int


def find_matched_events(
    ssa_events: list[ShadowingEvent],
    pha_events: list[ShadowingEvent],
) -> list[MatchedEvent]:
    """Find all matched pairs of SSA and PHA events that overlap in time.

    Two events match when they have the same ``rpo_index`` and their timestep
    ranges ``[start_timestep, end_timestep)`` have non-empty intersection.

    Parameters
    ----------
    ssa_events : list[ShadowingEvent]
        Events from SSA detection, sorted by ``start_timestep`` per RPO.
    pha_events : list[ShadowingEvent]
        Events from PHA detection, sorted by ``start_timestep`` per RPO.

    Returns
    -------
    list[MatchedEvent]
        All matched pairs with their intersection and union lengths.
    """
    pha_by_rpo: dict[int, list[ShadowingEvent]] = defaultdict(list)
    for event in pha_events:
        pha_by_rpo[event.rpo_index].append(event)

    matches: list[MatchedEvent] = []
    for ssa_event in ssa_events:
        rpo_pha = pha_by_rpo.get(ssa_event.rpo_index)
        if rpo_pha is None:
            continue
        for pha_event in rpo_pha:
            intersection = min(ssa_event.end_timestep, pha_event.end_timestep) - max(
                ssa_event.start_timestep, pha_event.start_timestep
            )
            if intersection <= 0:
                continue
            ssa_length = ssa_event.end_timestep - ssa_event.start_timestep
            pha_length = pha_event.end_timestep - pha_event.start_timestep
            union = ssa_length + pha_length - intersection
            matches.append(
                MatchedEvent(
                    ssa_event=ssa_event,
                    pha_event=pha_event,
                    intersection_length=intersection,
                    union_length=union,
                )
            )
    return matches
