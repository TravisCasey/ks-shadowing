"""Matched event computation for comparing SSA and PHA detection results."""

from collections import defaultdict
from dataclasses import dataclass

from ks_shadowing.core.event import ShadowingEvent


@dataclass(frozen=True, slots=True)
class EventMatch:
    """A matched group of SSA and PHA shadowing events on one RPO.

    A match is a connected component of the bipartite overlap graph: SSA and PHA
    events with the same ``rpo_index`` are linked whenever their
    ``[start_timestep, end_timestep)`` ranges overlap, and the match contains
    every event reachable through such links.

    Attributes
    ----------
    ssa_events : list[ShadowingEvent]
        The match's SSA events, sorted by ``start_timestep``. Non-empty.
    pha_events : list[ShadowingEvent]
        The match's PHA events, sorted by ``start_timestep``. Non-empty.
    ssa_length : int
        Timesteps covered by the SSA composite window.
    pha_length : int
        Timesteps covered by the PHA composite window.
    intersection_length : int
        Timesteps covered by both composite windows.
    union_length : int
        Timesteps covered by either composite window; equals
        ``ssa_length + pha_length - intersection_length``.
    """

    ssa_events: list[ShadowingEvent]
    pha_events: list[ShadowingEvent]
    ssa_length: int
    pha_length: int
    intersection_length: int
    union_length: int


def _covered_length(intervals: list[tuple[int, int]]) -> int:
    """Total timesteps covered by a union of half-open intervals."""
    total = 0
    covered_until = -1
    for start, end in sorted(intervals):
        if start > covered_until:
            total += end - start
            covered_until = end
        elif end > covered_until:
            total += end - covered_until
            covered_until = end
    return total


def match_events(
    ssa_events: list[ShadowingEvent],
    pha_events: list[ShadowingEvent],
) -> list[EventMatch]:
    """Group SSA and PHA events into matches by overlap on each RPO.

    Builds the bipartite overlap graph made up of SSA and PHA events with the
    same ``rpo_index``, linked whenever their ``[start_timestep, end_timestep)``
    ranges overlap, and returns one :class:`EventMatch` per connected component.
    An event that overlaps no event of the other method appears in no match.

    Parameters
    ----------
    ssa_events : list[ShadowingEvent]
        Events from SSA detection.
    pha_events : list[ShadowingEvent]
        Events from PHA detection.

    Returns
    -------
    list[EventMatch]
        One match per connected component, sorted by ``rpo_index`` and then by
        the first timestep either composite window covers.
    """
    pha_by_rpo: dict[int, list[int]] = defaultdict(list)
    for pha_index, event in enumerate(pha_events):
        pha_by_rpo[event.rpo_index].append(pha_index)

    # Union-find over ("s", index) / ("p", index) nodes; only events with at
    # least one overlap edge enter the forest.
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for ssa_index, ssa_event in enumerate(ssa_events):
        for pha_index in pha_by_rpo.get(ssa_event.rpo_index, []):
            pha_event = pha_events[pha_index]
            overlap = min(ssa_event.end_timestep, pha_event.end_timestep) - max(
                ssa_event.start_timestep, pha_event.start_timestep
            )
            if overlap <= 0:
                continue
            ssa_node, pha_node = ("s", ssa_index), ("p", pha_index)
            parent.setdefault(ssa_node, ssa_node)
            parent.setdefault(pha_node, pha_node)
            ssa_root, pha_root = find(ssa_node), find(pha_node)
            if ssa_root != pha_root:
                parent[ssa_root] = pha_root

    members: dict[tuple[str, int], tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for node in parent:
        side, index = node
        members[find(node)][0 if side == "s" else 1].append(index)

    matches: list[EventMatch] = []
    for ssa_indices, pha_indices in members.values():
        ssa_side = sorted(
            (ssa_events[index] for index in ssa_indices),
            key=lambda event: event.start_timestep,
        )
        pha_side = sorted(
            (pha_events[index] for index in pha_indices),
            key=lambda event: event.start_timestep,
        )
        ssa_intervals = [(event.start_timestep, event.end_timestep) for event in ssa_side]
        pha_intervals = [(event.start_timestep, event.end_timestep) for event in pha_side]
        ssa_length = _covered_length(ssa_intervals)
        pha_length = _covered_length(pha_intervals)
        union_length = _covered_length(ssa_intervals + pha_intervals)
        matches.append(
            EventMatch(
                ssa_events=ssa_side,
                pha_events=pha_side,
                ssa_length=ssa_length,
                pha_length=pha_length,
                intersection_length=ssa_length + pha_length - union_length,
                union_length=union_length,
            )
        )
    matches.sort(
        key=lambda match: (
            match.ssa_events[0].rpo_index,
            min(match.ssa_events[0].start_timestep, match.pha_events[0].start_timestep),
        )
    )
    return matches
