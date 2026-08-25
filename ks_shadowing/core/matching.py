"""Matched event computation for comparing SSA and PHA detection results."""

from collections import defaultdict
from dataclasses import dataclass

from ks_shadowing.core.event import ShadowingEvent


@dataclass(frozen=True, slots=True)
class EventMatch:
    """A matched group of SSA and PHA shadowing events on one RPO.

    SSA and PHA events with the same ``rpo_index`` are linked whenever their
    ``[start_timestep, end_timestep)`` ranges overlap. A transitive match is a
    connected component of this bipartite overlap graph and contains every
    event reachable through such links; a non-transitive match is a single
    overlap edge, holding exactly one event per side.

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


def _connected_components(
    edges: list[tuple[int, int]],
) -> list[tuple[list[int], list[int]]]:
    """Group overlap edges into connected components of the bipartite graph.

    Returns one ``(ssa_indices, pha_indices)`` pair per component.
    """
    # Union-find over ("s", index) / ("p", index) nodes; only events with at
    # least one overlap edge enter the forest.
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for ssa_index, pha_index in edges:
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
    return list(members.values())


def match_events(
    ssa_events: list[ShadowingEvent],
    pha_events: list[ShadowingEvent],
    *,
    transitive: bool = True,
) -> list[EventMatch]:
    """Group SSA and PHA events into matches by overlap on each RPO.

    Builds the bipartite overlap graph made up of SSA and PHA events with the
    same ``rpo_index``, linked whenever their ``[start_timestep, end_timestep)``
    ranges overlap. With ``transitive=True`` (the default), returns one
    :class:`EventMatch` per connected component; with ``transitive=False``,
    returns one :class:`EventMatch` per overlap edge, each holding exactly one
    event per side, so an event overlapping several events of the other method
    appears in several matches. An event that overlaps no event of the other
    method appears in no match, either way.

    Parameters
    ----------
    ssa_events : list[ShadowingEvent]
        Events from SSA detection.
    pha_events : list[ShadowingEvent]
        Events from PHA detection.
    transitive : bool, optional
        Whether to group matches into connected components (True, default) or
        report each overlapping pair separately (False). Keyword-only.

    Returns
    -------
    list[EventMatch]
        One match per connected component or per overlap edge, sorted by
        ``rpo_index`` and then by the first timestep either composite window
        covers.
    """
    pha_by_rpo: dict[int, list[int]] = defaultdict(list)
    for pha_index, event in enumerate(pha_events):
        pha_by_rpo[event.rpo_index].append(pha_index)

    edges: list[tuple[int, int]] = []
    for ssa_index, ssa_event in enumerate(ssa_events):
        for pha_index in pha_by_rpo.get(ssa_event.rpo_index, []):
            pha_event = pha_events[pha_index]
            overlap = min(ssa_event.end_timestep, pha_event.end_timestep) - max(
                ssa_event.start_timestep, pha_event.start_timestep
            )
            if overlap > 0:
                edges.append((ssa_index, pha_index))

    if transitive:
        groups = _connected_components(edges)
    else:
        groups = [([ssa_index], [pha_index]) for ssa_index, pha_index in edges]

    matches: list[EventMatch] = []
    for ssa_indices, pha_indices in groups:
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
