"""Shared shadowing event representation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShadowingEvent:
    """A detected shadowing episode between a trajectory and an RPO.

    A shadowing event represents a contiguous time interval where a chaotic
    trajectory closely follows an RPO, with the RPO phase advancing by 1 at
    each trajectory timestep (with wraparound at the period boundary).

    The `start_time` is inclusive and `end_time` is exclusive.
    """

    rpo_index: int
    start_time: int
    end_time: int
    mean_distance: float
    min_distance: float

    @property
    def duration(self) -> int:
        """Number of timesteps in the shadowing event."""
        return self.end_time - self.start_time
