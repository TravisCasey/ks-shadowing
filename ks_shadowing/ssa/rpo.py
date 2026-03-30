"""RPO trajectory representation for State Space Approach detection."""

from dataclasses import dataclass
from typing import Self

from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory


@dataclass
class _RPOStateSpace:
    """Precomputed RPO trajectory in spectral form for state-space detection.

    Holds both the source RPO metadata and its integrated trajectory as a
    :class:`~ks_shadowing.core.trajectory.KSTrajectory`. Used by the SSA
    detector for distance computation via
    :func:`~ks_shadowing.core.trajectory.shift_distances_sq`.

    Attributes
    ----------
    rpo : RPO
        Source RPO containing metadata (index, period, spatial_shift).
    trajectory : KSTrajectory
        Spectral trajectory over one period, with ``len(trajectory) ==
        time_steps``. Integrated with the RPO's specific timestep.
    """

    rpo: RPO
    trajectory: KSTrajectory

    @classmethod
    def from_rpo(cls, rpo: RPO, resolution: int) -> Self:
        """Integrate an RPO to a spectral trajectory over one period.

        Parameters
        ----------
        rpo : RPO
            Source RPO to integrate.
        resolution : int
            Number of physical-space grid points for the trajectory.
        """
        rpo_dt = rpo.period / rpo.time_steps
        trajectory = KSTrajectory.from_initial_state(
            rpo.fourier_coeffs, rpo_dt, rpo.time_steps + 1, resolution
        )[:-1]
        return cls(rpo=rpo, trajectory=trajectory)

    @property
    def time_steps(self) -> int:
        """Number of timesteps in one RPO period."""
        return self.rpo.time_steps

    @property
    def spatial_shift(self) -> float:
        """Spatial shift of the RPO over one period."""
        return self.rpo.spatial_shift

    @property
    def index(self) -> int:
        """Index of the source RPO."""
        return self.rpo.index

    @property
    def resolution(self) -> int:
        """Spatial resolution of the trajectory."""
        return self.trajectory.resolution
