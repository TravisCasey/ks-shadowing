"""RPO trajectory representation for State Space Approach detection."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ks_shadowing.core.integrator import ksint
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.transforms import interleaved_to_physical


@dataclass
class RPOStateSpace:
    """Precomputed RPO trajectory in physical space for state-space detection.

    Holds both the source RPO metadata and its integrated trajectory. Used
    by SSA detector for distance computation in physical space.

    Attributes:
        rpo: Source RPO containing metadata (index, period, spatial_shift).
        trajectory: Physical space trajectory over one period, shape
            `(time_steps, resolution)`. Integrated with the RPO's specific time
            step.
    """

    rpo: RPO
    trajectory: NDArray[np.float64]

    @classmethod
    def from_rpo(cls, rpo: RPO, resolution: int) -> Self:
        """Integrate an RPO and convert to physical space trajectory.

        Integrates the RPO for one full period using its native timestep,
        then transforms to physical space at the given resolution.
        """
        rpo_dt = rpo.period / rpo.time_steps
        fourier_trajectory = ksint(rpo.fourier_coeffs, rpo_dt, rpo.time_steps)[:-1]
        physical_trajectory = interleaved_to_physical(fourier_trajectory, resolution)
        return cls(rpo=rpo, trajectory=physical_trajectory)

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
        return self.trajectory.shape[1]
