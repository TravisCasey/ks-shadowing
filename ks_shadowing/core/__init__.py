"""Core infrastructure shared by SSA and PHA shadowing detection algorithms."""

from ks_shadowing.core.event import ShadowingEvent
from ks_shadowing.core.integrator import DOMAIN_SIZE, ksint
from ks_shadowing.core.rpo import RPO, load_rpos
from ks_shadowing.core.trajectory import (
    DEFAULT_CHUNK_SIZE,
    KSTrajectory,
    shift_distances_sq,
)

INTEGRATION_DT: float = 0.02
"""Native ETDRK4 integration timestep.

The KS equation is always integrated at this step. Trajectories may be
stored at a coarser sampling step via the ``save_interval`` parameter on
:func:`~ks_shadowing.core.integrator.ksint` and
:meth:`~ks_shadowing.core.trajectory.KSTrajectory.from_initial_state`; the
resulting :attr:`~ks_shadowing.core.trajectory.KSTrajectory.dt` is then
``INTEGRATION_DT * save_interval``.
"""

__all__: list[str] = [
    "DEFAULT_CHUNK_SIZE",
    "DOMAIN_SIZE",
    "INTEGRATION_DT",
    "RPO",
    "KSTrajectory",
    "ShadowingEvent",
    "ksint",
    "load_rpos",
    "shift_distances_sq",
]
