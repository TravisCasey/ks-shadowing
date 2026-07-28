"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ks_shadowing.core import INTEGRATION_DT
from ks_shadowing.core.rpo import RPO, load_rpos
from ks_shadowing.core.trajectory import KSTrajectory

DATA_DIR = Path(__file__).parent.parent / "data"
RPO_FILE = DATA_DIR / "rpos_selected.npz"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_initial_state(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Random valid initial condition (17 complex Fourier modes)."""
    modes = np.zeros(17, dtype=np.complex128)
    modes[1:16] = (rng.standard_normal(15) + 1j * rng.standard_normal(15)) * 0.1
    return modes


@pytest.fixture
def rpo_data_path() -> Path:
    """Path to the RPO .npz file."""
    if not RPO_FILE.exists():
        pytest.skip(f"RPO data file not found: {RPO_FILE}")
    return RPO_FILE


@pytest.fixture
def random_trajectory(rng: np.random.Generator) -> KSTrajectory:
    """Random KSTrajectory for tests that don't exercise the integrator."""
    modes = np.zeros((20, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((20, 15)) + 1j * rng.standard_normal((20, 15))) * 0.1
    return KSTrajectory(modes=modes, dt=0.02, resolution=64)


@pytest.fixture
def small_rpos(rpo_data_path: Path) -> list[RPO]:
    """First two RPOs of ``rpos_selected.npz`` (the shortest periods).

    Used by detection integration tests; keeps runtime small while still
    exercising the multi-RPO dispatch path.
    """
    return load_rpos(rpo_data_path)[:2]


@pytest.fixture
def short_trajectory(small_rpos: list[RPO]) -> KSTrajectory:
    """200-timestep trajectory at resolution 32 seeded from the shortest RPO."""
    rpo = small_rpos[0]
    return KSTrajectory.from_initial_state(
        rpo.modes, dt=INTEGRATION_DT, num_timesteps=200, resolution=32
    )
