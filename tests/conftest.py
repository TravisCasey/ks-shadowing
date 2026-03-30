"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
RPO_FILE = DATA_DIR / "rpos_selected.npz"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
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
