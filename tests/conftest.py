"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
RPO_MAT_FILE = DATA_DIR / "ks22h02t100ED.mat"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    """Random valid initial condition (30 coefficients)."""
    return rng.standard_normal(30) * 0.1


@pytest.fixture
def rpo_mat_path() -> Path:
    """Path to the RPO .mat file."""
    if not RPO_MAT_FILE.exists():
        pytest.skip(f"RPO data file not found: {RPO_MAT_FILE}")
    return RPO_MAT_FILE
