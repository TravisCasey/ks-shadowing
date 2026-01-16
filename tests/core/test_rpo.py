"""Tests for RPO data loading."""

from pathlib import Path

import numpy as np
import pytest

from ks_shadowing import RPO, load_all_rpos


class TestLoadRpo:
    def test_rpo_has_valid_fourier_coeffs(self, rpo_data_path: Path):
        """Loaded RPO has 30 Fourier coefficients."""
        rpo = RPO.load(rpo_data_path, 0)
        assert rpo.fourier_coeffs.shape == (30,)
        assert rpo.fourier_coeffs.dtype == np.float64

    def test_index_out_of_range_raises(self, rpo_data_path: Path):
        """Invalid index raises IndexError."""
        with pytest.raises(IndexError):
            RPO.load(rpo_data_path, -1)

        with pytest.raises(IndexError):
            RPO.load(rpo_data_path, 10000)


class TestLoadAllRpos:
    def test_indices_are_sequential(self, rpo_data_path: Path):
        """RPO indices are 0, 1, 2, ..., n-1."""
        rpos = load_all_rpos(rpo_data_path)
        assert all(isinstance(rpo, RPO) for rpo in rpos)

        indices = [rpo.index for rpo in rpos]
        assert indices == list(range(len(rpos)))
