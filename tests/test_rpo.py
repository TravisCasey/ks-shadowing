"""Tests for RPO data loading."""

from pathlib import Path

import pytest

from ks_shadowing import load_rpo


class TestLoadRpo:
    def test_index_out_of_range_raises(self, rpo_mat_path: Path):
        """Invalid index raises IndexError."""
        with pytest.raises(IndexError):
            load_rpo(rpo_mat_path, -1)

        with pytest.raises(IndexError):
            load_rpo(rpo_mat_path, 10000)
