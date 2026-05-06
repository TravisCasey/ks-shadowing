"""Tests for RPO data loading."""

from pathlib import Path

import numpy as np

from ks_shadowing.core.rpo import load_rpos


def test_load_rpos_indices_and_modes(rpo_data_path: Path) -> None:
    """``load_rpos`` returns ``RPO`` instances with sequential ``index`` and
    ``modes`` of shape ``(17,)`` and dtype ``complex128``."""
    rpos = load_rpos(rpo_data_path)
    assert len(rpos) > 0
    assert [rpo.index for rpo in rpos] == list(range(len(rpos)))
    for rpo in rpos:
        assert rpo.modes.shape == (17,)
        assert rpo.modes.dtype == np.complex128
