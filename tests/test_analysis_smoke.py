"""Smoke tests for the analysis module.

Most of these only run if the full panel has been rsynced to data/ on Mac.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trustworthy_sdm.io import CellID, GridBPaths


SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
RESULTS_ROOT = Path("data/results").resolve()


@pytest.fixture(scope="module")
def panel_available() -> None:
    if not SURFACES_ROOT.exists():
        pytest.skip("replicate_surfaces/ not present; rsync from VEGA first")
    if not RESULTS_ROOT.exists():
        pytest.skip("data/results not present; rsync from VEGA first")


def test_analyse_cell_torrentium_l3(panel_available) -> None:
    from trustworthy_sdm.analysis import analyse_cell

    paths = GridBPaths(root=RESULTS_ROOT)
    cell = CellID(
        "Austropotamobius torrentium (pooled)",
        "random_forest", "combined", "lowacc", 3,
    )
    result = analyse_cell(cell, SURFACES_ROOT, paths)
    assert result.n_replicates == 30
    assert result.n_pixels > 0
    assert 0.0 <= result.coverage <= 1.0
    assert result.median_width > 0
    # From pilot run: RF combined L3 should be ~0.884
    assert 0.85 < result.coverage < 0.92, f"unexpected coverage: {result.coverage}"


def test_analyse_panel_count(panel_available) -> None:
    from trustworthy_sdm.analysis import analyse_panel

    paths = GridBPaths(root=RESULTS_ROOT)
    df = analyse_panel(SURFACES_ROOT, paths)
    # 8 entities x 2 algos x 3 tracks x 3 levels = 144
    assert len(df) == 144, f"expected 144 cells, got {len(df)}"
    # All status sane
    assert (df["coverage"].between(0, 1)).all()
    assert (df["n_replicates"] == 30).all()
