"""Smoke tests for the conformal module.

Runs LOBO conformal on a single cell with known coverage from the panel
analysis. Verifies that:
  - The math runs end-to-end without errors.
  - Uncorrected coverage matches what `analyse_cell` produced (sanity check).
  - Conformal-corrected coverage is meaningfully closer to nominal than
    the uncorrected value.

Skipped if the panel data isn't present (fresh checkout).
"""

from __future__ import annotations

from pathlib import Path

import pytest


SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
RESULTS_ROOT = Path("data/results").resolve()


@pytest.fixture(scope="module")
def panel_available() -> None:
    if not SURFACES_ROOT.exists():
        pytest.skip("replicate_surfaces/ not present; rsync from VEGA first")
    if not RESULTS_ROOT.exists():
        pytest.skip("data/results not present; rsync from VEGA first")


def test_nonconformity_scores_basic() -> None:
    """Algebraic sanity: scores are 0 when bench in interval, > 0 outside."""
    import pandas as pd
    from trustworthy_sdm.conformal import nonconformity_scores

    bench = pd.Series([0.5, 0.1, 0.9, 0.5], index=[1, 2, 3, 4])
    lo = pd.Series([0.4, 0.3, 0.2, 0.6], index=[1, 2, 3, 4])
    hi = pd.Series([0.6, 0.5, 0.7, 0.8], index=[1, 2, 3, 4])

    s = nonconformity_scores(bench, lo, hi)
    # Pixel 1: 0.5 in [0.4, 0.6] -> 0
    # Pixel 2: 0.1 below 0.3      -> 0.2
    # Pixel 3: 0.9 above 0.7      -> 0.2
    # Pixel 4: 0.5 below 0.6      -> 0.1
    assert s.loc[1] == pytest.approx(0.0)
    assert s.loc[2] == pytest.approx(0.2)
    assert s.loc[3] == pytest.approx(0.2)
    assert s.loc[4] == pytest.approx(0.1)


def test_conformal_quantile_basic() -> None:
    """Quantile uses (1-alpha)(1+1/n) finite-sample correction."""
    import pandas as pd
    from trustworthy_sdm.conformal import conformal_quantile

    # n=10, alpha=0.05 -> rank = ceil(11 * 0.95) / 10 = 11/10 = 1.1, capped to 1.0
    # so quantile = max
    scores = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q = conformal_quantile(scores, alpha=0.05)
    assert q == pytest.approx(0.9)

    # n=99, alpha=0.05 -> rank = ceil(100 * 0.95) / 99 = 95/99 ≈ 0.9596
    # quantile method='higher' picks the smallest score >= 95th percentile
    scores2 = pd.Series([i / 100 for i in range(99)])
    q2 = conformal_quantile(scores2, alpha=0.05)
    # Should be around the 95th-96th value
    assert 0.94 < q2 < 0.97


def test_evaluate_cell_conformal_torrentium(panel_available) -> None:
    """Conformal on A. torrentium RF combined L10 should improve coverage."""
    from trustworthy_sdm.conformal import evaluate_cell_conformal
    from trustworthy_sdm.io import CellID, GridBPaths

    paths = GridBPaths(root=RESULTS_ROOT)
    cell = CellID(
        "Austropotamobius torrentium (pooled)",
        "random_forest", "combined", "lowacc", 10,
    )

    result = evaluate_cell_conformal(cell, SURFACES_ROOT, paths)

    # Sanity: structural
    assert result.n_pixels_total > 1000
    assert result.n_basins >= 2
    assert 0.0 <= result.coverage_uncorrected <= 1.0
    assert 0.0 <= result.coverage_conformal <= 1.0

    # Uncorrected should match what analyse_cell produced (~0.79 from
    # the full-panel summary). Allow generous tolerance because LOBO
    # weighting differs slightly from the global mean.
    assert 0.70 < result.coverage_uncorrected < 0.85, (
        f"unexpected uncorrected coverage: {result.coverage_uncorrected}"
    )

    # The headline claim: conformal correction moves coverage substantially
    # closer to nominal 0.95.
    assert result.coverage_conformal > result.coverage_uncorrected
    assert result.coverage_conformal > 0.88, (
        f"conformal correction too weak: {result.coverage_conformal}"
    )

    # The correction comes at a cost: wider intervals.
    assert result.median_width_conformal > result.median_width_uncorrected



def test_evaluate_panel_conformal_subset(panel_available) -> None:
    """Smoke test: run conformal panel evaluation on one entity.

    Validates the panel function's contract — returns DataFrame with
    expected columns, all coverage values in [0, 1], conformal coverage
    on average closer to nominal than uncorrected.
    """
    from trustworthy_sdm.analysis import evaluate_panel_conformal
    from trustworthy_sdm.io import GridBPaths

    paths = GridBPaths(root=RESULTS_ROOT)

    # Single-entity subset to keep the test fast — one entity x 2 algos x
    # 3 tracks x 3 levels = 18 cells. ~30 sec.
    df = evaluate_panel_conformal(
        SURFACES_ROOT, paths,
        entities=("Austropotamobius torrentium (pooled)",),
    )

    assert len(df) == 18, f"expected 18 cells, got {len(df)}"

    # Column contract
    expected_cols = {
        "entity", "algorithm", "track", "level",
        "n_pixels_total", "n_basins",
        "coverage_uncorrected", "coverage_conformal",
        "median_width_uncorrected", "median_width_conformal",
        "median_q_hat",
        "coverage_gap_pre", "coverage_gap_post",
        "width_inflation_factor",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"missing columns: {missing}"

    # Sanity bounds
    assert df["coverage_uncorrected"].between(0, 1).all()
    assert df["coverage_conformal"].between(0, 1).all()
    assert (df["n_basins"] >= 2).all()

    # Conformal correction moves coverage closer to nominal on average
    nominal = 0.95
    gap_pre = (nominal - df["coverage_uncorrected"]).abs().mean()
    gap_post = (nominal - df["coverage_conformal"]).abs().mean()
    assert gap_post < gap_pre, (
        f"conformal did not improve mean gap: pre={gap_pre:.4f}, post={gap_post:.4f}"
    )
