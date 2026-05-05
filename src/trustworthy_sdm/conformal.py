"""Conformal calibration of contaminated-ensemble intervals.

The paper's positive result. Documented in Paper A:

  Standard conformal prediction, applied per cell with leave-one-basin-out
  (LOBO) spatial folds, restores near-nominal coverage to the contaminated
  ensemble's intervals against the deterministic benchmark surface.

This module implements:

* Per-pixel non-conformity scores against the benchmark prediction.
* The conformal correction quantile (with finite-sample correction).
* Application of the correction to produce calibrated intervals.
* LOBO cross-validation that aggregates corrected-coverage and width-inflation
  across the entity's basins.

Designed to operate on data we already have on disk — no model refitting.
We reuse the regenerated 30-replicate surfaces (data/replicate_surfaces) and
the companion paper's deterministic benchmark surfaces (data/results/grid_b_full).

Per Lucian's review: framing is "conformal as a fix for ensemble
miscalibration," not "conformal width as a contamination signal." Width
inflation is reported as a cost-of-correction, not a diagnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_sdm.analysis import (
    ALPHA,
    benchmark_for,
    load_ensemble,
)
from trustworthy_sdm.io import CellID, GridBPaths

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-pixel scores

def nonconformity_scores(
    bench: pd.Series,
    lo: pd.Series,
    hi: pd.Series,
) -> pd.Series:
    """Per-pixel non-conformity: distance from benchmark to predicted interval.

    Standard quantile-based score: ``s_i = max(lo_i - bench_i, bench_i - hi_i, 0)``.
    Zero when bench is inside ``[lo, hi]``. Positive when bench is outside;
    larger means farther outside.

    All inputs must share the same index. NaN rows are dropped before return.
    """
    aligned = pd.concat({"bench": bench, "lo": lo, "hi": hi}, axis=1).dropna()
    below = aligned["lo"] - aligned["bench"]   # positive if bench < lo
    above = aligned["bench"] - aligned["hi"]   # positive if bench > hi
    s = pd.concat([below, above], axis=1).max(axis=1).clip(lower=0.0)
    return s


def conformal_quantile(scores: pd.Series, alpha: float = ALPHA) -> float:
    """Finite-sample (1 - alpha) conformal quantile.

    The classic split-conformal formula: take the
    ``ceil((n + 1) * (1 - alpha)) / n`` -th order statistic of the scores.
    This gives the marginal-coverage guarantee under exchangeability.
    """
    s = np.asarray(scores)
    n = len(s)
    if n == 0:
        raise ValueError("conformal_quantile requires non-empty scores")
    # Quantile rank in [0, 1]
    q_rank = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q_rank, method="higher"))


# ---------------------------------------------------------------------------
# Per-cell LOBO conformal evaluation

@dataclass
class ConformalResult:
    """Conformal-correction outcome for one cell, aggregated across folds."""
    entity: str
    algorithm: str
    track: str
    level: int
    n_pixels_total: int
    n_basins: int
    coverage_uncorrected: float
    coverage_conformal: float
    median_width_uncorrected: float
    median_width_conformal: float
    median_q_hat: float
    # Per-fold coverage so we can show variance later if useful
    fold_coverages: tuple[float, ...]


def evaluate_cell_conformal(
    cell: CellID,
    surfaces_root: Path,
    paths: GridBPaths,
    *,
    alpha: float = ALPHA,
) -> ConformalResult:
    """LOBO conformal calibration for one cell.

    Procedure
    ---------
    1. Load the cell's 30-replicate ensemble. Compute per-pixel ``[lo, hi]``
       at the (1 - alpha) empirical quantiles.
    2. Load the deterministic benchmark and the basin_id mapping.
    3. For each basin b: hold b out as test, calibrate on the remaining
       basins, derive q_hat, apply correction to test pixels' intervals,
       record coverage and width.
    4. Aggregate: pixel-weighted mean coverage across all test folds is the
       reported corrected coverage; the median q_hat is reported as a
       summary of the correction magnitude.
    """
    ens = load_ensemble(cell, surfaces_root)
    bench = benchmark_for(cell.entity, cell.algorithm, cell.track, paths)

    # Need basin_id per subc_id. The merged metrics table has it; load_existing
    # benchmark surface doesn't carry it. We pull it from the master CSV via
    # the upstream package — but that's heavy on Mac. Use the basin_id_lookup
    # helper below (loaded once per session).
    basins = _basin_id_lookup(cell.entity)

    # Align all three on shared subc_ids
    shared = ens.index.intersection(bench.index).intersection(basins.index)
    if len(shared) == 0:
        raise RuntimeError(f"no shared subc_ids for {cell.short()}")

    ens_a = ens.loc[shared]
    bench_a = bench.loc[shared]
    basin_a = basins.loc[shared]

    # Original ensemble interval
    lo = ens_a.quantile(alpha / 2, axis=1)
    hi = ens_a.quantile(1 - alpha / 2, axis=1)

    unique_basins = basin_a.dropna().unique()
    if len(unique_basins) < 2:
        raise RuntimeError(
            f"{cell.short()} has only {len(unique_basins)} basin(s); "
            "LOBO conformal requires >= 2 basins"
        )

    # LOBO loop
    fold_results: list[dict] = []
    for test_basin in unique_basins:
        is_test = (basin_a == test_basin)
        cal_mask = ~is_test

        # Calibration scores
        cal_scores = nonconformity_scores(
            bench_a[cal_mask], lo[cal_mask], hi[cal_mask]
        )
        if len(cal_scores) == 0:
            continue

        q_hat = conformal_quantile(cal_scores, alpha=alpha)

        # Apply correction on test fold
        lo_corr = lo[is_test] - q_hat
        hi_corr = hi[is_test] + q_hat

        # Coverage on test fold
        bench_test = bench_a[is_test]
        inside_unc = (bench_test >= lo[is_test]) & (bench_test <= hi[is_test])
        inside_corr = (bench_test >= lo_corr) & (bench_test <= hi_corr)

        fold_results.append({
            "basin": test_basin,
            "n_test": int(is_test.sum()),
            "q_hat": q_hat,
            "cov_unc": float(inside_unc.mean()),
            "cov_corr": float(inside_corr.mean()),
            "width_unc": float((hi[is_test] - lo[is_test]).median()),
            "width_corr": float((hi_corr - lo_corr).median()),
        })

    fold_df = pd.DataFrame(fold_results)
    if len(fold_df) == 0:
        raise RuntimeError(f"no folds produced for {cell.short()}")

    # Pixel-weighted means
    w = fold_df["n_test"]
    cov_unc = float((fold_df["cov_unc"] * w).sum() / w.sum())
    cov_corr = float((fold_df["cov_corr"] * w).sum() / w.sum())
    width_unc = float(np.average(fold_df["width_unc"], weights=w))
    width_corr = float(np.average(fold_df["width_corr"], weights=w))

    return ConformalResult(
        entity=cell.entity,
        algorithm=cell.algorithm,
        track=cell.track,
        level=cell.level,
        n_pixels_total=int(len(shared)),
        n_basins=int(len(unique_basins)),
        coverage_uncorrected=cov_unc,
        coverage_conformal=cov_corr,
        median_width_uncorrected=width_unc,
        median_width_conformal=width_corr,
        median_q_hat=float(fold_df["q_hat"].median()),
        fold_coverages=tuple(fold_df["cov_corr"].astype(float).tolist()),
    )


# ---------------------------------------------------------------------------
# Basin-id lookup (cached)

_BASIN_LOOKUP_CACHE: dict[str, pd.Series] = {}


def _basin_id_lookup(entity: str) -> pd.Series:
    """Return a Series indexed by subc_id with basin_id values for one entity.

    The basin_id mapping is a property of the network topology, not the
    species — it's the same for all entities. We could derive it once. But
    accessible-area pixels differ between entities, so we cache per entity
    after first computation. The values come from the master CSV via the
    upstream package's _prepare_entity_data.
    """
    if entity in _BASIN_LOOKUP_CACHE:
        return _BASIN_LOOKUP_CACHE[entity]

    # Try the local Grid B per-cell parquet first — companion paper's
    # results carry basin_id alongside subc_id in the saved per-cell rows.
    # If unavailable on this filesystem, fall back to upstream's prep.
    s = _basin_lookup_from_local_results(entity)
    if s is None:
        s = _basin_lookup_from_upstream(entity)
    if s is None:
        raise RuntimeError(
            f"could not derive basin_id lookup for {entity!r}; "
            "neither local results nor upstream prep produced it"
        )

    _BASIN_LOOKUP_CACHE[entity] = s
    log.info("basin lookup for %s: %d subc_ids, %d basins",
             entity, len(s), s.nunique())
    return s


def _basin_lookup_from_local_results(entity: str) -> pd.Series | None:
    """Try to find basin_id mapping in local data/results/ first.

    The companion paper's grid_b_full per-cell parquets contain accessible
    area with basin_id for each subc_id — that's the cheapest source.
    """
    from trustworthy_sdm.io import ENTITY_NAME_TO_DIR
    edir = ENTITY_NAME_TO_DIR[entity]
    candidates = list(Path("data/results/grid_b_full").rglob(f"{edir}*accessible*.parquet"))
    candidates += list(Path("data/results/grid_b_full").rglob(f"{edir}*master_table*.parquet"))
    for c in candidates:
        try:
            df = pd.read_parquet(c)
            if "subc_id" in df.columns and "basin_id" in df.columns:
                return df.drop_duplicates("subc_id").set_index("subc_id")["basin_id"]
        except Exception:  # noqa: BLE001
            continue
    return None


_MASTER_TABLE_CACHE: dict[str, pd.DataFrame] = {}


def _get_master_cached() -> pd.DataFrame | None:
    """Load the master CSV once per process. Returns None if not available."""
    import os
    master_path = os.environ.get("TS_MASTER_CSV")
    if master_path is None or not Path(master_path).is_file():
        log.warning("TS_MASTER_CSV not set or file missing; cannot use upstream prep")
        return None

    if master_path in _MASTER_TABLE_CACHE:
        return _MASTER_TABLE_CACHE[master_path]

    try:
        from sdm_robustness.io import load_master_table  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning("sdm_robustness not installed: %s", e)
        return None

    log.info("loading master table from %s", master_path)
    master = load_master_table(Path(master_path))
    if isinstance(master, tuple):
        master = master[0]
    _MASTER_TABLE_CACHE[master_path] = master
    log.info("master table cached: %d rows", len(master))
    return master


def _basin_lookup_from_upstream(entity: str) -> pd.Series | None:
    """Fall back to upstream's _prepare_entity_data with cached master table."""
    try:
        import contextlib
        from sdm_robustness.execution import runner as _runner_mod  # type: ignore[import-not-found]
        from sdm_robustness.execution.runner import _prepare_entity_data  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning("sdm_robustness import failed: %s", e)
        return None

    master = _get_master_cached()
    if master is None:
        return None

    upstream_root = Path(_runner_mod.__file__).resolve().parents[3]
    with contextlib.chdir(upstream_root):
        prepared = _prepare_entity_data(master, entity)

    acc = prepared["accessible_area"]
    if "basin_id" not in acc.columns:
        log.warning("entity %s: accessible_area missing basin_id", entity)
        return None
    return acc.drop_duplicates("subc_id").set_index("subc_id")["basin_id"]
