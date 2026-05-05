"""Full-panel coverage analysis for Paper A.

Loads regenerated ensemble surfaces, computes per-cell coverage and width
against the companion paper's deterministic benchmark surfaces, and produces
a long-format DataFrame that drives every Paper A figure.

Heavy lifting lives here so the notebook stays readable. Functions are
import-safe (no side effects on import).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_sdm.io import (
    DUAL_AXIS_ENTITIES,
    ENTITY_NAME_TO_DIR,
    FULL_PANEL_LOWACC_LEVELS,
    TRACKS,
    CellID,
    GridBPaths,
    load_existing_surface,
)

log = logging.getLogger(__name__)

ALPHA = 0.05  # 95% interval


# ---------------------------------------------------------------------------
# Per-cell statistics

@dataclass
class CellResult:
    """One cell's coverage and width result, plus diagnostics."""
    entity: str
    entity_dir: str
    algorithm: str
    track: str
    level: int
    n_replicates: int
    n_pixels: int
    coverage: float
    median_width: float
    mean_width: float
    median_signed_diff: float    # benchmark - ensemble_mean
    mean_abs_diff: float
    frac_diff_gt_0p1: float


def load_ensemble(cell: CellID, surfaces_root: Path) -> pd.DataFrame:
    """Load all replicate surfaces for one cell as a wide DataFrame
    indexed by subc_id with one column per replicate."""
    cell_dir = surfaces_root / cell.short()
    if not cell_dir.exists():
        raise FileNotFoundError(f"missing replicate-surface dir: {cell_dir}")

    frames = []
    for path in sorted(cell_dir.glob("rep_*.parquet")):
        s = pd.read_parquet(path).set_index("subc_id")["predicted_probability"]
        frames.append(s.rename(path.stem))
    if not frames:
        raise RuntimeError(f"no replicate surfaces in {cell_dir}")
    return pd.concat(frames, axis=1)


def benchmark_for(entity: str, algorithm: str, track: str, paths: GridBPaths) -> pd.Series:
    """Companion paper's deterministic benchmark surface, indexed by subc_id."""
    cell = CellID(entity, algorithm, track, axis="benchmark", level=0)
    df = load_existing_surface(paths, cell, kind="benchmark")
    return df.set_index("subc_id")["predicted_probability"]


def analyse_cell(
    cell: CellID,
    surfaces_root: Path,
    paths: GridBPaths,
) -> CellResult:
    """Compute coverage, width, and diagnostics for one cell."""
    ens = load_ensemble(cell, surfaces_root)
    bench = benchmark_for(cell.entity, cell.algorithm, cell.track, paths)

    # Align on shared subc_ids
    shared = ens.index.intersection(bench.index)
    ens_a = ens.loc[shared]
    bench_a = bench.loc[shared]

    lo = ens_a.quantile(ALPHA / 2, axis=1)
    hi = ens_a.quantile(1 - ALPHA / 2, axis=1)
    width = hi - lo

    inside = (bench_a >= lo) & (bench_a <= hi)
    coverage = float(inside.mean())

    ens_mean = ens_a.mean(axis=1)
    diff = bench_a - ens_mean

    return CellResult(
        entity=cell.entity,
        entity_dir=ENTITY_NAME_TO_DIR[cell.entity],
        algorithm=cell.algorithm,
        track=cell.track,
        level=cell.level,
        n_replicates=ens.shape[1],
        n_pixels=len(shared),
        coverage=coverage,
        median_width=float(width.median()),
        mean_width=float(width.mean()),
        median_signed_diff=float(diff.median()),
        mean_abs_diff=float(diff.abs().mean()),
        frac_diff_gt_0p1=float((diff.abs() > 0.1).mean()),
    )


def analyse_panel(
    surfaces_root: Path,
    paths: GridBPaths,
    entities: tuple[str, ...] = DUAL_AXIS_ENTITIES,
    algorithms: tuple[str, ...] = ("random_forest", "xgboost"),
    tracks: tuple[str, ...] = TRACKS,
    levels: tuple[int, ...] = FULL_PANEL_LOWACC_LEVELS,
    on_error: str = "log",   # "log" | "raise"
) -> pd.DataFrame:
    """Run analyse_cell over the full panel; return a long DataFrame.

    Robust: missing cells are logged (not fatal) so a partial panel still
    produces a usable table. Set on_error='raise' to abort on first error.
    """
    rows: list[dict] = []
    for entity in entities:
        for algorithm in algorithms:
            for track in tracks:
                for level in levels:
                    cell = CellID(entity, algorithm, track, axis="lowacc", level=level)
                    try:
                        result = analyse_cell(cell, surfaces_root, paths)
                        rows.append(asdict(result))
                    except FileNotFoundError as exc:
                        if on_error == "raise":
                            raise
                        log.warning("missing data for %s: %s", cell.short(), exc)
                    except Exception as exc:
                        if on_error == "raise":
                            raise
                        log.exception("error analysing %s", cell.short())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Asymmetry analysis (Lucian's point 2)

def asymmetry_by_benchmark_decile(
    cell: CellID,
    surfaces_root: Path,
    paths: GridBPaths,
    n_bins: int = 10,
) -> pd.DataFrame:
    """For one cell, bin pixels by benchmark suitability into deciles.
    For each bin compute the fraction of contaminated-ensemble means that
    are ABOVE the benchmark prediction (i.e. over-prediction)."""
    ens = load_ensemble(cell, surfaces_root)
    bench = benchmark_for(cell.entity, cell.algorithm, cell.track, paths)
    shared = ens.index.intersection(bench.index)
    ens_mean = ens.loc[shared].mean(axis=1)
    bench_a = bench.loc[shared]

    # Bin by benchmark suitability
    bins = pd.qcut(bench_a, q=n_bins, duplicates="drop")
    over = (ens_mean > bench_a).groupby(bins, observed=True).mean()
    median_diff = (ens_mean - bench_a).groupby(bins, observed=True).median()
    bin_centres = pd.Series(bins.cat.categories.map(lambda iv: iv.mid),
                            index=over.index, name="bench_mid")
    return pd.DataFrame({
        "bench_decile_mid": bin_centres.values,
        "frac_over_predicted": over.values,
        "median_diff": median_diff.values,
        "n_pixels": ens_mean.groupby(bins, observed=True).size().values,
    }).reset_index(drop=True)


def asymmetry_panel(
    surfaces_root: Path,
    paths: GridBPaths,
    level: int = 10,
    n_bins: int = 10,
    on_error: str = "log",
) -> pd.DataFrame:
    """Run asymmetry_by_benchmark_decile across the panel at one fixed level."""
    rows: list[dict] = []
    for entity in DUAL_AXIS_ENTITIES:
        for algorithm in ("random_forest", "xgboost"):
            for track in TRACKS:
                cell = CellID(entity, algorithm, track, axis="lowacc", level=level)
                try:
                    df = asymmetry_by_benchmark_decile(cell, surfaces_root, paths, n_bins=n_bins)
                    df["entity"] = entity
                    df["algorithm"] = algorithm
                    df["track"] = track
                    df["level"] = level
                    rows.append(df)
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    log.warning("asymmetry skip for %s/%s/%s: %s",
                                entity, algorithm, track, exc)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Sample size lookup (for F5 false-alarm panel)

def benchmark_n_per_entity(metrics: pd.DataFrame) -> dict[str, int]:
    """Return benchmark presence count per entity, derived from the merged
    metrics table's `benchmark_presence_n` column."""
    sub = metrics[metrics["axis"] == "benchmark"]
    out: dict[str, int] = {}
    for entity in sub["entity"].unique():
        n = sub.loc[sub["entity"] == entity, "benchmark_presence_n"].dropna()
        if len(n) > 0:
            out[entity] = int(n.iloc[0])
    return out



def asymmetry_panel_dual_resolution(
    surfaces_root: Path,
    paths: GridBPaths,
    level: int = 10,
    on_error: str = "log",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the asymmetry analysis at both 5-bin and 10-bin resolutions.

    Returns ``(headline_5bin, supplementary_10bin)``.

    Per Lucian's review (May 2026): 5-bin is the clean headline figure;
    10-bin retains noise-as-information for the supplementary. Cross-entity
    averaging is deliberately NOT done — the entity-by-entity overlay shows
    replication, which is the methodological point.
    """
    five = asymmetry_panel(surfaces_root, paths, level=level, n_bins=5, on_error=on_error)
    ten = asymmetry_panel(surfaces_root, paths, level=level, n_bins=10, on_error=on_error)
    return five, ten
