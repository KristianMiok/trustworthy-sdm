"""Regenerate per-replicate suitability surfaces by replaying ``fit_cv_cell``.

The companion paper's pipeline saved a single-prediction surface per cell,
not per replicate. Pixel-wise ensemble uncertainty therefore requires
re-running the original ``fit_cv_cell`` with each of the 30 logged seeds and
capturing the surface from each call.

This module is the only place we depend on the upstream ``sdm_robustness``
package. Keep that boundary thin.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_sdm.io import (
    CellID,
    GridBPaths,
    load_merged_metrics,
    replicate_seeds,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs the upstream fit_cv_cell needs

@dataclass
class RegenerationInputs:
    """Everything ``fit_cv_cell`` needs that is NOT derived from (cell, replicate, seed).

    Loading these once per (entity, axis) avoids repeating I/O for every replicate.
    The contamination pool depends on axis: snap_pool for snapping, lowacc_pool
    for lowacc. Benchmark cells (axis="benchmark") are run with snap_pool but
    level=0, so the pool is mathematically inert.
    """

    benchmark: pd.DataFrame
    contamination_pool: pd.DataFrame
    accessible_area: pd.DataFrame
    n_experiment: int


def replicate_surface_path(
    out_root: Path,
    cell: CellID,
    replicate: int,
) -> Path:
    """``{out_root}/{cell.short()}/rep_{NN:02d}.parquet``."""
    return out_root / cell.short() / f"rep_{replicate:02d}.parquet"


# ---------------------------------------------------------------------------
# Master-table cache (load once per process)

_MASTER_CACHE: dict[Path, pd.DataFrame] = {}


def _load_master_cached(master_csv: Path) -> pd.DataFrame:
    """Load the 115k-row master table once per process."""
    p = master_csv.resolve()
    if p not in _MASTER_CACHE:
        from sdm_robustness.io import load_master_table  # type: ignore[import-not-found]

        log.info("loading master table from %s", p)
        info_or_df = load_master_table(p)
        # load_master_table may return either a DataFrame or a (DataFrame, info)
        # tuple depending on the upstream version. Handle both defensively.
        if isinstance(info_or_df, tuple):
            df, _info = info_or_df
        else:
            df = info_or_df
        log.info("master table: %d rows, %d cols", len(df), df.shape[1])
        _MASTER_CACHE[p] = df
    return _MASTER_CACHE[p]


# ---------------------------------------------------------------------------
# Inputs assembly — wired to upstream _prepare_entity_data

def assemble_inputs(
    entity: str,
    track: str,                # noqa: ARG001  (kept for API symmetry; unused)
    axis: str,
    master_csv: Path,
) -> RegenerationInputs:
    """Build the three DataFrames that ``fit_cv_cell`` consumes for one entity/axis.

    Wraps upstream ``_prepare_entity_data``. For Grid B, ``n_experiment`` is the
    full benchmark size (no subsampling); upstream's runner sets this with the
    comment "Grid B: full benchmark, no cap" at runner.py:308.

    Parameters
    ----------
    entity : str
        Canonical entity name, e.g. "Austropotamobius torrentium (pooled)".
    track : str
        Spatial track. Currently unused inside ``fit_cv_cell``'s data-prep step
        (track-specific column selection happens inside the function), but kept
        in the signature for clarity at call sites.
    axis : str
        ``"snapping"`` or ``"lowacc"``. Determines which contamination pool is
        returned. ``"benchmark"`` is normalised to ``"snapping"`` upstream.
    master_csv : Path
        Path to ``combined_data_true_master.csv``.
    """
    # Upstream is private API (`_prepare_entity_data`) — we accept the
    # stability tradeoff because it's far cleaner than reimplementing
    # entity-data prep ourselves and risking a silent divergence from the
    # companion paper's behaviour.
    from sdm_robustness.execution.runner import _prepare_entity_data  # type: ignore[import-not-found]

    if axis not in {"snapping", "lowacc", "benchmark"}:
        raise ValueError(f"axis must be 'snapping', 'lowacc', or 'benchmark'; got {axis!r}")

    master = _load_master_cached(master_csv)
    prepared = _prepare_entity_data(master, entity)

    # Pick the contamination pool that matches this axis. Benchmark cells use
    # snap_pool by convention (it's mathematically inert at level=0).
    pool_key = "snap_pool" if axis in ("snapping", "benchmark") else "lowacc_pool"
    contamination_pool: pd.DataFrame = prepared[pool_key]
    benchmark: pd.DataFrame = prepared["benchmark"]
    accessible_area: pd.DataFrame = prepared["accessible_area"]

    log.info(
        "%s [%s]: benchmark_n=%d, %s_pool_n=%d, accessible_area_n=%d",
        entity, axis, len(benchmark), pool_key, len(contamination_pool), len(accessible_area),
    )

    # Grid B convention: use the full benchmark, no cap.
    n_experiment = len(benchmark)

    return RegenerationInputs(
        benchmark=benchmark,
        contamination_pool=contamination_pool,
        accessible_area=accessible_area,
        n_experiment=n_experiment,
    )


# ---------------------------------------------------------------------------
# Single-cell regeneration

def regenerate_cell(
    cell: CellID,
    inputs: RegenerationInputs,
    seeds_df: pd.DataFrame,
    out_root: Path,
    *,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """Regenerate the 30 replicate surfaces for one cell."""
    from sdm_robustness.pipeline.core import fit_cv_cell  # type: ignore[import-not-found]

    cell_out = out_root / cell.short()
    cell_out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for _, row in seeds_df.iterrows():
        rep = int(row["replicate"])
        seed = int(row["seed"])
        out_path = replicate_surface_path(out_root, cell, rep)

        base = {"cell": cell.short(), "replicate": rep, "seed": seed,
                "output_path": str(out_path)}

        if skip_existing and out_path.exists():
            summaries.append({**base, "status": "skipped_existing", "wall_seconds": 0.0})
            continue

        if dry_run:
            log.info("[dry-run] would regenerate %s rep=%d seed=%d", cell.short(), rep, seed)
            summaries.append({**base, "status": "dry_run", "wall_seconds": 0.0})
            continue

        upstream_axis = cell.axis if cell.axis != "benchmark" else "snapping"

        t0 = time.time()
        try:
            result = fit_cv_cell(
                benchmark=inputs.benchmark,
                contamination_pool=inputs.contamination_pool,
                accessible_area=inputs.accessible_area,
                entity=cell.entity,
                algorithm=cell.algorithm,
                track=cell.track,
                axis=upstream_axis,
                level=cell.level,
                replicate=rep,
                seed=seed,
                n_experiment=inputs.n_experiment,
                return_artifacts=True,
            )
        except Exception as exc:  # noqa: BLE001 - we log full traceback below
            log.exception("fit_cv_cell failed for %s rep=%d", cell.short(), rep)
            summaries.append({
                **base,
                "status": f"error: {type(exc).__name__}: {exc}",
                "wall_seconds": time.time() - t0,
            })
            continue

        surface = result.get("_run_surface")
        if surface is None or len(surface) == 0:
            log.warning("empty surface for %s rep=%d", cell.short(), rep)
            summaries.append({
                **base, "status": "empty_surface", "wall_seconds": time.time() - t0,
            })
            continue

        if len(surface) != len(inputs.accessible_area):
            log.error(
                "surface length mismatch for %s rep=%d: surface=%d, acc=%d",
                cell.short(), rep, len(surface), len(inputs.accessible_area),
            )
            summaries.append({
                **base, "status": "length_mismatch", "wall_seconds": time.time() - t0,
            })
            continue

        df = pd.DataFrame({
            "subc_id": inputs.accessible_area["subc_id"].values,
            "predicted_probability": np.asarray(surface, dtype=float),
        })
        df.to_parquet(out_path, index=False)

        summaries.append({**base, "status": "ok", "wall_seconds": time.time() - t0})

    return summaries


# ---------------------------------------------------------------------------
# Multi-cell driver

def regenerate_cells(
    cells: Iterable[CellID],
    paths: GridBPaths,
    master_csv: Path,
    out_root: Path,
    *,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Regenerate replicate surfaces for many cells, with input caching by entity+axis."""
    metrics = load_merged_metrics(paths)
    summaries: list[dict] = []
    inputs_cache: dict[tuple[str, str, str], RegenerationInputs] = {}

    for cell in cells:
        cache_axis = "snapping" if cell.axis == "benchmark" else cell.axis
        key = (cell.entity, cell.track, cache_axis)
        if key not in inputs_cache:
            log.info("loading inputs for entity=%s track=%s axis=%s", *key)
            inputs_cache[key] = assemble_inputs(
                entity=cell.entity, track=cell.track, axis=cache_axis,
                master_csv=master_csv,
            )

        seeds_df = replicate_seeds(metrics, cell)
        log.info("regenerating %s (%d replicates)", cell.short(), len(seeds_df))

        cell_summaries = regenerate_cell(
            cell=cell,
            inputs=inputs_cache[key],
            seeds_df=seeds_df,
            out_root=out_root,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
        summaries.extend(cell_summaries)

    return pd.DataFrame(summaries)
