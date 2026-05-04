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

# Lazy import: we import inside the function so that ``trustworthy-sdm-inspect``
# can run on a machine that does not have sdm_robustness installed.
# (The actual regeneration commands obviously do require it.)
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

    Loading these once per (entity, track) avoids repeating I/O for every replicate.
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
# Inputs assembly
#
# IMPLEMENTATION PENDING. We need to find the function in the upstream repo
# (or one of its scripts) that splits the master CSV into (benchmark,
# contamination pool, accessible area) for a given entity/axis. The
# orchestrator-level code lives outside ``sdm_robustness/pipeline/``. Once
# located, ``assemble_inputs`` becomes a thin wrapper around it.
#
# See ``scripts/inspect_orchestrator.sh`` for the probe script.

def assemble_inputs(
    entity: str,
    track: str,
    axis: str,
    master_csv: Path,
) -> RegenerationInputs:
    """Build the three DataFrames that ``fit_cv_cell`` consumes for one entity.

    .. warning::
        Implementation pending. Run ``scripts/inspect_orchestrator.sh`` on
        VEGA to locate the upstream entity-data-prep function, then wire
        this function to call it.
    """
    raise NotImplementedError(
        "assemble_inputs is pending: needs upstream orchestrator inspection. "
        "Run scripts/inspect_orchestrator.sh on VEGA, paste output to planning thread."
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
    """Regenerate the 30 replicate surfaces for one cell.

    Parameters
    ----------
    cell : CellID
        Cell descriptor.
    inputs : RegenerationInputs
        Pre-loaded benchmark / pool / accessible_area DataFrames.
    seeds_df : DataFrame
        From ``replicate_seeds(metrics, cell)``. Columns ``replicate``, ``seed``.
    out_root : Path
        Root for replicate surface output. Surfaces go to
        ``{out_root}/{cell.short()}/rep_NN.parquet``.
    skip_existing : bool
        If True, replicates whose output already exists are not recomputed.
    dry_run : bool
        If True, only logs what would be done; does not call ``fit_cv_cell``.

    Returns
    -------
    list[dict]
        One summary dict per replicate, with keys
        ``cell``, ``replicate``, ``seed``, ``status``, ``wall_seconds``, ``output_path``.
    """
    # Lazy-import the upstream dependency so this module can be imported in
    # contexts (CI, doctest, ``trustworthy-sdm-inspect``) where sdm_robustness
    # may not be installed.
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

        # Benchmark cells use level=0 and an axis label that fit_cv_cell
        # recognises. The axis only affects which contamination pool is
        # consulted; at level=0 contamination is the empty set so the choice
        # is mathematically irrelevant. We canonicalise on "snapping".
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
        except Exception as exc:  # noqa: BLE001 - we log the full traceback
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

        # The surface is aligned to inputs.accessible_area's row order. The
        # canonical sub-catchment id column is 'subc_id'. Persist as parquet
        # matching the schema of the existing single-prediction surfaces.
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
    """Regenerate replicate surfaces for many cells, with input caching by entity.

    Inputs only depend on ``(entity, track, axis)``, so we cache them across
    (algorithm, level) combinations within the same entity.
    """
    metrics = load_merged_metrics(paths)
    summaries: list[dict] = []
    inputs_cache: dict[tuple[str, str, str], RegenerationInputs] = {}

    for cell in cells:
        # The contamination pool depends on the axis; at benchmark (axis=benchmark
        # or level=0) it is unused, but we still pass a pool for type compatibility.
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
