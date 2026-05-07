"""Pilot driver: run Symba on Austropotamobius torrentium (pooled), L0 + L20.

This script exists because the upstream sdm-robustness merged metrics CSV
contains only RF/XGBoost/Maxent rows — there are no logged Symba seeds to
look up. So we mint deterministic seeds locally and call fit_cv_cell
directly, producing per-replicate Symba surfaces in the same parquet
format that regen.py produces for RF/XGBoost.

Output layout matches regen.py:
    {out_root}/{cell.short()}/rep_{NN:02d}.parquet
where cell.short() is e.g.
    Austropotamobius_torrentium_pooled__symba__combined__benchmark__L0
    Austropotamobius_torrentium_pooled__symba__combined__lowacc__L20

After this completes, point your existing analysis tools at the new cell
directories alongside the RF/XGBoost ones for direct comparison.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_sdm.io import CellID
from trustworthy_sdm.regen import (
    RegenerationInputs,
    assemble_inputs,
    replicate_surface_path,
)

log = logging.getLogger("symba-pilot")

# -------------------------------------------------------------------------
# Pilot configuration — locked here, not in YAML, to keep the script
# self-contained and obvious.
# -------------------------------------------------------------------------
ENTITY = "Austropotamobius torrentium (pooled)"
TRACK = "combined"
ALGORITHM = "symba"
N_REPLICATES = 30
MASTER_SEED = 20260507  # today's date, distinct from the 20260416 used upstream

# Two cells: the L0 benchmark and the L20 lowacc stress test.
CELLS = [
    {"axis": "benchmark", "level": 0},
    {"axis": "lowacc",    "level": 20},
]


def mint_seeds(master_seed: int, cell_short: str, n: int) -> pd.DataFrame:
    """Generate n deterministic seeds from (master_seed, cell_short).

    Different cells get different seeds, runs are reproducible across
    invocations, and the (replicate, seed) mapping is logged.
    """
    # Use cell_short as a salt: SHA-256 -> int -> RNG. Stable across machines.
    import hashlib
    salt = int(hashlib.sha256(cell_short.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(master_seed ^ salt)
    seeds = rng.integers(low=1, high=2**31 - 1, size=n, dtype=np.int64)
    return pd.DataFrame({"replicate": np.arange(n, dtype=int), "seed": seeds})


def run_cell(
    cell: CellID,
    inputs: RegenerationInputs,
    seeds_df: pd.DataFrame,
    out_root: Path,
    *,
    skip_existing: bool = True,
) -> list[dict]:
    """Run all replicates for one cell. Mirrors regen.regenerate_cell."""
    from sdm_robustness.pipeline.core import fit_cv_cell  # type: ignore[import-not-found]

    cell_out = out_root / cell.short()
    cell_out.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for _, row in seeds_df.iterrows():
        rep = int(row["replicate"])
        seed = int(row["seed"])
        out_path = replicate_surface_path(out_root, cell, rep)
        base = {
            "cell": cell.short(),
            "replicate": rep,
            "seed": seed,
            "output_path": str(out_path),
        }

        if skip_existing and out_path.exists():
            summaries.append({**base, "status": "skipped_existing", "wall_seconds": 0.0})
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
        except Exception as exc:  # noqa: BLE001
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
            summaries.append({**base, "status": "empty_surface",
                              "wall_seconds": time.time() - t0})
            continue

        if len(surface) != len(inputs.accessible_area):
            log.error(
                "surface length mismatch for %s rep=%d: surface=%d, acc=%d",
                cell.short(), rep, len(surface), len(inputs.accessible_area),
            )
            summaries.append({**base, "status": "length_mismatch",
                              "wall_seconds": time.time() - t0})
            continue

        df = pd.DataFrame({
            "subc_id": inputs.accessible_area["subc_id"].values,
            "predicted_probability": np.asarray(surface, dtype=float),
        })
        df.to_parquet(out_path, index=False)
        summaries.append({**base, "status": "ok", "wall_seconds": time.time() - t0})
        log.info("  rep=%02d seed=%d: %.1fs", rep, seed, time.time() - t0)

    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Symba pilot runner.")
    parser.add_argument(
        "--master-csv", type=Path, required=True,
        help="Path to combined_data_true_master.csv",
    )
    parser.add_argument(
        "--out-root", type=Path, required=True,
        help="Output root for replicate surfaces (e.g. data/replicate_surfaces).",
    )
    parser.add_argument(
        "--n-replicates", type=int, default=N_REPLICATES,
        help=f"Number of replicates per cell (default {N_REPLICATES}).",
    )
    parser.add_argument(
        "--cells", choices=["all", "benchmark", "lowacc"], default="all",
        help="Which cell(s) to run (default all).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-run replicates even if output parquet already exists.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_root = args.out_root.expanduser().resolve()
    master_csv = args.master_csv.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cells_to_run = CELLS
    if args.cells == "benchmark":
        cells_to_run = [c for c in CELLS if c["axis"] == "benchmark"]
    elif args.cells == "lowacc":
        cells_to_run = [c for c in CELLS if c["axis"] == "lowacc"]

    log.info("entity:        %s", ENTITY)
    log.info("algorithm:     %s", ALGORITHM)
    log.info("track:         %s", TRACK)
    log.info("master_csv:    %s", master_csv)
    log.info("out_root:      %s", out_root)
    log.info("n_replicates:  %d", args.n_replicates)
    log.info("cells:         %s", [(c["axis"], c["level"]) for c in cells_to_run])

    all_summaries: list[dict] = []
    for cell_spec in cells_to_run:
        cell = CellID(
            entity=ENTITY,
            algorithm=ALGORITHM,
            track=TRACK,
            axis=cell_spec["axis"],
            level=cell_spec["level"],
        )
        log.info("=" * 70)
        log.info("CELL: %s", cell.short())

        # Inputs are axis-dependent: load once per (entity, axis).
        log.info("loading inputs ...")
        inputs = assemble_inputs(
            entity=ENTITY,
            track=TRACK,
            axis=cell_spec["axis"],
            master_csv=master_csv,
        )

        seeds_df = mint_seeds(MASTER_SEED, cell.short(), args.n_replicates)
        seeds_path = out_root / cell.short() / "seeds.csv"
        seeds_path.parent.mkdir(parents=True, exist_ok=True)
        seeds_df.to_csv(seeds_path, index=False)
        log.info("seeds logged to %s", seeds_path)

        summaries = run_cell(
            cell=cell,
            inputs=inputs,
            seeds_df=seeds_df,
            out_root=out_root,
            skip_existing=not args.no_skip_existing,
        )
        all_summaries.extend(summaries)

        ok = sum(1 for s in summaries if s["status"] == "ok")
        skipped = sum(1 for s in summaries if s["status"] == "skipped_existing")
        errors = [s for s in summaries if s["status"].startswith("error")]
        log.info("cell done: ok=%d skipped=%d errors=%d", ok, skipped, len(errors))
        for e in errors[:3]:
            log.error("  rep=%d: %s", e["replicate"], e["status"])

    summary_path = out_root / "symba_pilot_run_summary.csv"
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    log.info("=" * 70)
    log.info("run summary: %s", summary_path)

    n_errors = sum(1 for s in all_summaries if s["status"].startswith("error"))
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
