"""Command-line entry points for trustworthy-sdm.

Exposes:

* ``trustworthy-sdm-inspect`` — sanity-check imports and configuration.
* ``trustworthy-sdm-regenerate`` — regenerate per-replicate surfaces.

Both wrap ``argparse`` and exit with non-zero status on error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import trustworthy_sdm
from trustworthy_sdm.io import (
    DUAL_AXIS_ENTITIES,
    GridBPaths,
    iter_pilot_cells,
    load_merged_metrics,
    replicate_seeds,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# inspect

def inspect_main(argv: list[str] | None = None) -> int:
    """Verify imports, paths, and replicate availability for the pilot cells."""
    parser = argparse.ArgumentParser(
        prog="trustworthy-sdm-inspect",
        description="Verify environment and Grid B layout for the pilot run.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path to the results root that contains grid_b_full/, grid_b_merged/, etc.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("inspect")

    log.info("trustworthy_sdm version: %s", trustworthy_sdm.__version__)

    # Try the upstream import. Inspect is allowed to succeed without it
    # (so you can run it on a freshly-cloned Mac that hasn't installed
    # sdm-robustness yet) — but we report what we find.
    try:
        from sdm_robustness.pipeline.core import fit_cv_cell  # type: ignore[import-not-found]

        log.info("sdm_robustness.fit_cv_cell: importable (%s)", fit_cv_cell.__module__)
    except ImportError as exc:
        log.warning("sdm_robustness not importable: %s", exc)
        log.warning("install with:  pip install -e <path-to-sdm-robustness>")

    # Grid B paths
    paths = GridBPaths(root=args.results_root.expanduser().resolve())
    log.info("grid_b_full: %s", paths.grid_b_full)
    log.info("grid_b_merged: %s (exists=%s)",
             paths.grid_b_merged, paths.grid_b_merged.exists())
    log.info("benchmark_stability: %s", paths.benchmark_stability)

    if not paths.grid_b_merged.exists():
        log.error("Cannot find merged metrics file. Aborting.")
        return 2

    metrics = load_merged_metrics(paths)
    log.info("merged metrics: shape=%s", metrics.shape)
    log.info("entities present: %d", metrics["entity"].nunique())

    # For each pilot cell, can we find the 30 seeds?
    log.info("checking pilot cells:")
    all_ok = True
    for cell in iter_pilot_cells():
        try:
            seeds = replicate_seeds(metrics, cell)
            log.info("  %s: %d replicates", cell.short(), len(seeds))
            if len(seeds) != 30:
                log.warning("    expected 30 replicates, got %d", len(seeds))
                all_ok = False
        except KeyError as exc:
            log.error("  %s: %s", cell.short(), exc)
            all_ok = False

    log.info("dual-axis entities defined in trustworthy_sdm.io: %d", len(DUAL_AXIS_ENTITIES))

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# regenerate

def regenerate_main(argv: list[str] | None = None) -> int:
    """Regenerate per-replicate surfaces for the pilot cells (or a custom selection)."""
    parser = argparse.ArgumentParser(
        prog="trustworthy-sdm-regenerate",
        description="Regenerate per-replicate suitability surfaces by replaying fit_cv_cell.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path containing grid_b_full/, grid_b_merged/, etc.",
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        required=True,
        help="Path to combined_data_true_master.csv.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root for replicate surfaces.",
    )
    parser.add_argument(
        "--cells",
        choices=["pilot"],
        default="pilot",
        help="Which set of cells to regenerate. Currently only 'pilot' is wired.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be done without calling fit_cv_cell.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute even if output parquet already exists.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("regenerate")

    # Late import so that --help works even if sdm_robustness is not installed.
    from trustworthy_sdm.regen import regenerate_cells

    paths = GridBPaths(root=args.results_root.expanduser().resolve())
    cells = list(iter_pilot_cells()) if args.cells == "pilot" else []
    log.info("regenerating %d cells", len(cells))

    summary = regenerate_cells(
        cells=cells,
        paths=paths,
        master_csv=args.master_csv.expanduser().resolve(),
        out_root=args.out_root.expanduser().resolve(),
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )

    # Write the run summary to disk so we have a record of what happened.
    summary_path = args.out_root / "regeneration_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    log.info("wrote summary to %s", summary_path)

    n_ok = int((summary["status"] == "ok").sum()) if len(summary) else 0
    n_err = int(summary["status"].astype(str).str.startswith("error").sum()) if len(summary) else 0
    log.info("done: %d ok, %d errors, %d total", n_ok, n_err, len(summary))
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(regenerate_main())
