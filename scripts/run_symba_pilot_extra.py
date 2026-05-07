"""Extra Symba cells for A. torrentium combined: L3 and L10.

Run alongside the original run_symba_pilot.py to fill in the
intermediate contamination levels for the calibration comparison.
Same seed-minting strategy and output format.
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_symba_pilot import mint_seeds, run_cell  # reuse helpers

log = logging.getLogger("symba-pilot-extra")

ENTITY = "Austropotamobius torrentium (pooled)"
TRACK = "combined"
ALGORITHM = "symba"
N_REPLICATES = 30
MASTER_SEED = 20260507

# The two contamination levels we still need.
CELLS = [
    {"axis": "lowacc", "level": 3},
    {"axis": "lowacc", "level": 10},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_root = args.out_root.expanduser().resolve()
    master_csv = args.master_csv.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cells_to_run = CELLS
    array_filter = None
    if args.array_index is not None:
        n_per_cell = N_REPLICATES
        total_units = len(cells_to_run) * n_per_cell
        if not (0 <= args.array_index < total_units):
            log.error("array-index %d out of range [0, %d)", args.array_index, total_units)
            return 2
        cell_idx, rep_idx = divmod(args.array_index, n_per_cell)
        only = cells_to_run[cell_idx]
        cells_to_run = [only]
        array_filter = {(only["axis"], only["level"]): {rep_idx}}
        log.info("array-index %d -> axis=%s level=%d replicate=%d",
                 args.array_index, only["axis"], only["level"], rep_idx)

    log.info("entity:        %s", ENTITY)
    log.info("algorithm:     %s", ALGORITHM)
    log.info("track:         %s", TRACK)
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
        log.info("loading inputs ...")
        inputs = assemble_inputs(
            entity=ENTITY,
            track=TRACK,
            axis=cell_spec["axis"],
            master_csv=master_csv,
        )
        seeds_df = mint_seeds(MASTER_SEED, cell.short(), N_REPLICATES)
        seeds_path = out_root / cell.short() / "seeds.csv"
        seeds_path.parent.mkdir(parents=True, exist_ok=True)
        seeds_df.to_csv(seeds_path, index=False)

        replicates_filter = None
        if array_filter is not None:
            replicates_filter = array_filter.get((cell_spec["axis"], cell_spec["level"]))
        summaries = run_cell(
            cell=cell,
            inputs=inputs,
            seeds_df=seeds_df,
            out_root=out_root,
            skip_existing=True,
            replicates_filter=replicates_filter,
        )
        all_summaries.extend(summaries)
        ok = sum(1 for s in summaries if s["status"] == "ok")
        log.info("cell done: ok=%d", ok)

    return 0


if __name__ == "__main__":
    sys.exit(main())
