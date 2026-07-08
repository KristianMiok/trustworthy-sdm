#!/usr/bin/env python
"""
run_overpred_ci.py -- threshold-FREE hardened over-prediction signal.

Backbone finding for the JAE paper, with NO tau anywhere: at each contamination
level, how much does the contaminated ensemble over-predict suitability
(prediction - benchmark) within fixed benchmark-suitability bands, and how
robust is that to the contamination draw?

Uncertainty source = the 30 replicate surfaces (different low-accuracy
contamination draws at the same level). Band membership is defined by the
benchmark (fixed across replicates), so only the prediction varies. For each
band we report, across the 30 replicates, the cell-mean and cell-median
over-prediction with a 2.5-97.5 percentile interval.

The finding is robust if the edge-band interval at L20 sits clearly above the
interval at L3 (growth is not contamination-draw noise).

Honesty (M2): benchmark is the clean-data MODEL, not truth. Over-prediction
means "vs what clean data would predict", not "vs reality".

Run from repo root:  python run_overpred_ci.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

try:
    import fastparquet  # noqa: F401
    pd.options.io.parquet.engine = "fastparquet"
except Exception:  # noqa: BLE001
    pass

from trustworthy_sdm.io import CellID, GridBPaths
from trustworthy_sdm.analysis import load_ensemble, benchmark_for

# --- config -----------------------------------------------------------------
SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
PATHS = GridBPaths(root=Path("data/results").resolve())

ENTITY = "Pacifastacus leniusculus (alien)"
ALGO = "random_forest"
TRACK = "combined"
LEVELS = (3, 10, 20)
BANDS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0001]
EDGE_BAND = "[0.1, 0.3)"   # the lower edge of suitability -- headline band


def _ci(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {
        "mean": round(float(a.mean()), 4),
        "lo2.5": round(float(np.percentile(a, 2.5)), 4),
        "hi97.5": round(float(np.percentile(a, 97.5)), 4),
    }


def analyse_level_ci(level: int) -> dict:
    cell = CellID(ENTITY, ALGO, TRACK, axis="lowacc", level=level)
    ens = load_ensemble(cell, SURFACES_ROOT)               # subc_id x 30 reps
    bench = benchmark_for(cell.entity, cell.algorithm, cell.track, PATHS)
    shared = ens.index.intersection(bench.index)
    ens = ens.loc[shared]
    bench = bench.loc[shared]

    band = pd.cut(bench, BANDS, right=False, include_lowest=True)
    cats = list(band.cat.categories)

    rep_mean = {str(b): [] for b in cats}
    rep_med = {str(b): [] for b in cats}
    for col in ens.columns:                                # one replicate surface
        div = ens[col] - bench                             # >0 = over-prediction
        gm = div.groupby(band, observed=False).mean()
        gd = div.groupby(band, observed=False).median()
        for b in cats:
            rep_mean[str(b)].append(float(gm.loc[b]))
            rep_med[str(b)].append(float(gd.loc[b]))

    out = {}
    for b in cats:
        out[str(b)] = {
            "n_cells": int((band == b).sum()),
            "cellmean_div": _ci(rep_mean[str(b)]),          # across 30 reps
            "cellmedian_div": _ci(rep_med[str(b)]),
        }
    return out


def main() -> None:
    report = {}
    for level in LEVELS:
        report[f"L{level}"] = analyse_level_ci(level)

    print("=" * 70)
    pprint(report, sort_dicts=False)
    print("=" * 70)

    # headline: edge band, dose-response with replicate intervals
    print(f"\nHEADLINE -- over-prediction in the edge band {EDGE_BAND}")
    print("cell-mean (prediction - benchmark), mean [2.5-97.5% across 30 replicates]:")
    prev_hi = None
    for level in LEVELS:
        c = report[f"L{level}"][EDGE_BAND]["cellmean_div"]
        sep = ""
        if prev_hi is not None:
            sep = "  <- interval clears previous level" if c["lo2.5"] > prev_hi else "  <- OVERLAPS previous level"
        print(f"  L{level:>2}: {c['mean']:.4f}  [{c['lo2.5']:.4f}, {c['hi97.5']:.4f}]{sep}")
        prev_hi = c["hi97.5"]
    print("\nIf each level's interval clears the previous, the dose-response is")
    print("robust to the contamination draw and holds with no threshold at all.")


if __name__ == "__main__":
    main()
