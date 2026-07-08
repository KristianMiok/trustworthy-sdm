#!/usr/bin/env python
"""
run_overprediction.py -- point-level directional over-prediction of P. leniusculus.

Tests Option 3: the directional applied story lives in the POINT prediction,
independent of the symmetric conformal correction. Per contamination level it
reports whether training on contaminated data over-predicts suitability, and
whether that over-prediction concentrates in low-benchmark-suitability habitat
(the periphery), where surveillance budgets are most easily misdirected.

Reports per level (RF / combined, L3/L10/L20):
  * signed divergence  div = ensemble_mean - benchmark   (>0 = over-prediction)
      overall, and by fixed benchmark-suitability bands
  * point-level decision flips at tau (NO intervals):
      false_suitable   : p > tau  and  p_bench <= tau   (over-prediction into unsuitable)
      false_unsuitable : p <= tau and  p_bench >  tau   (under-prediction)
    counts, domain fraction, net directional asymmetry, and where the
    false-suitable cells sit on the benchmark scale.

Honesty (M2): benchmark is the clean-data MODEL, not ground truth. Read
"false_suitable" as "contaminated model calls it suitable where the clean model
would not", not "wrong vs reality".

Run from repo root:  python run_overprediction.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# src-layout importable from a bare terminal (PyCharm adds src as a source root)
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

# dodge the pyarrow "Repetition level histogram size mismatch" on the surfaces
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
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

ENTITY = "Pacifastacus leniusculus (alien)"
ALGO = "random_forest"
TRACK = "combined"
LEVELS = (3, 10, 20)
TAU = 0.5
TAU_SWEEP = (0.4, 0.5, 0.6)
BANDS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0001]   # benchmark-suitability bands


def load_point(cell: CellID) -> pd.DataFrame:
    """Ensemble mean (point suitability) and benchmark, aligned on subc_ids."""
    ens = load_ensemble(cell, SURFACES_ROOT)
    bench = benchmark_for(cell.entity, cell.algorithm, cell.track, PATHS)
    shared = ens.index.intersection(bench.index)
    return pd.DataFrame({
        "cell_id": np.asarray(shared),
        "p": ens.loc[shared].mean(axis=1).to_numpy(),
        "p_bench": bench.loc[shared].to_numpy(),
    })


def analyse_level(level: int, tau: float):
    cell = CellID(ENTITY, ALGO, TRACK, axis="lowacc", level=level)
    df = load_point(cell)
    df["div"] = df["p"] - df["p_bench"]   # >0 = over-prediction

    overall = {
        "n_cells": int(len(df)),
        "median_div": round(float(df["div"].median()), 4),
        "mean_div": round(float(df["div"].mean()), 4),
        "frac_overpred_gt_+0.1": round(float((df["div"] > 0.1).mean()), 4),
        "frac_underpred_lt_-0.1": round(float((df["div"] < -0.1).mean()), 4),
    }

    # signed divergence by fixed benchmark-suitability band
    band = pd.cut(df["p_bench"], BANDS, right=False, include_lowest=True)
    bt = df.groupby(band, observed=False)["div"].agg(["median", "mean", "size"])
    band_tbl = {
        str(idx): {"median_div": round(float(r["median"]), 4),
                   "mean_div": round(float(r["mean"]), 4),
                   "n": int(r["size"])}
        for idx, r in bt.iterrows()
    }

    fs = (df["p"] > tau) & (df["p_bench"] <= tau)    # false-suitable (over-pred)
    fu = (df["p"] <= tau) & (df["p_bench"] > tau)    # false-unsuitable (under-pred)
    flips = {
        "n_false_suitable": int(fs.sum()),
        "frac_false_suitable": round(float(fs.mean()), 5),
        "n_false_unsuitable": int(fu.sum()),
        "frac_false_unsuitable": round(float(fu.mean()), 5),
        "net_overpred_cells": int(fs.sum() - fu.sum()),
        "median_pbench_at_false_suitable": (round(float(df.loc[fs, "p_bench"].median()), 4)
                                            if fs.any() else float("nan")),
        "median_p_at_false_suitable": (round(float(df.loc[fs, "p"].median()), 4)
                                       if fs.any() else float("nan")),
    }
    df["false_suitable"] = fs
    df["false_unsuitable"] = fu
    return df, overall, band_tbl, flips


def main() -> None:
    report: dict = {}
    for level in LEVELS:
        df, overall, band_tbl, flips = analyse_level(level, TAU)
        df.to_csv(OUT / f"overpred_pleniusculus_L{level}.csv", index=False)
        report[f"L{level}"] = {"overall": overall,
                               "by_benchmark_band": band_tbl,
                               "point_flips_tau0.5": flips}

    tau_tbl = {}
    for tau in TAU_SWEEP:
        _, _, _, flips = analyse_level(10, tau)
        tau_tbl[f"tau={tau}"] = {"n_false_suitable": flips["n_false_suitable"],
                                 "n_false_unsuitable": flips["n_false_unsuitable"],
                                 "net_overpred_cells": flips["net_overpred_cells"]}
    report["tau_sensitivity_L10"] = tau_tbl

    print("=" * 70)
    pprint(report, sort_dicts=False)
    print("=" * 70)
    print(f"\nPer-pixel CSVs: {OUT}/overpred_pleniusculus_L*.csv")
    print("Read: does over-prediction (div>0) grow in the LOW benchmark bands")
    print("([0,0.1),[0.1,0.3)) as contamination rises? That is the periphery signal.")


if __name__ == "__main__":
    main()
