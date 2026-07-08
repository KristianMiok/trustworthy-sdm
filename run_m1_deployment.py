#!/usr/bin/env python
"""
run_m1_deployment.py -- M1 deployment-realism validation for the JAE worked case.

The worked-case correction calibrates against the companion paper's benchmark
surface. A practitioner in the field does not have that file. M1 shows they can
RECONSTRUCT an equivalent reference from their OWN high-accuracy records (the same
accuracy flag), using a standard ensemble SDM, and that the conformal correction
restores coverage against that reconstructed reference just as it does against the
idealised benchmark.

Design (option A -- combined-track domain where the predictor set is defined):
  1. Load the 12 combined predictors for the P. leniusculus grid from the master.
     Three UPSTREAM predictors are structurally NaN on headwater cells (no upstream
     catchment), so the combined-track reference is defined on the ~64.5% of grid
     cells with complete env. Headwater cells are an explicit out-of-scope limit
     (the companion Paper 9 topic), reported honestly.
  2. Reconstruct the reference: a 10-member RandomForest ensemble (mean) trained on
     HIGH-accuracy presences + background, on the same predictors as the benchmark.
     Realistic (ensemble SDM is standard) but stabilised, so the comparison isolates
     "does the high-accuracy subset reconstruct an adequate reference", not RF noise.
  3. Correlate the reconstruction with the companion benchmark. High correlation is
     EXPECTED and is the point: the reference shares presence data with the
     benchmark (exactly the practitioner's situation), so the claim is
     REPRODUCIBILITY of the reference, not statistical independence.
  4. Run LOBO conformal (the project's own conformal.py) on the contaminated
     ensemble, once against the benchmark and once against the reconstructed
     reference, and compare restored coverage across L3/L10/L20.

M1 is supported if reconstructed-reference coverage tracks benchmark coverage.

Run from the trustworthy-sdm repo root:  python run_m1_deployment.py
"""
from __future__ import annotations

import json
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

from sklearn.ensemble import RandomForestClassifier

from trustworthy_sdm.io import CellID, GridBPaths
from trustworthy_sdm.analysis import ALPHA, load_ensemble, benchmark_for
from trustworthy_sdm.conformal import nonconformity_scores, conformal_quantile

MASTER = Path("/Users/kristianmiok/Desktop/Papers/sdm-robustness/data/raw/"
              "combined_data_true_master.csv")
PRED_JSON = Path("data/protocol_b_predictor_sets.json")
BASIN_LOOKUP = Path("data/results/basin_lookup.parquet")
SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
PATHS = GridBPaths(root=Path("data/results").resolve())
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)

ENTITY = "Pacifastacus leniusculus (alien)"
ALGO, TRACK = "random_forest", "combined"
SPECIES_MATCH = "leniusculus"
LEVELS = (3, 10, 20)
N_REF_MODELS = 10          # reconstructed-reference ensemble size
N_BG = 10000               # background sample for the reference
N_TREES = 300


def predictors() -> list[str]:
    ps = json.load(open(PRED_JSON))
    return ps[f"{ENTITY}|||{TRACK}"]["predictors"]


def reconstruct_reference(preds, grid_ids):
    """10-RF ensemble mean trained on HIGH-accuracy presences + background,
    predicted on the complete-env grid cells. Returns a Series (subc_id -> p_ref)."""
    cols = ["subc_id", "Crayfish_scientific_name", "Accuracy"] + preds
    m = pd.read_csv(MASTER, usecols=cols)

    # grid env (complete cases on the predictor set), pick a non-NaN row per subc_id
    grid_env = (m[m.subc_id.isin(grid_ids)]
                .dropna(subset=preds)
                .drop_duplicates("subc_id")
                .set_index("subc_id"))
    print(f"  grid cells with complete combined-predictor env: "
          f"{len(grid_env)}/{len(grid_ids)} ({len(grid_env)/len(grid_ids):.1%})")

    # HIGH-accuracy presences of the target species, complete env
    pres = m[m["Crayfish_scientific_name"].str.contains(SPECIES_MATCH, case=False, na=False)]
    pres_hi = pres[(pres["Accuracy"] == "High")].dropna(subset=preds)
    X_pres = pres_hi[preds].to_numpy()
    print(f"  high-accuracy presences (complete env): {len(X_pres)}")

    # background: random grid cells (target-group not available for crayfish grid)
    rng = np.random.default_rng(0)
    bg = grid_env.sample(n=min(N_BG, len(grid_env)), random_state=0)
    X_bg = bg[preds].to_numpy()

    X_dom = grid_env[preds].to_numpy()
    preds_acc = np.zeros((N_REF_MODELS, len(grid_env)))
    for k in range(N_REF_MODELS):
        Xtr = np.vstack([X_pres, X_bg])
        ytr = np.r_[np.ones(len(X_pres)), np.zeros(len(X_bg))]
        rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1,
                                    random_state=100 + k, min_samples_leaf=2)
        rf.fit(Xtr, ytr)
        preds_acc[k] = rf.predict_proba(X_dom)[:, 1]
    p_ref = pd.Series(preds_acc.mean(axis=0), index=grid_env.index, name="p_ref")
    return p_ref


def lobo_coverage(cell, reference: pd.Series, basins: pd.Series, alpha=ALPHA):
    """LOBO conformal coverage of `reference` by the contaminated ensemble's
    intervals, uncorrected and corrected -- reference plays the benchmark's role."""
    ens = load_ensemble(cell, SURFACES_ROOT)
    shared = ens.index.intersection(reference.index).intersection(basins.index)
    ens_a = ens.loc[shared]
    ref = reference.loc[shared].to_numpy()
    basin = np.asarray(basins.loc[shared])
    lo = ens_a.quantile(alpha / 2, axis=1).to_numpy()
    hi = ens_a.quantile(1 - alpha / 2, axis=1).to_numpy()

    inside_unc = np.zeros(len(shared), bool)
    inside_cor = np.zeros(len(shared), bool)
    for b in pd.unique(basin[~pd.isna(basin)]):
        te = basin == b
        cal = ~te
        s = nonconformity_scores(pd.Series(ref[cal]), pd.Series(lo[cal]), pd.Series(hi[cal]))
        if len(s) == 0:
            continue
        q = conformal_quantile(s, alpha=alpha)
        inside_unc[te] = (ref[te] >= lo[te]) & (ref[te] <= hi[te])
        inside_cor[te] = (ref[te] >= lo[te] - q) & (ref[te] <= hi[te] + q)
    return float(inside_unc.mean()), float(inside_cor.mean()), int(len(shared))


def main() -> None:
    preds = predictors()
    print(f"combined predictors ({len(preds)}): {preds}")

    bench = benchmark_for(ENTITY, ALGO, TRACK, PATHS)
    grid_ids = set(bench.index)
    basins = pd.read_parquet(BASIN_LOOKUP).drop_duplicates("subc_id").set_index("subc_id")["basin_id"]

    print("\nreconstructing practitioner reference from high-accuracy records ...")
    p_ref = reconstruct_reference(preds, grid_ids)

    # (3) reconstruction vs benchmark on shared cells
    shared = p_ref.index.intersection(bench.index)
    r = float(np.corrcoef(p_ref.loc[shared], bench.loc[shared])[0, 1])
    mae = float(np.abs(p_ref.loc[shared].to_numpy() - bench.loc[shared].to_numpy()).mean())
    print(f"\nreconstructed reference vs companion benchmark on {len(shared)} cells:")
    print(f"  Pearson r = {r:.3f}   MAE = {mae:.3f}   "
          f"(high r EXPECTED -- shared presence data; the claim is reproducibility)")

    # (4) coverage against benchmark vs against reconstructed reference
    report = {}
    for L in LEVELS:
        cell = CellID(ENTITY, ALGO, TRACK, axis="lowacc", level=L)
        cu_b, cc_b, n_b = lobo_coverage(cell, bench, basins)
        cu_r, cc_r, n_r = lobo_coverage(cell, p_ref, basins)
        report[f"L{L}"] = {
            "vs_benchmark":  {"cov_uncorrected": round(cu_b, 3), "cov_conformal": round(cc_b, 3), "n": n_b},
            "vs_reconstructed": {"cov_uncorrected": round(cu_r, 3), "cov_conformal": round(cc_r, 3), "n": n_r},
        }
        print(f"  L{L}: benchmark {cu_b:.3f}->{cc_b:.3f} | reconstructed {cu_r:.3f}->{cc_r:.3f}")

    pd.DataFrame([
        {"level": L, **{f"{k}_{kk}": vv for k, v in d.items() for kk, vv in v.items()}}
        for L, d in report.items()
    ]).to_csv(REPORTS / "m1_deployment_coverage.csv", index=False)

    print("\n" + "=" * 66)
    pprint({"reconstruction": {"pearson_r": round(r, 3), "mae": round(mae, 3), "n_cells": len(shared)},
            "coverage": report}, sort_dicts=False)
    print("=" * 66)
    print("\nM1 supported if conformal coverage vs the reconstructed reference tracks")
    print("coverage vs the benchmark (both restored near nominal). Domain = combined-")
    print("track cells with complete env; headwater cells (structural upstream-NaN)")
    print("are an explicit out-of-scope limit. Wrote reports/m1_deployment_coverage.csv")


if __name__ == "__main__":
    main()
