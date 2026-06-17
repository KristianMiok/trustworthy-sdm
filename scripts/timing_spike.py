"""Protocol B timing spike — worst-case cell.

Runs all four consensus algorithms (GLM, GAM, RF, XGBoost) once each through
the FULL LOBO fold structure on the single worst-case cell:
    Procambarus clarkii (alien), combined track, highest contamination level.
Logs, per algorithm: wall time, peak memory, and GAM convergence warnings.

Purpose: produce a real wall-time + memory projection to decide whether the
full 72-cell Protocol B panel is tractable, or whether to scope down to
4-5 representative entities. Reports a projection at the end.

Run on the machine that has the master CSV + the .venv:
    cd ~/Desktop/Papers/trustworthy-sdm
    export TS_MASTER_CSV=/Users/kristianmiok/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv
    .venv/bin/python scripts/timing_spike.py
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- the worst-case cell ---
ENTITY = "Procambarus clarkii (alien)"
TRACK = "combined"
AXIS = "lowacc"     # low-accuracy contamination axis (vs "snapping")
LEVEL = 20          # highest low-accuracy contamination
SEED = 20260617
N_SPLITS = 5
LOOO_THRESHOLD = 15

# VIF reduction params for GLM/GAM
VIF_THRESHOLD = 10.0
VIF_MAX_KEEP = 12


def log(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def get_master_csv() -> Path:
    p = os.environ.get("TS_MASTER_CSV")
    if not p:
        raise SystemExit("Set TS_MASTER_CSV to the master CSV path first.")
    path = Path(p)
    if not path.exists():
        raise SystemExit(f"TS_MASTER_CSV does not exist: {path}")
    return path


def main() -> None:
    from sdm_robustness.pipeline.core import (
        get_track_columns,
        clean_predictors,
        contaminate_presence_set,
        assign_basin_folds,
        build_model,
        predict_suitability_surface,
        compute_performance_metrics,
    )
    from trustworthy_sdm.regen import assemble_inputs
    from trustworthy_sdm.protocol_b import (
        reduce_predictors_vif,
        GAMClassifier,
        CONSENSUS_ALGORITHMS,
        REGRESSION_ALGORITHMS,
    )

    master_csv = get_master_csv()
    log(f"loading entity data for: {ENTITY}")
    t_load = time.time()
    inputs = assemble_inputs(
        entity=ENTITY, track=TRACK, axis=AXIS, master_csv=master_csv,
    )
    benchmark = inputs.benchmark
    contamination_pool = inputs.contamination_pool
    accessible_area = inputs.accessible_area
    log(f"  loaded in {time.time()-t_load:.1f}s  "
        f"(benchmark={len(benchmark)}, pool={len(contamination_pool)}, acc={len(accessible_area)})")

    feat_cols = get_track_columns(benchmark, TRACK)
    kept = clean_predictors(benchmark, feat_cols)
    log(f"  full track predictors: {len(kept)}")

    # Contaminate + assign folds (shared machinery, same as Protocol A)
    medians = benchmark[kept].median(numeric_only=True)
    contaminated = contaminate_presence_set(
        benchmark=benchmark, contamination_pool=contamination_pool,
        level_pct=LEVEL, seed=SEED,
    ).copy()
    contaminated = contaminated[["subc_id", "basin_id"] + kept].copy()
    contaminated[kept] = contaminated[kept].fillna(medians)
    acc = accessible_area[["subc_id", "basin_id"] + kept].copy()
    acc[kept] = acc[kept].fillna(medians)

    fold_map = assign_basin_folds(
        contaminated["basin_id"], n_splits=N_SPLITS, looo_threshold=LOOO_THRESHOLD,
    )
    contaminated["fold"] = contaminated["basin_id"].astype(str).map(fold_map)
    folds = sorted(pd.Series(contaminated["fold"]).dropna().unique().tolist())
    log(f"  LOBO folds: {len(folds)}")

    # VIF reduction for GLM/GAM, computed once on the (contaminated) presence set
    # In production this is computed on CLEAN benchmark and frozen; for the spike
    # we just need representative timing, so compute on what we have.
    log("computing VIF-reduced predictor set for GLM/GAM ...")
    t_vif = time.time()
    pres_only = contaminated[contaminated["fold"].notna()]
    y_assoc = np.ones(len(pres_only))  # placeholder; real run uses pres/abs labels
    # Build a quick pres/background matrix for association ranking
    neg_for_vif = acc.sample(n=min(len(pres_only), len(acc)), replace=False, random_state=SEED)
    X_vif = pd.concat([pres_only[kept], neg_for_vif[kept]], axis=0).reset_index(drop=True)
    y_vif = np.array([1]*len(pres_only) + [0]*len(neg_for_vif))
    reduced_cols = reduce_predictors_vif(
        X_vif, y_vif, vif_threshold=VIF_THRESHOLD, max_keep=VIF_MAX_KEEP,
    )
    log(f"  VIF reduced {len(kept)} -> {len(reduced_cols)} in {time.time()-t_vif:.1f}s")
    log(f"  kept: {reduced_cols}")

    results = {}

    for algo in CONSENSUS_ALGORITHMS:
        use_cols = reduced_cols if algo in REGRESSION_ALGORITHMS else kept
        log(f"--- {algo}  ({len(use_cols)} predictors) ---")
        tracemalloc.start()
        t0 = time.time()
        n_folds_done = 0
        conv_warnings = []
        try:
            for fold in folds:
                pres_train = contaminated[contaminated["fold"] != fold]
                pres_test = contaminated[contaminated["fold"] == fold]
                if pres_train.empty or pres_test.empty:
                    continue
                n_neg = len(contaminated)
                if n_neg > len(acc):
                    n_neg = len(acc)
                neg = acc.sample(n=n_neg, replace=False, random_state=SEED + int(fold))
                neg["fold"] = neg["basin_id"].astype(str).map(fold_map).fillna(0).astype(int)
                neg_train = neg[neg["fold"] != fold]
                neg_test = neg[neg["fold"] == fold]

                x_train = pd.concat([pres_train[use_cols], neg_train[use_cols]], axis=0)
                y_train = np.array([1]*len(pres_train) + [0]*len(neg_train))
                x_test = pd.concat([pres_test[use_cols], neg_test[use_cols]], axis=0)
                y_test = np.array([1]*len(pres_test) + [0]*len(neg_test))
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                if algo == "glm":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(
                        C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED,
                    )
                    model.fit(x_train.values, y_train)
                elif algo == "gam":
                    model = GAMClassifier(n_splines=10, lam=0.6, seed=SEED)
                    model.fit(x_train.values, y_train)
                    conv_warnings.extend(model.convergence_warnings_)
                else:
                    model = build_model(algo, seed=SEED, n_jobs=-1)
                    model.fit(x_train, y_train)
                n_folds_done += 1
            dt = time.time() - t0
            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[algo] = {
                "wall_s": dt, "peak_mb": peak/1e6,
                "folds": n_folds_done, "conv_warnings": len(conv_warnings),
                "status": "ok",
            }
            log(f"  {algo}: {dt:.1f}s, peak {peak/1e6:.0f} MB, "
                f"{n_folds_done} folds, {len(conv_warnings)} convergence warnings")
        except Exception as e:
            tracemalloc.stop()
            results[algo] = {"status": f"FAILED: {type(e).__name__}: {e}"}
            log(f"  {algo} FAILED: {type(e).__name__}: {e}")

    # ---- projection ----
    log("")
    log("=" * 60)
    log("PROJECTION")
    log("=" * 60)
    per_cell = sum(r.get("wall_s", 0) for r in results.values())
    log(f"one cell, all 4 algorithms, full LOBO: {per_cell:.1f}s "
        f"({per_cell/60:.1f} min)")
    # full panel = 8 entities x 3 tracks x 3 levels = 72 cells
    # but this cell is the WORST case; real mean will be lower
    full_72 = per_cell * 72
    log(f"naive full-72 upper bound (worst cell x 72): "
        f"{full_72/3600:.1f} h single-threaded")
    log(f"  (real panel lower — this is the largest entity at highest contamination)")
    log(f"representative-5 entities x 3 tracks x 3 levels = 45 cells: "
        f"{per_cell*45/3600:.1f} h upper bound")
    log("")
    for algo, r in results.items():
        if r.get("status") == "ok":
            log(f"  {algo:14s}: {r['wall_s']:6.1f}s  peak {r['peak_mb']:5.0f}MB  "
                f"warnings={r['conv_warnings']}")
        else:
            log(f"  {algo:14s}: {r['status']}")


if __name__ == "__main__":
    main()
