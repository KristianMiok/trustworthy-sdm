"""Protocol B runner: algorithm-consensus ensembles for the trustworthy-sdm paper.

For each (entity, track) it writes:
  - a CLEAN consensus benchmark:   {dir}__consensus__{track}__benchmark__L0/
  - contaminated consensus cells:  {dir}__consensus__{track}__lowacc__L{level}/
each containing one surface parquet per algorithm (rep_00..rep_03), in the same
format the existing load_ensemble() consumes. Downstream coverage / conformal /
asymmetry analysis then runs UNCHANGED on these surfaces.

The four consensus members:
    rep_00 = GLM   (L2 logistic, VIF-reduced predictors)
    rep_01 = GAM   (LogisticGAM, VIF-reduced predictors)
    rep_02 = RF    (full track predictors)
    rep_03 = XGBoost (full track predictors)

Shared with Protocol A (identical data prep): contaminate_presence_set,
assign_basin_folds, get_track_columns, clean_predictors.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("protocol_b")

CONSENSUS_ORDER = ["glm", "gam", "random_forest", "xgboost"]
ALGO_TO_REP = {algo: f"rep_{i:02d}" for i, algo in enumerate(CONSENSUS_ORDER)}

ENTITIES = [
    "Procambarus clarkii (alien)", "Pacifastacus leniusculus (alien)",
    "Faxonius limosus (alien)", "Astacus astacus",
    "Procambarus clarkii (native)", "Pontastacus leptodactylus (pooled)",
    "Austropotamobius torrentium (pooled)", "Austropotamobius fulcisianus (pooled)",
]
TRACKS = ["local_only", "upstream_only", "combined"]
LEVELS = [3, 10, 20]
AXIS = "lowacc"
SEED = 20260617
PA_RATIO = 1.0
MISSING_THRESHOLD_PCT = 70.0
N_SPLITS = 5
LOOO_THRESHOLD = 15


def fit_one_algorithm(algo, x_train, y_train, x_pred, seed):
    """Fit one consensus member and return predicted P(presence) over x_pred rows.

    Regression members (GLM, GAM) require standardised features: without scaling,
    lbfgs fails to converge and the GAM logit link overflows. Tree members
    (RF, XGBoost) are scale-invariant and use raw features.
    """
    if algo == "glm":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed),
        )
        m.fit(x_train.values, y_train)
        return m.predict_proba(x_pred.values)[:, 1]
    if algo == "gam":
        from sklearn.preprocessing import StandardScaler
        from trustworthy_sdm.protocol_b import GAMClassifier
        scaler = StandardScaler().fit(x_train.values)
        m = GAMClassifier(n_splines=10, lam=0.6, seed=seed)
        m.fit(scaler.transform(x_train.values), y_train)
        return m.predict_proba(scaler.transform(x_pred.values))[:, 1]
    # tree-based via upstream build_model (RF/XGBoost), full predictors, no scaling
    from sdm_robustness.pipeline.core import build_model
    m = build_model(algo, seed=seed, n_jobs=-1)
    m.fit(x_train, y_train)
    p = m.predict_proba(x_pred)
    return np.asarray(p)[:, -1] if np.asarray(p).ndim == 2 else np.asarray(p, dtype=float)


def consensus_surfaces_for_cell(inputs, track, level, reduced_cols, full_cols, seed):
    """Fit all 4 algorithms once on the (contaminated if level>0) training set,
    return {algo: pd.Series indexed by subc_id} predicted over accessible area."""
    from sdm_robustness.pipeline.core import contaminate_presence_set

    benchmark = inputs.benchmark
    contamination_pool = inputs.contamination_pool
    acc = inputs.accessible_area

    # build the training presence set (clean if level==0, else contaminated)
    if level == 0:
        pres = benchmark.copy()
    else:
        pres = contaminate_presence_set(
            benchmark=benchmark, contamination_pool=contamination_pool,
            level_pct=level, seed=seed,
        ).copy()

    surfaces = {}
    for algo in CONSENSUS_ORDER:
        cols = reduced_cols if algo in {"glm", "gam"} else full_cols
        med = benchmark[cols].median(numeric_only=True)
        pres_x = pres[cols].fillna(med)
        n_neg = min(int(round(len(pres_x) * PA_RATIO)), len(acc))
        neg = acc.sample(n=n_neg, replace=False, random_state=seed)
        neg_x = neg[cols].fillna(med)
        x_train = pd.concat([pres_x, neg_x], axis=0)
        y_train = np.array([1] * len(pres_x) + [0] * len(neg_x))
        acc_x = acc[cols].fillna(med)
        pred = fit_one_algorithm(algo, x_train, y_train, acc_x, seed)
        surfaces[algo] = pd.Series(np.clip(pred, 0, 1), index=acc["subc_id"].values)
    return surfaces


def write_cell(surfaces, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for algo, s in surfaces.items():
        rep = ALGO_TO_REP[algo]
        df = pd.DataFrame({"subc_id": s.index.values,
                           "predicted_probability": s.values.astype(float)})
        df.to_parquet(out_dir / f"{rep}.parquet", index=False)


def main():
    from trustworthy_sdm.regen import assemble_inputs
    from sdm_robustness.pipeline.core import get_track_columns, clean_predictors
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces-root", default="data/replicate_surfaces_protocol_b")
    ap.add_argument("--predictor-sets", default="data/protocol_b_predictor_sets.json")
    ap.add_argument("--entities", nargs="*", default=None, help="subset for testing")
    ap.add_argument("--master-csv", default=os.environ.get("TS_MASTER_CSV"))
    args = ap.parse_args()

    surfaces_root = (REPO_ROOT / args.surfaces_root).resolve()
    pred_sets = json.load(open(REPO_ROOT / args.predictor_sets))
    master_csv = Path(args.master_csv)
    entities = args.entities or ENTITIES

    grand_t0 = time.time()
    for entity in entities:
        edir = ENTITY_NAME_TO_DIR[entity]
        t_load = time.time()
        inputs = assemble_inputs(entity=entity, track="combined", axis=AXIS, master_csv=master_csv)
        log.info(f"{entity}: loaded in {time.time()-t_load:.1f}s")

        for track in TRACKS:
            full_cols = clean_predictors(
                inputs.benchmark, get_track_columns(inputs.benchmark, track),
                missing_threshold_pct=MISSING_THRESHOLD_PCT,
            )
            reduced_cols = pred_sets[f"{entity}|||{track}"]["predictors"]

            # clean benchmark (level 0)
            t0 = time.time()
            surf = consensus_surfaces_for_cell(inputs, track, 0, reduced_cols, full_cols, SEED)
            bench_dir = surfaces_root / f"{edir}__consensus__{track}__benchmark__L0"
            write_cell(surf, bench_dir)
            log.info(f"  {track} benchmark L0: {time.time()-t0:.1f}s -> {bench_dir.name}")

            # contaminated cells
            for level in LEVELS:
                t0 = time.time()
                surf = consensus_surfaces_for_cell(inputs, track, level, reduced_cols, full_cols, SEED)
                cell_dir = surfaces_root / f"{edir}__consensus__{track}__lowacc__L{level}"
                write_cell(surf, cell_dir)
                log.info(f"  {track} L{level}: {time.time()-t0:.1f}s -> {cell_dir.name}")

    log.info(f"DONE in {time.time()-grand_t0:.1f}s. surfaces_root={surfaces_root}")


if __name__ == "__main__":
    main()
