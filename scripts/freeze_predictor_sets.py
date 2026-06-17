"""Freeze the VIF-reduced predictor sets for GLM/GAM, once, on CLEAN data.

For each (entity x track) the GLM/GAM consensus members use a VIF-reduced
predictor set. To keep that set fixed across contamination levels and
replicates (so contamination cannot change WHICH predictors a model uses),
we compute it once here on the clean benchmark and cache it to JSON.

The real Protocol B runner reads this cache instead of recomputing VIF.

Run:
    cd ~/Desktop/Papers/trustworthy-sdm
    export TS_MASTER_CSV=/Users/kristianmiok/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv
    .venv/bin/python scripts/freeze_predictor_sets.py

Output:
    data/protocol_b_predictor_sets.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The 8 DUAL-AXIS entities (same panel as Protocol A)
ENTITIES = [
    "Procambarus clarkii (alien)",
    "Pacifastacus leniusculus (alien)",
    "Faxonius limosus (alien)",
    "Astacus astacus",
    "Procambarus clarkii (native)",
    "Pontastacus leptodactylus (pooled)",
    "Austropotamobius torrentium (pooled)",
    "Austropotamobius fulcisianus (pooled)",
]
TRACKS = ["local_only", "upstream_only", "combined"]
AXIS = "lowacc"
VIF_THRESHOLD = 10.0
VIF_MAX_KEEP = 12
PA_RATIO = 1.0
SEED = 20260617
# Protocol B relaxes the clean_predictors missingness threshold from the
# default 30% to 70%, so catchment (upstream) predictors that are ~63%
# complete for four entities survive and are median-imputed rather than
# dropped. This keeps the upstream_only track non-degenerate for those
# entities. Protocol A is unaffected (uses the default 30%).
MISSING_THRESHOLD_PCT = 70.0

OUT_PATH = REPO_ROOT / "data" / "protocol_b_predictor_sets.json"


def log(msg: str) -> None:
    print(f"[freeze] {msg}", flush=True)


def main() -> None:
    from sdm_robustness.pipeline.core import get_track_columns, clean_predictors
    from trustworthy_sdm.regen import assemble_inputs
    from trustworthy_sdm.protocol_b import reduce_predictors_vif

    master_csv = os.environ.get("TS_MASTER_CSV")
    if not master_csv:
        raise SystemExit("Set TS_MASTER_CSV first.")
    master_csv = Path(master_csv)

    result: dict = {
        "_meta": {
            "vif_threshold": VIF_THRESHOLD,
            "max_keep": VIF_MAX_KEEP,
            "pa_ratio": PA_RATIO,
            "seed": SEED,
            "axis": AXIS,
            "missing_threshold_pct": MISSING_THRESHOLD_PCT,
            "note": "VIF-reduced predictor sets for GLM/GAM consensus members, "
                    "computed on CLEAN benchmark (level=0), frozen across "
                    "contamination levels. RF/XGBoost use the full track set. "
                    "Missingness threshold relaxed to 70% (vs upstream default 30%) "
                    "so ~63%-complete catchment predictors are imputed, not dropped.",
        }
    }

    grand_t0 = time.time()
    for entity in ENTITIES:
        # load once per entity (data prep is track-agnostic in the columns it returns;
        # track filtering happens via get_track_columns below)
        t_load = time.time()
        inputs = assemble_inputs(
            entity=entity, track="combined", axis=AXIS, master_csv=master_csv,
        )
        benchmark = inputs.benchmark
        accessible_area = inputs.accessible_area
        log(f"{entity}: loaded benchmark={len(benchmark)} acc={len(accessible_area)} "
            f"in {time.time()-t_load:.1f}s")

        for track in TRACKS:
            feat_cols = get_track_columns(benchmark, track)
            kept_full = clean_predictors(
                benchmark, feat_cols, missing_threshold_pct=MISSING_THRESHOLD_PCT,
            )

            # Build clean pres/background matrix for VIF + response association.
            # CLEAN = uncontaminated benchmark presences vs accessible-area background.
            medians = benchmark[kept_full].median(numeric_only=True)
            pres = benchmark[kept_full].fillna(medians)
            n_neg = min(int(round(len(pres) * PA_RATIO)), len(accessible_area))
            neg = accessible_area.sample(n=n_neg, replace=False, random_state=SEED)
            neg = neg[kept_full].fillna(medians)
            X = pd.concat([pres, neg], axis=0).reset_index(drop=True).copy()
            y = np.array([1] * len(pres) + [0] * len(neg))

            t0 = time.time()
            reduced = reduce_predictors_vif(
                X, y, vif_threshold=VIF_THRESHOLD, max_keep=VIF_MAX_KEEP,
            )
            dt = time.time() - t0
            key = f"{entity}|||{track}"
            result[key] = {
                "entity": entity,
                "track": track,
                "n_full": int(len(kept_full)),
                "n_reduced": int(len(reduced)),
                "predictors": reduced,
            }
            log(f"  {track:14s}: {len(kept_full):3d} -> {len(reduced):2d} preds in {dt:.1f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {OUT_PATH}")
    log(f"total time {time.time()-grand_t0:.1f}s for {len(ENTITIES)} entities x {len(TRACKS)} tracks")


if __name__ == "__main__":
    main()
