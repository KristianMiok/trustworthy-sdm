"""Protocol B adequacy check (Table 1) — REAL AUC/TSS per (entity x track).

Re-fits the clean (L0) 4-algorithm consensus and predicts at BOTH presence
locations and the target-group background, then scores presence-vs-background:
  - AUC  (threshold-free discrimination)
  - TSS  (max of sensitivity+specificity-1 over thresholds; standard SDM convention)
  - ens_coherence (mean pairwise Pearson among the 4 member surfaces over the
    prediction domain) -- the honest replacement for the ill-defined
    "Pearson correlation with the clean-model surface" (that would be self-corr=1).

These are APPARENT (resubstitution) metrics: presences are the training presences,
not a held-out set, so report them as in-sample adequacy, not cross-validated skill.

Usage:
    .venv/bin/python scripts/adequacy_check.py --entity "Astacus astacus" --track local_only   # smoke test
    .venv/bin/python scripts/adequacy_check.py --all                                            # full 24-cell table
"""
from __future__ import annotations
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from trustworthy_sdm.regen import assemble_inputs  # noqa: E402
from trustworthy_sdm.analysis import DUAL_AXIS_ENTITIES, TRACKS  # noqa: E402

# reuse the EXACT fit logic from the runner so adequacy matches the surfaces
sys.path.insert(0, str(REPO / "scripts"))
from run_protocol_b import fit_one_algorithm, CONSENSUS_ORDER, PA_RATIO  # noqa: E402
from sdm_robustness.pipeline.core import get_track_columns  # noqa: E402

AXIS = "lowacc"
import json
PRED_SETS = json.load(open(REPO / "data" / "protocol_b_predictor_sets.json"))


def reduced_cols_for(entity, track):
    key = f"{entity}|||{track}"
    if key in PRED_SETS and key != "_meta":
        return PRED_SETS[key]["predictors"]
    return None


def tss_score(y_true, y_score):
    """Max TSS over candidate thresholds (standard SDM convention)."""
    order = np.argsort(y_score)
    ys = np.asarray(y_true)[order]
    P = ys.sum(); N = len(ys) - P
    if P == 0 or N == 0:
        return np.nan
    thresholds = np.unique(y_score)
    best = -1.0
    for t in thresholds:
        pred = y_score >= t
        tp = np.sum(pred & (np.asarray(y_true) == 1))
        fp = np.sum(pred & (np.asarray(y_true) == 0))
        sens = tp / P
        spec = 1 - fp / N
        best = max(best, sens + spec - 1)
    return best


def adequacy_one(entity, track, master_csv, seed=0):
    inp = assemble_inputs(entity=entity, track=track, axis=AXIS, master_csv=master_csv)
    benchmark = inp.benchmark
    acc = inp.accessible_area
    full_cols = get_track_columns(benchmark, track)
    reduced = reduced_cols_for(entity, track)
    reduced_cols = reduced if reduced else full_cols

    # clean presences = benchmark; background = accessible area (target-group)
    pres = benchmark.copy()
    member_pred_pres = {}
    member_pred_bg = {}
    for algo in CONSENSUS_ORDER:
        cols = reduced_cols if algo in {"glm", "gam"} else full_cols
        med = benchmark[cols].median(numeric_only=True)
        pres_x = pres[cols].fillna(med)
        n_neg = min(int(round(len(pres_x) * PA_RATIO)), len(acc))
        neg = acc.sample(n=n_neg, replace=False, random_state=seed)
        neg_x = neg[cols].fillna(med)
        x_train = pd.concat([pres_x, neg_x], axis=0)
        y_train = np.array([1] * len(pres_x) + [0] * len(neg_x))
        # predict at presences and at ALL background
        pred_pres = fit_one_algorithm(algo, x_train, y_train, pres_x, seed)
        pred_bg = fit_one_algorithm(algo, x_train, y_train, acc[cols].fillna(med), seed)
        member_pred_pres[algo] = np.clip(pred_pres, 0, 1)
        member_pred_bg[algo] = np.clip(pred_bg, 0, 1)

    # consensus = mean across the 4 members (matches Protocol B point estimate)
    # NOTE manuscript says median for point estimate -- we use median here)
    cons_pres = np.mean(np.column_stack([member_pred_pres[a] for a in CONSENSUS_ORDER]), axis=1)
    cons_bg = np.mean(np.column_stack([member_pred_bg[a] for a in CONSENSUS_ORDER]), axis=1)

    y_true = np.r_[np.ones(len(cons_pres)), np.zeros(len(cons_bg))]
    y_score = np.r_[cons_pres, cons_bg]
    auc = roc_auc_score(y_true, y_score)
    tss = tss_score(y_true, y_score)

    # ensemble coherence: mean pairwise Pearson among members over background domain
    M = np.column_stack([member_pred_bg[a] for a in CONSENSUS_ORDER])
    cors = []
    for i in range(M.shape[1]):
        for j in range(i + 1, M.shape[1]):
            c = np.corrcoef(M[:, i], M[:, j])[0, 1]
            cors.append(c)
    ens_coh = float(np.nanmean(cors))

    return dict(entity=entity, track=track, n_presence=len(cons_pres),
                n_background=len(cons_bg), AUC=round(auc, 3),
                TSS=round(float(tss), 3), ens_coherence=round(ens_coh, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    master_csv = Path(os.environ["TS_MASTER_CSV"])
    warnings.filterwarnings("ignore")

    if args.all:
        rows = []
        for e in DUAL_AXIS_ENTITIES:
            for t in TRACKS:
                try:
                    r = adequacy_one(e, t, master_csv)
                    rows.append(r)
                    print(f"{e:42s} {t:14s} AUC={r['AUC']:.3f} TSS={r['TSS']:.3f} coh={r['ens_coherence']:.3f}")
                except Exception as ex:  # noqa: BLE001
                    print(f"FAIL {e}|{t}: {ex}")
        df = pd.DataFrame(rows)
        out = REPO / "figures" / "adequacy_check_protocol_b.csv"
        df.to_csv(out, index=False)
        print(f"\nwrote {out} ({len(df)} cells)")
        print("\n=== Table 1 ranges ===")
        for m in ["AUC", "TSS", "ens_coherence"]:
            print(f"  {m}: {df[m].min():.3f} – {df[m].max():.3f} (median {df[m].median():.3f})")
    else:
        e = args.entity or "Astacus astacus"
        t = args.track or "local_only"
        r = adequacy_one(e, t, master_csv)
        print("SMOKE TEST:", r)


if __name__ == "__main__":
    main()
