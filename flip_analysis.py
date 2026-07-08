"""
flip_analysis.py
================
Worked-case flip analysis for the JAE paper
(Pacifastacus leniusculus, Central European invasion front).

WHY THIS IS DEFINED THE WAY IT IS
---------------------------------
The conformal correction calibrates prediction INTERVALS (width inflation);
it does NOT de-bias the point suitability. So a "flip" must live in the
DECISION a manager takes under calibrated uncertainty, not in movement of the
point estimate. A naive "corrected point map - uncorrected point map" gives
zero flips (points don't move) and invites the reviewer objection:
"your correction changes intervals, not predictions -- so what decision changed?"

The answer: a manager conditions action on confidence, not on the bare point.
The correction widens overconfident intervals, which withdraws false-confident
"suitable" cells (and reopens wrongly-dismissed ones). The correction is a
reliability filter, not a point de-biaser -- and that IS the contribution.

DECISION RULE
-------------
For a suitability threshold tau and a prediction interval [lo, hi]:
    actionable-suitable  (SURVEIL)  : lo > tau     # confidently above tau
    confidently-unsuitable (DISMISS): hi < tau     # confidently below tau
    uncertain                       : otherwise    # interval straddles tau

    FLIP A (withdrawn) : SURVEIL uncorrected, NOT corrected
                         (lo_unc > tau) & (lo_cor <= tau)
                         -> false-confident suitable cells pulled back.
                            Budget-saving story at the invasion periphery.

    FLIP B (recovered) : DISMISS uncorrected, NOT corrected
                         (hi_unc < tau) & (hi_cor >= tau)
                         -> wrongly written-off cells reopened for checking.
                            Lucian's "or vice versa".

INPUT CONTRACT
--------------
pandas DataFrame, one row per grid cell of the modelling domain, columns:
    cell_id : any cell identifier (kept only so you can join back to geometry)
    p       : point suitability (contaminated ensemble)
    lo_unc, hi_unc : UNCORRECTED interval bounds
    lo_cor, hi_cor : CORRECTED (conformal) interval bounds
    p_bench : (optional) contamination-free benchmark point suitability;
              exists only in the experiment. Used ONLY for the M1 bridge,
              never for the flip counts.

Map your pipeline's per-cell outputs onto these columns. If your correction is
stored as a per-cell width-inflation FACTOR rather than explicit corrected
bounds, build lo_cor/hi_cor with corrected_bounds_from_inflation() first.

tau: pass whatever threshold the main text already uses (fixed 0.5, maxSSS,
prevalence-based, ...). Consistency with the rest of the paper matters more
than the choice of rule. maxSSS_threshold() is here if you want it.

FIGURE
------
This module deliberately does not draw the map -- your main figures already set
the quality standard. Join the returned flip_A_withdrawn / flip_B_recovered
columns back to your grid geometry and render them as a categorical overlay in
your existing figure pipeline (candidate Fig 5, or a panel on an existing one).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["cell_id", "p", "lo_unc", "hi_unc", "lo_cor", "hi_cor"]


def corrected_bounds_from_inflation(p, lo_unc, hi_unc, inflation):
    """Corrected interval = point +/- (uncorrected half-width * inflation).

    Use ONLY if your conformal step is expressed as a per-cell inflation factor
    on the interval half-width rather than as explicit corrected bounds.
    Clipped to [0, 1] for a suitability score.
    """
    p = np.asarray(p, float)
    half_unc = (np.asarray(hi_unc, float) - np.asarray(lo_unc, float)) / 2.0
    half_cor = half_unc * np.asarray(inflation, float)
    return np.clip(p - half_cor, 0.0, 1.0), np.clip(p + half_cor, 0.0, 1.0)


def maxSSS_threshold(scores_pres, scores_bg):
    """tau maximising (sensitivity + specificity) from presence/background scores.

    Only use if you don't already have a threshold from the main pipeline.
    """
    s_p = np.asarray(scores_pres, float)
    s_b = np.asarray(scores_bg, float)
    cand = np.unique(np.concatenate([s_p, s_b]))
    sens = np.array([(s_p >= t).mean() for t in cand])
    spec = np.array([(s_b < t).mean() for t in cand])
    return float(cand[np.argmax(sens + spec)])


def _decision(lo, hi, tau):
    """Vectorised decision state: 1=SURVEIL, -1=DISMISS, 0=UNCERTAIN."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    state = np.zeros(len(lo), dtype=int)
    state[lo > tau] = 1
    state[hi < tau] = -1
    return state


def compute_flips(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Add per-cell decision states, flip labels, widths and inflation."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = df.copy()
    out["dec_unc"] = _decision(out["lo_unc"], out["hi_unc"], tau)
    out["dec_cor"] = _decision(out["lo_cor"], out["hi_cor"], tau)
    out["flip_A_withdrawn"] = (out["lo_unc"] > tau) & (out["lo_cor"] <= tau)
    out["flip_B_recovered"] = (out["hi_unc"] < tau) & (out["hi_cor"] >= tau)
    out["width_unc"] = out["hi_unc"] - out["lo_unc"]
    out["width_cor"] = out["hi_cor"] - out["lo_cor"]
    out["width_inflation"] = out["width_cor"] / out["width_unc"].replace(0, np.nan)
    return out


def summarise(flips: pd.DataFrame, tau: float) -> dict:
    """The numbers to report: counts, domain fraction, width inflation at flips."""
    n = len(flips)
    A = flips["flip_A_withdrawn"].to_numpy()
    B = flips["flip_B_recovered"].to_numpy()

    def infl(mask):
        v = flips.loc[mask, "width_inflation"].to_numpy()
        v = v[np.isfinite(v)]
        if not len(v):
            return {"median": float("nan"), "mean": float("nan")}
        return {"median": float(np.median(v)), "mean": float(np.mean(v))}

    return {
        "tau": tau,
        "n_cells_domain": n,
        "flip_A_withdrawn": {"n": int(A.sum()), "frac_domain": float(A.mean()),
                             "width_inflation": infl(A)},
        "flip_B_recovered": {"n": int(B.sum()), "frac_domain": float(B.mean()),
                             "width_inflation": infl(B)},
        "width_inflation_all": infl(np.ones(n, bool)),
        "decision_shift": {
            "n_surveil_uncorrected": int((flips["dec_unc"] == 1).sum()),
            "n_surveil_corrected":   int((flips["dec_cor"] == 1).sum()),
            "n_dismiss_uncorrected": int((flips["dec_unc"] == -1).sum()),
            "n_dismiss_corrected":   int((flips["dec_cor"] == -1).sum()),
        },
    }


def benchmark_bridge(flips: pd.DataFrame) -> dict:
    """Experiment-only check that ties the worked case to deployment realism (M1).

    Requires 'p_bench'. Shows (a) flipped cells carry larger contaminated-vs-
    benchmark divergence than non-flipped cells, and (b) the width-inflation
    factor ranks the divergent cells -- i.e. on the ground, where no benchmark
    exists, width inflation alone recovers the same signal. (For the paper use
    scipy.stats.spearmanr; the no-scipy rank correlation below is for a quick run.)
    """
    if "p_bench" not in flips.columns:
        return {"available": False, "reason": "no p_bench column"}
    div = (flips["p"] - flips["p_bench"]).abs().to_numpy()
    flipped = (flips["flip_A_withdrawn"] | flips["flip_B_recovered"]).to_numpy()
    infl = flips["width_inflation"].to_numpy()
    ok = np.isfinite(infl)

    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x))
        return r

    rho = (float(np.corrcoef(_rank(infl[ok]), _rank(div[ok]))[0, 1])
           if ok.sum() > 2 else float("nan"))
    return {
        "available": True,
        "mean_divergence_flipped": float(div[flipped].mean()) if flipped.any() else float("nan"),
        "mean_divergence_not_flipped": float(div[~flipped].mean()) if (~flipped).any() else float("nan"),
        "rank_corr_widthinflation_vs_divergence": rho,
    }


if __name__ == "__main__":
    # Synthetic smoke test -- delete once wired to the real pipeline.
    from pprint import pprint
    rng = np.random.default_rng(0)
    n = 2000
    p = rng.beta(2, 3, n)
    half_unc = rng.uniform(0.03, 0.08, n)                       # overconfident, narrow
    lo_unc, hi_unc = np.clip(p - half_unc, 0, 1), np.clip(p + half_unc, 0, 1)
    inflation = 1.0 + rng.uniform(0.5, 2.5, n)                  # correction widens
    lo_cor, hi_cor = corrected_bounds_from_inflation(p, lo_unc, hi_unc, inflation)
    p_bench = np.clip(p - 0.15 * (inflation - 1.0) / 2.5 + rng.normal(0, 0.02, n), 0, 1)

    df = pd.DataFrame(dict(cell_id=np.arange(n), p=p, lo_unc=lo_unc, hi_unc=hi_unc,
                           lo_cor=lo_cor, hi_cor=hi_cor, p_bench=p_bench))
    tau = 0.5
    flips = compute_flips(df, tau)
    print("=== summarise ===");      pprint(summarise(flips, tau))
    print("=== benchmark_bridge ==="); pprint(benchmark_bridge(flips))
