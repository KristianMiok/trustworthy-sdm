"""Protocol B: algorithm-consensus ensemble for the trustworthy-sdm paper.

Protocol A (existing): one algorithm fit N times with seed perturbation;
    95% interval = quantile across replicates.
Protocol B (this module): K distinct algorithms each fit once on the same
    training data; ensemble = the set of K algorithm predictions;
    uncertainty = quantile across algorithms, point estimate = median.

The four consensus algorithms span regression-based (GLM, GAM) and tree-based
(RF, XGBoost) paradigms. GLM/GAM receive a VIF-reduced predictor set
(computed once on the clean benchmark per entity x track, frozen across
contamination levels); RF/XGBoost receive the full track predictor set,
exactly as in Protocol A.

This module reuses the upstream contamination + LOBO-fold machinery
(contaminate_presence_set, assign_basin_folds) so Protocol A and Protocol B
share identical data preparation and differ ONLY in ensemble construction.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONSENSUS_ALGORITHMS = ("glm", "gam", "random_forest", "xgboost")
REGRESSION_ALGORITHMS = frozenset({"glm", "gam"})  # get VIF-reduced predictors


# ======================================================================
# VIF predictor reduction (response-aware), for GLM/GAM only
# ======================================================================
def reduce_predictors_vif(
    X: pd.DataFrame,
    y: np.ndarray | None = None,
    *,
    vif_threshold: float = 10.0,
    max_keep: int = 12,
) -> list[str]:
    """Fast VIF reduction that preserves response-informative predictors.

    Uses the inverse-correlation-matrix identity (VIF_i = (R^-1)_ii) to obtain
    all variance-inflation factors in a single matrix inversion per iteration,
    rather than fitting one OLS regression per predictor. Iteratively drops the
    highest-VIF predictor; when several are near-tied at the top (within 10% of
    the max), drops the one LEAST associated with the response, so an
    informative representative of each correlated group survives.

    Deterministic given (X, y). Intended to run once on the CLEAN benchmark for
    an (entity, track), then be frozen across contamination levels.

    Returns the kept column names.
    """
    cols = [c for c in X.columns if X[c].nunique() > 1]
    if not cols:
        return []
    Xw = X[cols].copy()
    Xw = (Xw - Xw.mean()) / Xw.std(ddof=0)
    Xw = Xw.fillna(0.0)

    # response association (standardised covariance == correlation, since Xw is z-scored)
    if y is not None:
        yv = np.asarray(y, dtype=float)
        ys = (yv - yv.mean()) / (np.std(yv) + 1e-12)
        assoc = {c: abs(float(np.mean(Xw[c].values * ys))) for c in cols}
    else:
        assoc = {c: 0.0 for c in cols}

    cols = list(cols)
    while len(cols) > 1:
        M = Xw[cols].values
        R = np.corrcoef(M, rowvar=False)
        if R.ndim == 0:  # single column edge case
            break
        try:
            Rinv = np.linalg.inv(R + np.eye(len(cols)) * 1e-8)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.pinv(R)
        vifs = np.clip(np.diag(Rinv), 0.0, None)
        max_vif = float(np.nanmax(vifs))
        if max_vif < vif_threshold and len(cols) <= max_keep:
            break
        cluster = [cols[i] for i in range(len(cols)) if vifs[i] >= max_vif * 0.9]
        drop_col = min(cluster, key=lambda c: assoc[c])
        cols.remove(drop_col)
    return cols


# ======================================================================
# sklearn-compatible LogisticGAM wrapper
# ======================================================================
class GAMClassifier:
    """Numerically robust sklearn-style wrapper around pygam.LogisticGAM.

    Exposes .fit / .predict_proba (returns (n,2)) / .predict.

    Robustness measures (regression-based SDM members can otherwise overflow
    the logit link on near-separable or extreme-scale data):
      - features should be standardised by the caller before fitting;
      - probabilities are obtained via predict_mu (no manual exp), so the
        exp-overflow path in pygam's link is avoided;
      - any non-finite prediction is replaced by 0.5 and all probabilities are
        clipped to [1e-6, 1-1e-6];
      - if the fit raises, the member degrades gracefully to an uninformative
        0.5 surface (logged via .failed_) rather than emitting NaNs that would
        corrupt the consensus.
    Genuine convergence warnings are captured in .convergence_warnings_.
    """

    def __init__(self, n_splines: int = 10, lam: float = 0.6, seed: int = 0,
                 max_iter: int = 200):
        self.n_splines = n_splines
        self.lam = lam
        self.seed = seed
        self.max_iter = max_iter
        self.gam_ = None
        self.failed_ = False
        self.convergence_warnings_: list[str] = []

    def fit(self, X, y):
        from pygam import LogisticGAM, s

        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=int)
        n_features = Xv.shape[1]
        terms = s(0, n_splines=self.n_splines)
        for i in range(1, n_features):
            terms = terms + s(i, n_splines=self.n_splines)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            try:
                gam = LogisticGAM(terms, lam=self.lam, max_iter=self.max_iter)
                gam.fit(Xv, yv)
                self.gam_ = gam
                self.convergence_warnings_ = [
                    str(w.message) for w in wlist
                    if "converge" in str(w.message).lower()
                ]
            except Exception as e:  # noqa: BLE001
                self.failed_ = True
                self.convergence_warnings_ = [f"fit failed: {e}"]
        return self

    def predict_proba(self, X):
        Xv = np.asarray(X, dtype=float)
        if self.failed_ or self.gam_ is None:
            p = np.full(len(Xv), 0.5)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    p = np.asarray(self.gam_.predict_mu(Xv), dtype=float).ravel()
                except Exception:  # noqa: BLE001
                    p = np.full(len(Xv), 0.5)
            p = np.where(np.isfinite(p), p, 0.5)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# quick self-test
if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    n = 500
    base = rng.normal(size=(n, 4))
    block = base @ rng.normal(size=(4, 16)) + rng.normal(scale=0.1, size=(n, 16))
    Xf = np.column_stack([base, block, rng.normal(size=(n, 5))])
    cols = [f"l_V{i}" for i in range(Xf.shape[1])]
    Xdf = pd.DataFrame(Xf, columns=cols)
    yv = (rng.uniform(size=n) < 1/(1+np.exp(-(base @ [1.5,-1,.8,-.6])))).astype(int)
    kept = reduce_predictors_vif(Xdf, yv, max_keep=12)
    print(f"VIF: {len(cols)} -> {len(kept)} kept")
    g = GAMClassifier().fit(Xdf[kept].values, yv)
    print("GAM proba shape:", g.predict_proba(Xdf[kept].values).shape)
    print("convergence warnings:", len(g.convergence_warnings_))
    print("module self-test OK")
