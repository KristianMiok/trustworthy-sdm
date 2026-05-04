# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
# %% [markdown]
# # Pilot: ensemble coverage of contaminated SDM surfaces
#
# **Goal of this notebook.** Take the regenerated 30-replicate surfaces for
# *Austropotamobius torrentium* (pooled), combined-track, RF + XGBoost, at
# benchmark (level 0) and lowacc max (level 20). Compute pixel-wise ensemble
# mean and variance from the 30 members. Use the benchmark mean as the
# "ground-truth" target. Compute empirical coverage of the level-20 ensemble's
# 95% interval against the benchmark mean.
#
# **What this should show, if Paper A's headline is real.** Coverage at level 0
# should be near 0.95 (an honest interval); coverage at level 20 should be
# materially below 0.95 (under-coverage from a too-narrow interval that does
# not know about the contamination).
#
# **What it would show if we're wrong.** Coverage close to 0.95 at both
# levels — meaning the ensemble already captures the contamination shift, and
# Paper A's premise is wrong. We'd need to think hard about what to do next.
#
# This is the smallest possible test of the paper's central claim.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustworthy_sdm.io import CellID

# %% [markdown]
# ## Configuration

# %%
ENTITY = "Austropotamobius torrentium (pooled)"
ENTITY_DIR = "Austropotamobius_torrentium_pooled"
TRACK = "combined"

REPLICATE_SURFACES_ROOT = Path("./data/replicate_surfaces").resolve()
ALPHA = 0.05  # 95% interval


# %% [markdown]
# ## Load all 30 replicate surfaces for one cell

# %%
def load_ensemble(cell: CellID, root: Path) -> pd.DataFrame:
    """Load all replicate surfaces for one cell into a wide DataFrame.

    Returns a DataFrame indexed by ``subc_id`` with one column per replicate.
    Replicates with missing files are skipped with a warning.
    """
    cell_dir = root / cell.short()
    if not cell_dir.exists():
        raise FileNotFoundError(f"missing replicate-surface dir: {cell_dir}")

    ensemble = None
    n_loaded = 0
    for rep in range(30):
        path = cell_dir / f"rep_{rep:02d}.parquet"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        df = pd.read_parquet(path).set_index("subc_id")["predicted_probability"]
        df = df.rename(f"rep_{rep:02d}")
        if ensemble is None:
            ensemble = df.to_frame()
        else:
            ensemble = ensemble.join(df, how="outer")
        n_loaded += 1

    if ensemble is None:
        raise RuntimeError(f"no replicate surfaces loaded from {cell_dir}")
    print(f"  loaded {n_loaded}/30 replicates for {cell.short()}")
    return ensemble


# %%
cells = {
    "rf_bench": CellID(ENTITY, "random_forest", TRACK, "benchmark", 0),
    "rf_l20": CellID(ENTITY, "random_forest", TRACK, "lowacc", 20),
    "xgb_bench": CellID(ENTITY, "xgboost", TRACK, "benchmark", 0),
    "xgb_l20": CellID(ENTITY, "xgboost", TRACK, "lowacc", 20),
}

ensembles = {name: load_ensemble(cell, REPLICATE_SURFACES_ROOT) for name, cell in cells.items()}
for name, ens in ensembles.items():
    print(f"{name}: shape={ens.shape}")


# %% [markdown]
# ## Compute pixel-wise ensemble statistics

# %%
def ensemble_stats(ens: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean": ens.mean(axis=1),
        "sd": ens.std(axis=1, ddof=1),
        "lo95": ens.quantile(ALPHA / 2, axis=1),
        "hi95": ens.quantile(1 - ALPHA / 2, axis=1),
    })


stats = {name: ensemble_stats(ens) for name, ens in ensembles.items()}
for name, s in stats.items():
    print(f"{name}: mean range [{s['mean'].min():.3f}, {s['mean'].max():.3f}], "
          f"median width = {(s['hi95'] - s['lo95']).median():.3f}")


# %% [markdown]
# ## Coverage analysis
#
# Definition: at each segment, does the level-20 ensemble's 95% empirical
# interval contain the level-0 ensemble mean?
#
# Honest interval: ~95% of segments covered.
# Miscalibrated interval: <95% — the contaminated ensemble is overconfident
# about a target it does not match.

# %%
def coverage(target: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    """Fraction of `target` values that fall inside [lo, hi]."""
    aligned = pd.concat({"t": target, "lo": lo, "hi": hi}, axis=1).dropna()
    inside = (aligned["t"] >= aligned["lo"]) & (aligned["t"] <= aligned["hi"])
    return float(inside.mean())


print("=== empirical coverage of level-20 95% interval vs level-0 mean ===")
print(f"RF:      {coverage(stats['rf_bench']['mean'], stats['rf_l20']['lo95'], stats['rf_l20']['hi95']):.3f}")
print(f"XGBoost: {coverage(stats['xgb_bench']['mean'], stats['xgb_l20']['lo95'], stats['xgb_l20']['hi95']):.3f}")

print("\n=== sanity: coverage of level-0 95% interval vs level-0 mean (should be ≈1.0) ===")
print(f"RF:      {coverage(stats['rf_bench']['mean'], stats['rf_bench']['lo95'], stats['rf_bench']['hi95']):.3f}")
print(f"XGBoost: {coverage(stats['xgb_bench']['mean'], stats['xgb_bench']['lo95'], stats['xgb_bench']['hi95']):.3f}")


# %% [markdown]
# ## Figure 1 (draft): pixel-wise interval width vs prediction
#
# A wider interval at higher contamination would be a self-aware ensemble.
# A flat or even narrower interval is the headline failure mode.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, algo in zip(axes, ["rf", "xgb"], strict=True):
    s_b = stats[f"{algo}_bench"]
    s_c = stats[f"{algo}_l20"]
    ax.scatter(s_b["mean"], s_b["hi95"] - s_b["lo95"], s=2, alpha=0.3, label="benchmark (L0)")
    ax.scatter(s_c["mean"], s_c["hi95"] - s_c["lo95"], s=2, alpha=0.3, label="lowacc max (L20)")
    ax.set_xlabel("ensemble mean predicted probability")
    ax.set_title("Random Forest" if algo == "rf" else "XGBoost")
    ax.legend(markerscale=4)
axes[0].set_ylabel("95% interval width")
fig.suptitle("Pixel-wise ensemble interval width by contamination level — A. torrentium combined")
fig.tight_layout()
fig.savefig("./figures/01_pilot_width_vs_prediction.png", dpi=150)
plt.show()
