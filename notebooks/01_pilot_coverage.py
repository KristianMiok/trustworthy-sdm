# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
# %% [markdown]
# # Pilot: ensemble coverage of contaminated SDM surfaces (Path 2)
#
# **Goal.** Take the regenerated 30-replicate ensembles for *Austropotamobius
# torrentium* (pooled), combined-track, RF + XGBoost, at lowacc levels
# 3, 10, 20. Use the companion paper's saved deterministic benchmark
# surface as the reference target. Compute pixel-wise coverage of each
# contaminated ensemble's 95% empirical interval against the benchmark.
#
# **Path 2 limitation flagged.** This pilot uses a single deterministic
# benchmark, not a benchmark ensemble. We cannot make claims about whether
# the benchmark itself is well-calibrated. The paper claim is asymmetric:
# "contaminated ensembles produce overconfident intervals against the
# clean reference." If the pilot is interesting, we upgrade to symmetric
# benchmark ensembles for the full panel (Path 1, ~28 h on VEGA).
#
# **What this pilot should show, if Paper A's headline is real.**
# Coverage at L3 close to 0.95, dropping monotonically through L10 to L20.
# Width should also rise monotonically with contamination — that's the
# diagnostic signal.
#
# **What it would show if we're wrong.** Flat coverage near 0.95, flat
# width. Means the contaminated ensemble already accounts for the shift.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustworthy_sdm.io import (
    CellID,
    GridBPaths,
    iter_pilot_cells,
    load_existing_surface,
)

# %% [markdown]
# ## Configuration

# %%
ENTITY = "Austropotamobius torrentium (pooled)"
TRACK = "combined"

REPLICATE_SURFACES_ROOT = Path("./data/replicate_surfaces").resolve()
GRID_B_PATHS = GridBPaths(root=Path("./data/results").resolve())
ALPHA = 0.05  # 95% interval


# %% [markdown]
# ## Load contaminated ensembles (30 replicates per cell)

# %%
def load_ensemble(cell: CellID, root: Path) -> pd.DataFrame:
    """Load all available replicate surfaces for one cell as a wide DataFrame
    indexed by subc_id with one column per replicate."""
    cell_dir = root / cell.short()
    if not cell_dir.exists():
        raise FileNotFoundError(f"missing replicate-surface dir: {cell_dir}")

    frames = []
    for rep in range(30):
        path = cell_dir / f"rep_{rep:02d}.parquet"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        df = pd.read_parquet(path).set_index("subc_id")["predicted_probability"]
        frames.append(df.rename(f"rep_{rep:02d}"))

    if not frames:
        raise RuntimeError(f"no replicate surfaces loaded from {cell_dir}")
    ensemble = pd.concat(frames, axis=1)
    print(f"  loaded {ensemble.shape[1]}/30 replicates for {cell.short()}")
    return ensemble


cells = list(iter_pilot_cells())
ensembles: dict[str, pd.DataFrame] = {}
for cell in cells:
    key = f"{cell.algorithm}_L{cell.level}"
    ensembles[key] = load_ensemble(cell, REPLICATE_SURFACES_ROOT)


# %% [markdown]
# ## Load benchmark (deterministic) from disk
#
# Path 2: the companion paper's saved benchmark surface is a single
# deterministic prediction. One column per cell, aligned by subc_id.

# %%
def load_benchmark_for(algorithm: str) -> pd.Series:
    cell = CellID(ENTITY, algorithm, TRACK, axis="benchmark", level=0)
    df = load_existing_surface(GRID_B_PATHS, cell, kind="benchmark")
    return df.set_index("subc_id")["predicted_probability"].rename(f"{algorithm}_bench")


bench_rf = load_benchmark_for("random_forest")
bench_xgb = load_benchmark_for("xgboost")
print(f"benchmark RF: {len(bench_rf)} segments, range [{bench_rf.min():.3f}, {bench_rf.max():.3f}]")
print(f"benchmark XGB: {len(bench_xgb)} segments, range [{bench_xgb.min():.3f}, {bench_xgb.max():.3f}]")


# %% [markdown]
# ## Pixel-wise ensemble statistics for each contaminated cell

# %%
def ensemble_stats(ens: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean": ens.mean(axis=1),
        "sd": ens.std(axis=1, ddof=1),
        "lo95": ens.quantile(ALPHA / 2, axis=1),
        "hi95": ens.quantile(1 - ALPHA / 2, axis=1),
    })


stats = {key: ensemble_stats(ens) for key, ens in ensembles.items()}
for key, s in stats.items():
    print(f"{key}: mean range [{s['mean'].min():.3f}, {s['mean'].max():.3f}], "
          f"median width = {(s['hi95'] - s['lo95']).median():.3f}")


# %% [markdown]
# ## Coverage analysis: does the contaminated ensemble's 95% interval contain the benchmark?

# %%
def coverage(target: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    """Fraction of `target` values that fall inside [lo, hi] across shared subc_ids."""
    aligned = pd.concat({"t": target, "lo": lo, "hi": hi}, axis=1).dropna()
    inside = (aligned["t"] >= aligned["lo"]) & (aligned["t"] <= aligned["hi"])
    return float(inside.mean())


print("=== empirical coverage of 95% contaminated-ensemble interval vs deterministic benchmark ===")
rows = []
for cell in cells:
    bench = bench_rf if cell.algorithm == "random_forest" else bench_xgb
    s = stats[f"{cell.algorithm}_L{cell.level}"]
    cov = coverage(bench, s["lo95"], s["hi95"])
    width = (s["hi95"] - s["lo95"]).median()
    rows.append({
        "algorithm": cell.algorithm,
        "level": cell.level,
        "coverage": cov,
        "median_width": width,
    })

cov_df = pd.DataFrame(rows).sort_values(["algorithm", "level"])
print(cov_df.to_string(index=False))

# %% [markdown]
# ### Reading the table
#
# - **Coverage near 0.95 at all levels** would mean the contaminated ensemble
#   correctly knows it disagrees with the benchmark. Headline negative result
#   is dead.
# - **Coverage falling with level** (e.g. 0.85 at L3 → 0.50 at L20) is the
#   headline confirmation: the ensemble underestimates its own error.
# - **Coverage flat and high but median_width also rising sharply** would
#   indicate calibrated-but-conservative — also interesting, different paper.

# %% [markdown]
# ## Figure 1 (draft): coverage curve

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, algo in zip(axes, ["random_forest", "xgboost"], strict=True):
    sub = cov_df[cov_df["algorithm"] == algo]
    ax.plot(sub["level"], sub["coverage"], "o-", linewidth=2, markersize=8)
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=1, label="nominal 0.95")
    ax.set_xlabel("contamination level (% lowacc)")
    ax.set_title("Random Forest" if algo == "random_forest" else "XGBoost")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
axes[0].set_ylabel("empirical coverage")
fig.suptitle(
    "Pilot F1: contaminated-ensemble 95% interval coverage of deterministic benchmark\n"
    "A. torrentium combined — 30 replicates per cell"
)
fig.tight_layout()
Path("./figures").mkdir(exist_ok=True)
fig.savefig("./figures/01_pilot_coverage_curve.png", dpi=150)
plt.show()


# %% [markdown]
# ## Figure 2 (draft): width curve (the diagnostic signal)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, algo in zip(axes, ["random_forest", "xgboost"], strict=True):
    sub = cov_df[cov_df["algorithm"] == algo]
    ax.plot(sub["level"], sub["median_width"], "o-", linewidth=2, markersize=8)
    ax.set_xlabel("contamination level (% lowacc)")
    ax.set_title("Random Forest" if algo == "random_forest" else "XGBoost")
axes[0].set_ylabel("median 95% interval width")
fig.suptitle(
    "Pilot F2: ensemble interval width by contamination level\n"
    "Rising width = self-aware ensemble; flat width = the diagnostic problem"
)
fig.tight_layout()
fig.savefig("./figures/02_pilot_width_curve.png", dpi=150)
plt.show()
