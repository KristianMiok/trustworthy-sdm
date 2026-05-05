# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
# %% [markdown]
# # Paper A — conformal calibration result
#
# The positive result. We documented in F1-F5 that ensemble variance is
# miscalibrated under contamination. Here we apply leave-one-basin-out (LOBO)
# split conformal calibration and show that it restores near-nominal
# coverage across the panel, at the cost of widened intervals.
#
# **Method.** For each cell (entity × algorithm × track × contamination
# level), partition pixels by basin_id. For each basin held out as test,
# compute non-conformity scores ``s_i = max(lo_i - bench_i, bench_i - hi_i, 0)``
# on the remaining basins (calibration), take the (1-α) finite-sample
# quantile q_hat, and apply ``[lo - q_hat, hi + q_hat]`` to test pixels.
# Coverage is computed against the deterministic benchmark surface
# (companion paper).
#
# **Three figures:**
# - F6: corrected vs uncorrected coverage scatter — the punch.
# - F7: corrected coverage curve, faceted like F1 — flat near 0.95.
# - F8: width inflation vs miscalibration severity — the cost.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trustworthy_sdm.analysis import evaluate_panel_conformal
from trustworthy_sdm.io import DUAL_AXIS_ENTITIES, GridBPaths

SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
GRID_B_PATHS = GridBPaths(root=Path("data/results").resolve())
FIG_DIR = Path("figures").resolve()
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("notebook")

# Sanity-check master CSV is set
master = os.environ.get("TS_MASTER_CSV")
if not master:
    print("WARNING: TS_MASTER_CSV not set; basin_id lookup will fail")
else:
    print(f"using master CSV: {master}")


# %% [markdown]
# ## Run conformal over the full panel (~2-3 min)
#
# Master CSV loads once and gets cached. Each cell is then ~1-2 seconds.

# %%
print("running conformal panel evaluation...")
panel_conformal = evaluate_panel_conformal(SURFACES_ROOT, GRID_B_PATHS)
print(f"completed cells: {len(panel_conformal)} / 144")
panel_conformal.to_csv(FIG_DIR / "panel_conformal.csv", index=False)


# %% [markdown]
# ## Headline numbers

# %%
agg = (
    panel_conformal.groupby(["algorithm", "level"])
    .agg(
        cov_unc_mean=("coverage_uncorrected", "mean"),
        cov_corr_mean=("coverage_conformal", "mean"),
        cov_corr_min=("coverage_conformal", "min"),
        cov_corr_max=("coverage_conformal", "max"),
        width_inflation=("width_inflation_factor", "mean"),
        q_hat_mean=("median_q_hat", "mean"),
    )
    .round(3)
)
print("=== conformal correction summary by (algorithm, level) ===")
print(agg)


# %% [markdown]
# ## F6 — corrected vs uncorrected coverage scatter
#
# The single image that tells the paper's story. X axis: ensemble's empirical
# coverage. Y axis: conformal-corrected coverage. Identity line for reference.
# Dashed nominal at 0.95. If conformal works, every point sits on or near
# y=0.95 regardless of x.

# %%
def short_entity(name: str) -> str:
    base = name.split(" (")[0]
    parts = base.split()
    s = f"{parts[0][0]}. {parts[1]}" if len(parts) >= 2 else base
    if "(" in name:
        suffix = name[name.index("(") + 1 : name.rindex(")")][0]
        s = f"{s} ({suffix})"
    return s


panel_conformal["entity_short"] = panel_conformal["entity"].map(short_entity)

fig, ax = plt.subplots(figsize=(7.5, 6))
algo_markers = {"random_forest": "o", "xgboost": "^"}
palette = sns.color_palette("tab10", n_colors=len(DUAL_AXIS_ENTITIES))
ent_colors = dict(zip(DUAL_AXIS_ENTITIES, palette, strict=True))

for _, row in panel_conformal.iterrows():
    ax.scatter(
        row["coverage_uncorrected"], row["coverage_conformal"],
        marker=algo_markers[row["algorithm"]],
        color=ent_colors[row["entity"]],
        s=55, alpha=0.7, edgecolor="white", linewidth=0.6,
    )

ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1, label="identity")
ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="nominal 0.95")
ax.axvline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.3)

# Legend: entity colors + algorithm markers, separately for clarity
from matplotlib.lines import Line2D
ent_handles = [
    Line2D([0], [0], marker="o", linestyle="", color=ent_colors[e],
           markersize=7, label=short_entity(e))
    for e in DUAL_AXIS_ENTITIES
]
algo_handles = [
    Line2D([0], [0], marker="o", linestyle="", color="grey",
           markersize=7, label="Random Forest"),
    Line2D([0], [0], marker="^", linestyle="", color="grey",
           markersize=7, label="XGBoost"),
]
leg1 = ax.legend(handles=ent_handles, loc="lower right",
                 title="entity", fontsize=8, title_fontsize=9, frameon=True)
ax.add_artist(leg1)
ax.legend(handles=algo_handles, loc="upper left",
          title="algorithm", fontsize=8, title_fontsize=9, frameon=True)

ax.set_xlim(0.35, 1.02)
ax.set_ylim(0.35, 1.02)
ax.set_xlabel("empirical coverage of contaminated ensemble interval")
ax.set_ylabel("empirical coverage of conformal-corrected interval")
ax.set_title(
    "F6: split conformal calibration restores near-nominal coverage\n"
    "(LOBO basin folds; one point per cell, n=144)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "F6_conformal_scatter.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'F6_conformal_scatter.png'}")


# %% [markdown]
# ## F7 — corrected coverage curve faceted like F1
#
# Direct visual comparison with F1. The contaminated ensemble's monotonic
# decline (F1) becomes a flat near-nominal line under conformal correction.

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
algos = ["random_forest", "xgboost"]
tracks = ["local_only", "upstream_only", "combined"]

for i, algo in enumerate(algos):
    for j, track in enumerate(tracks):
        ax = axes[i, j]
        sub = panel_conformal[(panel_conformal.algorithm == algo)
                              & (panel_conformal.track == track)]
        for k, entity in enumerate(DUAL_AXIS_ENTITIES):
            line = sub[sub.entity == entity].sort_values("level")
            if len(line) == 0:
                continue
            ax.plot(line.level, line.coverage_conformal, "o-",
                    color=palette[k], linewidth=1.5, markersize=5,
                    label=short_entity(entity), alpha=0.85)
        ax.axhline(0.95, color="grey", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.02)
        ax.set_title(
            f"{algo.replace('_', ' ').title()} — {track.replace('_', ' ')}",
            fontsize=10,
        )
        if i == 1:
            ax.set_xlabel("contamination level (% lowacc)")
        if j == 0:
            ax.set_ylabel("empirical coverage (corrected)")

axes[0, 2].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8, frameon=False)
fig.suptitle(
    "F7: conformal-corrected coverage is flat near nominal 0.95 across the panel",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "F7_conformal_coverage_curve.png", dpi=150,
            bbox_inches="tight")
print(f"saved {FIG_DIR / 'F7_conformal_coverage_curve.png'}")


# %% [markdown]
# ## F8 — width inflation cost
#
# What practitioners pay for the calibration. X axis: how miscalibrated the
# uncorrected ensemble was (gap from nominal). Y axis: width inflation
# factor from conformal correction.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for _, row in panel_conformal.iterrows():
    ax.scatter(
        row["coverage_gap_pre"], row["width_inflation_factor"],
        marker=algo_markers[row["algorithm"]],
        color=ent_colors[row["entity"]],
        s=55, alpha=0.7, edgecolor="white", linewidth=0.6,
    )

ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("uncorrected coverage gap from nominal (0.95 - cov_unc)")
ax.set_ylabel("width inflation factor (corrected / uncorrected)")
ax.set_title(
    "F8: cost of conformal correction scales with miscalibration severity\n"
    "(width inflation factor; markers = algorithm; colors = entity)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "F8_width_inflation.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'F8_width_inflation.png'}")


# %% [markdown]
# ## Final summary

# %%
print("\n=== summary ===")
print(f"panel cells with conformal evaluation: {len(panel_conformal)} / 144")
print()
print("=== overall corrected coverage ===")
print(panel_conformal["coverage_conformal"].describe().round(3))
print()
print("=== algorithm contrast ===")
print(panel_conformal.groupby("algorithm")["coverage_conformal"].agg(
    ["mean", "min", "max"]).round(3))
print()
print("=== width inflation factor ===")
print(panel_conformal["width_inflation_factor"].describe().round(3))
