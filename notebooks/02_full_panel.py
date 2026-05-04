# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
# %% [markdown]
# # Full panel: Paper A figures
#
# Loads all 144 ensembles (8 entities × 2 algorithms × 3 tracks × 3 lowacc levels),
# computes coverage and width against the companion paper's deterministic
# benchmark, and produces the five Paper A figures plus an asymmetry analysis
# motivated by Lucian's review of the pilot.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trustworthy_sdm.analysis import (
    analyse_panel,
    asymmetry_panel,
    benchmark_n_per_entity,
)
from trustworthy_sdm.io import (
    DUAL_AXIS_ENTITIES,
    GridBPaths,
    load_merged_metrics,
)

SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
GRID_B_PATHS = GridBPaths(root=Path("data/results").resolve())
FIG_DIR = Path("figures").resolve()
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("notebook")


# %% [markdown]
# ## Load and analyse — ~1-2 min

# %%
print("running panel analysis...")
panel = analyse_panel(SURFACES_ROOT, GRID_B_PATHS)
print(f"panel rows: {len(panel)} (expected 144)")
panel.to_csv(FIG_DIR / "panel_summary.csv", index=False)
print(panel.head())


# %% [markdown]
# ## Coverage table grouped by algorithm and level (headline numbers)

# %%
agg = (
    panel.groupby(["algorithm", "level"])
    .agg(
        coverage_mean=("coverage", "mean"),
        coverage_min=("coverage", "min"),
        coverage_max=("coverage", "max"),
        median_width_mean=("median_width", "mean"),
    )
    .round(3)
)
print("=== coverage and width summary by (algorithm, level) ===")
print(agg)


# %% [markdown]
# ## F1 — coverage curve, faceted by (algorithm, track), one line per entity

# %%
def short_entity(name: str) -> str:
    """Compact label for legends — 'A. torrentium (p)' from the canonical name."""
    base = name.split(" (")[0]
    parts = base.split()
    if len(parts) >= 2:
        s = f"{parts[0][0]}. {parts[1]}"
    else:
        s = base
    if "(" in name:
        suffix = name[name.index("(") + 1 : name.rindex(")")][0]
        s = f"{s} ({suffix})"
    return s


panel["entity_short"] = panel["entity"].map(short_entity)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
algos = ["random_forest", "xgboost"]
tracks = ["local_only", "upstream_only", "combined"]
palette = sns.color_palette("tab10", n_colors=len(DUAL_AXIS_ENTITIES))

for i, algo in enumerate(algos):
    for j, track in enumerate(tracks):
        ax = axes[i, j]
        sub = panel[(panel.algorithm == algo) & (panel.track == track)]
        for k, entity in enumerate(DUAL_AXIS_ENTITIES):
            line = sub[sub.entity == entity].sort_values("level")
            if len(line) == 0:
                continue
            ax.plot(line.level, line.coverage, "o-", color=palette[k],
                    linewidth=1.5, markersize=5, label=short_entity(entity),
                    alpha=0.85)
        ax.axhline(0.95, color="grey", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{algo.replace('_', ' ').title()} — {track.replace('_', ' ')}",
                     fontsize=10)
        if i == 1:
            ax.set_xlabel("contamination level (% lowacc)")
        if j == 0:
            ax.set_ylabel("empirical coverage")

axes[0, 2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                  frameon=False)
fig.suptitle("F1: 95% interval coverage of contaminated ensemble vs deterministic benchmark",
             fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "F1_coverage_curve.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'F1_coverage_curve.png'}")


# %% [markdown]
# ## F2 — width curve, same faceting

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=False)
for i, algo in enumerate(algos):
    for j, track in enumerate(tracks):
        ax = axes[i, j]
        sub = panel[(panel.algorithm == algo) & (panel.track == track)]
        for k, entity in enumerate(DUAL_AXIS_ENTITIES):
            line = sub[sub.entity == entity].sort_values("level")
            if len(line) == 0:
                continue
            ax.plot(line.level, line.median_width, "o-", color=palette[k],
                    linewidth=1.5, markersize=5, alpha=0.85)
        ax.set_title(f"{algo.replace('_', ' ').title()} — {track.replace('_', ' ')}",
                     fontsize=10)
        if i == 1:
            ax.set_xlabel("contamination level (% lowacc)")
        if j == 0:
            ax.set_ylabel("median 95% interval width")
fig.suptitle("F2: ensemble interval width by contamination level — diagnostic signal",
             fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "F2_width_curve.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'F2_width_curve.png'}")


# %% [markdown]
# ## F3 — algorithm contrast at L20 (Lucian's elevation of XGB overconfidence)

# %%
contrast = (
    panel[panel.level == 20]
    .groupby(["entity", "algorithm"])["coverage"]
    .mean()
    .unstack("algorithm")
    .reset_index()
)
contrast["entity_short"] = contrast["entity"].map(short_entity)
contrast = contrast.sort_values("random_forest").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(contrast))
ax.bar(x - 0.2, contrast["random_forest"], 0.4, label="Random Forest",
       color=sns.color_palette("Set2")[0])
ax.bar(x + 0.2, contrast["xgboost"], 0.4, label="XGBoost",
       color=sns.color_palette("Set2")[1])
ax.axhline(0.95, color="grey", linestyle="--", linewidth=1, label="nominal 0.95")
ax.set_xticks(x)
ax.set_xticklabels(contrast["entity_short"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("coverage at L20 (averaged over tracks)")
ax.set_ylim(0, 1.02)
ax.set_title("F3: XGBoost more overconfident than RF across entities at L20")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "F3_algorithm_contrast.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'F3_algorithm_contrast.png'}")


# %% [markdown]
# ## F4 — asymmetric mismatch by benchmark suitability decile (Lucian's point 2)
#
# At L=10, bin pixels by benchmark suitability (deciles) and ask: what fraction
# of contaminated-ensemble means are ABOVE the benchmark in each bin?
# A flat 0.5 line means symmetric; >0.5 in low-suitability bins is the
# spatial-bias signature.

# %%
print("running asymmetry analysis at L10 (this also takes a minute)...")
asym = asymmetry_panel(SURFACES_ROOT, GRID_B_PATHS, level=10)
asym.to_csv(FIG_DIR / "asymmetry_L10.csv", index=False)

if len(asym) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, algo in zip(axes, algos, strict=True):
        sub = asym[asym.algorithm == algo]
        for k, entity in enumerate(DUAL_AXIS_ENTITIES):
            row = sub[sub.entity == entity]
            if len(row) == 0:
                continue
            agg_e = row.groupby("bench_decile_mid", as_index=False)["frac_over_predicted"].mean()
            ax.plot(agg_e["bench_decile_mid"], agg_e["frac_over_predicted"],
                    "o-", color=palette[k], alpha=0.7, markersize=4,
                    label=short_entity(entity) if algo == "xgboost" else None)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("benchmark suitability (decile midpoint)")
        ax.set_title(algo.replace("_", " ").title())
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of pixels with ensemble_mean > benchmark")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                   frameon=False)
    fig.suptitle(
        "F4: asymmetric over-prediction at low-suitability pixels (L=10, averaged across tracks)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F4_asymmetry_by_decile.png", dpi=150, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'F4_asymmetry_by_decile.png'}")
else:
    print("WARNING: asymmetry panel was empty")


# %% [markdown]
# ## F5 — false-alarm panel: width-vs-n at L3 (proxy for L0)
#
# Lucian's request: width-vs-n at level 0 isolates the niche-breadth/sample-size
# confound. We don't have L0 ensembles in Path 2 (companion paper saved a
# deterministic benchmark, not 30 replicates). Use L3 width as a proxy. Caveat
# stated in caption.

# %%
metrics = load_merged_metrics(GRID_B_PATHS)
n_per_entity = benchmark_n_per_entity(metrics)
print(f"benchmark_n per entity: {n_per_entity}")

# Width at L3 averaged over algorithm and track per entity
wn = (
    panel[panel.level == 3]
    .groupby("entity", as_index=False)["median_width"]
    .mean()
)
wn["n"] = wn["entity"].map(n_per_entity)
wn["entity_short"] = wn["entity"].map(short_entity)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(wn["n"], wn["median_width"], s=70, color=sns.color_palette("Set2")[2])
for _, r in wn.iterrows():
    ax.annotate(r["entity_short"], (r["n"], r["median_width"]),
                xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("benchmark presence count (n)")
ax.set_ylabel("median 95% interval width at L3 (averaged over algorithm and track)")
ax.set_title("F5: width-vs-n at L3 (proxy for L0; see caveat)")
fig.tight_layout()
fig.savefig(FIG_DIR / "F5_width_vs_n.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'F5_width_vs_n.png'}")
print("  CAVEAT: L3 used as proxy for L0 because Path 2 has no benchmark ensemble.")


# %% [markdown]
# ## Final summary printed for the chat

# %%
print("\n=== SUMMARY ===")
print(f"panel rows analysed: {len(panel)}")
print(f"  entities: {panel.entity.nunique()}")
print(f"  cells per entity: {len(panel) // panel.entity.nunique()}")
print()
print("=== headline coverage by (algorithm, level) ===")
print(panel.groupby(["algorithm", "level"])["coverage"].agg(["mean", "min", "max"]).round(3))
print()
print("=== headline width by (algorithm, level) ===")
print(panel.groupby(["algorithm", "level"])["median_width"].agg(["mean", "min", "max"]).round(3))
