"""Protocol B figures — algorithm-consensus ensemble.

Mirrors the Paper A figure style (notebooks/02_full_panel.py): seaborn
whitegrid/notebook, tab10 palette one colour per entity, short_entity legend
labels, dpi=150. Reads the Protocol B panel CSVs produced by
scripts/evaluate_protocol_b.py and scripts/asymmetry_protocol_b.py.

Outputs (figures/):
    PB_F1_coverage_curve.png          coverage vs contamination, per track
    PB_F4_asymmetry_by_decile.png     KEYSTONE: directional over-prediction
    PB_F4_asymmetry_10bin_supp.png    supplementary 10-bin resolution
    PB_F7_conformal_coverage_curve.png  uncorrected vs conformal coverage
    PB_F8_width_inflation.png         width cost of the correction

Run:
    cd ~/Desktop/Papers/trustworthy-sdm
    .venv/bin/python notebooks/04_protocol_b_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401
import pandas as pd
import seaborn as sns

FIG_DIR = Path("figures").resolve()
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("notebook")

TRACKS = ["local_only", "upstream_only", "combined"]
LEVELS = [3, 10, 20]


def short_entity(name: str) -> str:
    base = name.split(" (")[0]
    parts = base.split()
    s = f"{parts[0][0]}. {parts[1]}" if len(parts) >= 2 else base
    if "(" in name:
        s = f"{s} ({name[name.index('(') + 1:name.rindex(')')][0]})"
    return s


summary = pd.read_csv(FIG_DIR / "panel_summary_protocol_b.csv")
conf = pd.read_csv(FIG_DIR / "panel_conformal_protocol_b.csv")
asym5 = pd.read_csv(FIG_DIR / "asymmetry_protocol_b_5bin.csv")
asym10 = pd.read_csv(FIG_DIR / "asymmetry_protocol_b_10bin.csv")

ents = sorted(summary.entity.unique())
palette = sns.color_palette("tab10", n_colors=len(ents))


def plot_asymmetry(asym: pd.DataFrame, n_bins: int, out_name: str, suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for j, track in enumerate(TRACKS):
        ax = axes[j]
        for k, e in enumerate(ents):
            row = asym[(asym.track == track) & (asym.entity == e)]
            if len(row):
                agg = row.groupby("bench_decile_mid", as_index=False)["frac_over_predicted"].mean()
                ax.plot(agg.bench_decile_mid, agg.frac_over_predicted, "o-",
                        color=palette[k], alpha=0.75, markersize=4, label=short_entity(e))
        ax.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax.set_xlabel(f"benchmark suitability ({n_bins}-bin midpoint)")
        ax.set_title(track.replace("_", " "), fontsize=10)
    axes[0].set_ylabel("fraction with consensus mean > benchmark")
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle(
        f"PB-F4{suffix}: directional over-prediction at low-suitability pixels "
        f"(consensus, L10, {n_bins}-bin)", fontsize=11)
    fig.savefig(FIG_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_name}")


# ---- PB_F4 keystone (5-bin) + supplementary (10-bin) ----
plot_asymmetry(asym5, 5, "PB_F4_asymmetry_by_decile.png", "")
plot_asymmetry(asym10, 10, "PB_F4_asymmetry_10bin_supp.png", " (supp.)")

# ---- PB_F1 coverage curve ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
for j, track in enumerate(TRACKS):
    ax = axes[j]
    for k, e in enumerate(ents):
        line = summary[(summary.track == track) & (summary.entity == e)].sort_values("level")
        if len(line):
            ax.plot(line.level, line.coverage, "o-", color=palette[k],
                    alpha=0.75, markersize=5, label=short_entity(e))
    ax.axhline(0.95, ls="--", c="black", lw=1)
    ax.set_xlabel("contamination level (% lowacc)")
    ax.set_xticks(LEVELS)
    ax.set_title(track.replace("_", " "), fontsize=10)
axes[0].set_ylabel("empirical coverage")
axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.suptitle("PB-F1: consensus coverage degrades with contamination", fontsize=11)
fig.savefig(FIG_DIR / "PB_F1_coverage_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote PB_F1_coverage_curve.png")

# ---- PB_F7 conformal restoration ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
for j, track in enumerate(TRACKS):
    ax = axes[j]
    sub = conf[conf.track == track]
    agg = sub.groupby("level").agg(
        unc=("coverage_uncorrected", "mean"),
        cor=("coverage_conformal", "mean")).reset_index()
    ax.plot(agg.level, agg.unc, "o-", c="tab:red", label="uncorrected", markersize=6)
    ax.plot(agg.level, agg.cor, "s-", c="tab:green", label="conformal", markersize=6)
    ax.axhline(0.95, ls="--", c="black", lw=1)
    ax.set_xlabel("contamination level (% lowacc)")
    ax.set_xticks(LEVELS)
    ax.set_title(track.replace("_", " "), fontsize=10)
    ax.set_ylim(0.4, 1.02)
axes[0].set_ylabel("empirical coverage")
axes[0].legend(loc="lower left", fontsize=9)
fig.suptitle("PB-F7: LOBO split conformal restores coverage (consensus)", fontsize=11)
fig.savefig(FIG_DIR / "PB_F7_conformal_coverage_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote PB_F7_conformal_coverage_curve.png")

# ---- PB_F8 width inflation ----
fig, ax = plt.subplots(figsize=(8.5, 5))
agg = conf.groupby(["track", "level"])["width_inflation_factor"].mean().reset_index()
for track in TRACKS:
    s = agg[agg.track == track]
    ax.plot(s.level, s.width_inflation_factor, "o-", label=track.replace("_", " "), markersize=6)
ax.axhline(1.0, ls="--", c="grey", lw=0.8)
ax.set_xlabel("contamination level (% lowacc)")
ax.set_xticks(LEVELS)
ax.set_ylabel("width inflation factor (conformal / uncorrected)")
ax.set_title("PB-F8: interval-width cost of conformal correction")
ax.legend()
fig.savefig(FIG_DIR / "PB_F8_width_inflation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote PB_F8_width_inflation.png")

print("\nAll Protocol B figures written to figures/")
