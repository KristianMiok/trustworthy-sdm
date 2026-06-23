"""Publication-quality Protocol B figures for JRSI submission.

Spec (Lucian): 300 dpi, Arial (fall back to Helvetica/Liberation Sans/DejaVu),
ONE unified species palette across ALL figures, sub-panel labels (a)(b)(c) on
multi-panel figures, tonally subdued gridlines, output as BOTH vector PDF and PNG.

Reads the regenerated mean-benchmark CSVs. Produces:
  Fig 1  = PB-F1  coverage degradation (3 panels by track, 8 entities)
  Fig 2  = PB-F4  directional over-prediction, 5-bin KEYSTONE (3 panels, 8 entities)
  Fig 3  = PB-F7  conformal restoration (3 panels, uncorrected vs conformal)
  Fig 4  = PB-F8  width inflation factor (1 panel, 3 tracks)
  Fig S1 = robustness, 4 full-upstream entities: coverage + asymmetry combined
  Fig S2 = PB-F4 10-bin

Output: figures/pub/Fig{1,2,3,4}.{pdf,png}, FigS{1,2}.{pdf,png}

Run:
    .venv/bin/python notebooks/05_publication_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

FIG = Path("figures")
OUT = FIG / "pub"
OUT.mkdir(exist_ok=True)

# ---- font selection: Arial -> Helvetica -> Liberation Sans -> DejaVu ----
_available = set(f.name for f in font_manager.fontManager.ttflist)
for _cand in ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "Arimo", "DejaVu Sans"]:
    if _cand in _available:
        CHOSEN_FONT = _cand
        break
else:
    CHOSEN_FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [CHOSEN_FONT, "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.grid": True,
    "grid.color": "#d9d9d9",      # subdued
    "grid.linewidth": 0.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
print(f"font in use: {CHOSEN_FONT}")

TRACKS = ["local_only", "upstream_only", "combined"]
TRACK_LABEL = {"local_only": "local-only", "upstream_only": "upstream-only", "combined": "combined"}
LEVELS = [3, 10, 20]

# ---- ONE fixed species->colour mapping, used in EVERY figure ----
import matplotlib.cm as cm
ENTITIES = [
    "Astacus astacus",
    "Austropotamobius fulcisianus (pooled)",
    "Austropotamobius torrentium (pooled)",
    "Faxonius limosus (alien)",
    "Pacifastacus leniusculus (alien)",
    "Pontastacus leptodactylus (pooled)",
    "Procambarus clarkii (alien)",
    "Procambarus clarkii (native)",
]
_tab10 = plt.get_cmap("tab10").colors
SPECIES_COLOUR = {e: _tab10[i] for i, e in enumerate(ENTITIES)}


def short_entity(name: str) -> str:
    base = name.split(" (")[0]
    parts = base.split()
    s = f"{parts[0][0]}. {parts[1]}" if len(parts) >= 2 else base
    if "(" in name:
        s = f"{s} ({name[name.index('(') + 1:name.rindex(')')][0]})"
    return s


def panel_label(ax, letter):
    ax.text(-0.08, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf + .png")


def legend_handles(entities):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=SPECIES_COLOUR[e], marker="o", lw=1.5,
                   markersize=4, label=short_entity(e)) for e in entities]


# =========================================================================
def fig_coverage(summary, entities, stem, suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharex=True, sharey=True)
    for j, track in enumerate(TRACKS):
        ax = axes[j]
        for e in entities:
            line = summary[(summary.track == track) & (summary.entity == e)].sort_values("level")
            if len(line):
                ax.plot(line.level, line.coverage, "o-", color=SPECIES_COLOUR[e],
                        alpha=0.85, markersize=4, linewidth=1.3)
        ax.axhline(0.95, ls="--", c="#222222", lw=0.9)
        ax.set_xlabel("contamination level (% low-accuracy)")
        ax.set_xticks(LEVELS)
        ax.set_title(TRACK_LABEL[track])
        panel_label(ax, "abc"[j])
    axes[0].set_ylabel("empirical coverage")
    axes[2].legend(handles=legend_handles(entities), bbox_to_anchor=(1.02, 1),
                   loc="upper left", frameon=False)
    save(fig, stem)


def fig_asymmetry(asym, entities, n_bins, stem):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)
    for j, track in enumerate(TRACKS):
        ax = axes[j]
        sub_track = asym[asym.track == track]
        for e in entities:
            row = sub_track[sub_track.entity == e]
            if len(row):
                agg = row.groupby("bench_decile_mid", as_index=False)["frac_over_predicted"].mean().sort_values("bench_decile_mid")
                ax.plot(agg.bench_decile_mid, agg.frac_over_predicted, "o-",
                        color=SPECIES_COLOUR[e], alpha=0.45, markersize=3, linewidth=0.9)
        st = sub_track[sub_track.entity.isin(entities)].copy()
        if len(st):
            st["rank"] = st.groupby("entity")["bench_decile_mid"].rank()
            g = st.groupby("rank").agg(x=("bench_decile_mid","mean"), y=("frac_over_predicted","mean")).sort_values("x")
            ax.plot(g.x, g.y, "o-", color="#111111", markersize=5, linewidth=2.2, zorder=10, label="cross-entity mean")
        ax.axhline(0.5, ls="--", c="#888888", lw=0.8)
        ax.set_xlabel(f"benchmark suitability ({n_bins}-bin midpoint)")
        ax.set_title(TRACK_LABEL[track])
        panel_label(ax, "abc"[j])
    axes[0].set_ylabel("fraction over-predicted")
    axes[2].legend(handles=legend_handles(entities), bbox_to_anchor=(1.02, 1),
                   loc="upper left", frameon=False)
    save(fig, stem)


def fig_conformal(conf, stem):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharex=True, sharey=True)
    for j, track in enumerate(TRACKS):
        ax = axes[j]
        sub = conf[conf.track == track]
        agg = sub.groupby("level").agg(unc=("coverage_uncorrected", "mean"),
                                       cor=("coverage_conformal", "mean")).reset_index()
        ax.plot(agg.level, agg.unc, "o-", c="#c0392b", markersize=5, linewidth=1.4, label="uncorrected")
        ax.plot(agg.level, agg.cor, "s-", c="#27ae60", markersize=5, linewidth=1.4, label="conformal")
        ax.axhline(0.95, ls="--", c="#222222", lw=0.9)
        ax.set_xlabel("contamination level (% low-accuracy)")
        ax.set_xticks(LEVELS)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(TRACK_LABEL[track])
        panel_label(ax, "abc"[j])
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(loc="lower left", frameon=False)
    save(fig, stem)


def fig_width(conf, stem):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    agg = conf.groupby(["track", "level"])["width_inflation_factor"].mean().reset_index()
    track_c = {"local_only": "#2c7fb8", "upstream_only": "#d95f0e", "combined": "#756bb1"}
    for track in TRACKS:
        s = agg[agg.track == track]
        ax.plot(s.level, s.width_inflation_factor, "o-", color=track_c[track],
                markersize=5, linewidth=1.4, label=TRACK_LABEL[track])
    ax.axhline(1.0, ls="--", c="#888888", lw=0.8)
    ax.set_xlabel("contamination level (% low-accuracy)")
    ax.set_xticks(LEVELS)
    ax.set_ylabel("width inflation factor (conformal / uncorrected)")
    ax.legend(frameon=False)
    save(fig, stem)


def fig_s1_combined(summary, asym, entities, stem):
    """S1: coverage (top row) + asymmetry (bottom row), 4 full-upstream entities."""
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for j, track in enumerate(TRACKS):
        ax = axes[0, j]
        for e in entities:
            line = summary[(summary.track == track) & (summary.entity == e)].sort_values("level")
            if len(line):
                ax.plot(line.level, line.coverage, "o-", color=SPECIES_COLOUR[e],
                        alpha=0.85, markersize=4, linewidth=1.3)
        ax.axhline(0.95, ls="--", c="#222222", lw=0.9)
        ax.set_xticks(LEVELS); ax.set_title(TRACK_LABEL[track])
        ax.set_xlabel("contamination level (% low-accuracy)")
        panel_label(ax, "abc"[j])
    axes[0, 0].set_ylabel("empirical coverage")
    for j, track in enumerate(TRACKS):
        ax = axes[1, j]
        sub_track = asym[asym.track == track]
        for e in entities:
            row = sub_track[sub_track.entity == e]
            if len(row):
                agg = row.groupby("bench_decile_mid", as_index=False)["frac_over_predicted"].mean().sort_values("bench_decile_mid")
                ax.plot(agg.bench_decile_mid, agg.frac_over_predicted, "o-",
                        color=SPECIES_COLOUR[e], alpha=0.45, markersize=3, linewidth=0.9)
        st = sub_track[sub_track.entity.isin(entities)].copy()
        if len(st):
            st["rank"] = st.groupby("entity")["bench_decile_mid"].rank()
            g = st.groupby("rank").agg(x=("bench_decile_mid","mean"), y=("frac_over_predicted","mean")).sort_values("x")
            ax.plot(g.x, g.y, "o-", color="#111111", markersize=5, linewidth=2.2, zorder=10)
        ax.axhline(0.5, ls="--", c="#888888", lw=0.8)
        ax.set_xlabel("benchmark suitability (5-bin midpoint)")
        panel_label(ax, "def"[j])
    axes[1, 0].set_ylabel("fraction over-predicted")
    axes[0, 2].legend(handles=legend_handles(entities), bbox_to_anchor=(1.02, 1),
                      loc="upper left", frameon=False)
    save(fig, stem)


def main():
    summary = pd.read_csv(FIG / "panel_summary_protocol_b.csv")
    conf = pd.read_csv(FIG / "panel_conformal_protocol_b.csv")
    asym5 = pd.read_csv(FIG / "asymmetry_protocol_b_5bin_L10.csv")
    asym10 = pd.read_csv(FIG / "asymmetry_protocol_b_10bin_L10.csv")

    all_ents = [e for e in ENTITIES if e in set(summary.entity)]
    full_upstream = [
        "Pacifastacus leniusculus (alien)", "Faxonius limosus (alien)",
        "Astacus astacus", "Austropotamobius torrentium (pooled)",
    ]

    fig_coverage(summary, all_ents, "Fig1_coverage")
    fig_asymmetry(asym5, all_ents, 5, "Fig2_asymmetry_keystone")
    fig_conformal(conf, "Fig3_conformal_restoration")
    fig_width(conf, "Fig4_width_inflation")
    fig_s1_combined(summary, asym5, full_upstream, "FigS1_robustness")
    fig_asymmetry(asym10, all_ents, 10, "FigS2_asymmetry_10bin")
    print(f"\nAll figures in {OUT}/ (PDF + PNG, 300 dpi, font={CHOSEN_FONT})")


if __name__ == "__main__":
    main()
