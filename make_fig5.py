#!/usr/bin/env python
"""
make_fig5.py -- final two-panel Fig 5 for the P. leniusculus worked case.

(a) contamination-induced divergence surface (prediction - clean benchmark) on
    the Sava-upper-Danube corridor. Diverging palette centered at 0, both signs
    visible (over-prediction warm = marginal-habitat inflation; under-prediction
    cool = core-habitat deflation -- not hidden). Coordinates from the master's
    network-snapped points, with a PURITY FILTER: Hydrography90m subc_ids are not
    globally unique, and this P. leniusculus domain is global (native NW America,
    invaded Europe and East Asia), so a few IDs carry cross-continent coordinates;
    we drop those and clip to the corridor.
(b) directional miscalibration by benchmark-suitability band with 30-replicate
    intervals -- the threshold-free dose-response.

Run from repo root:  python make_fig5.py [--level 20]
Writes figures/fig5.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from run_overpred_ci import analyse_level_ci, LEVELS, BANDS

MASTER = Path("/Users/kristianmiok/Desktop/Papers/sdm-robustness/data/raw/"
              "combined_data_true_master.csv")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

CORRIDOR = (13.0, 17.5, 45.3, 49.2)   # Sava-upper-Danube bbox -- tune with Lucian
EPS = 0.1                             # deg; per-subc_id coord spread above this = collision


def pure_coords(keep: set[int]) -> pd.DataFrame:
    m = pd.read_csv(MASTER, usecols=["subc_id", "lat_snap", "long_snap"]).dropna()
    m = m[m.subc_id.isin(keep)]
    g = m.groupby("subc_id").agg(lat=("lat_snap", "median"), lon=("long_snap", "median"),
                                 lat_std=("lat_snap", "std"), lon_std=("long_snap", "std"))
    pure = (g.lat_std.isna() | (g.lat_std < EPS)) & (g.lon_std.isna() | (g.lon_std < EPS))
    print(f"coordinate purity: kept {int(pure.sum())}, "
          f"dropped {int((~pure).sum())} cross-region-collision subc_ids")
    return g.loc[pure, ["lat", "lon"]]


def panel_a(ax, level: int):
    df = pd.read_csv(OUT / f"overpred_pleniusculus_L{level}.csv")
    d = df.merge(pure_coords(set(df.cell_id)), left_on="cell_id", right_index=True, how="inner")
    lo_lon, hi_lon, lo_lat, hi_lat = CORRIDOR
    d = d[d.lon.between(lo_lon, hi_lon) & d.lat.between(lo_lat, hi_lat)]
    print(f"panel (a): {len(d)} corridor cells at L{level}")

    vmax = float(np.nanpercentile(np.abs(d["div"]), 98)) or 0.1
    # The divergence is strongly right-skewed (over-prediction dominates; only a
    # few % of cells are negative). A symmetric palette wastes half its range on
    # empty blue and buries the small core-deflation signal. Use an ASYMMETRIC
    # diverging norm keyed to the actual distribution, with 0 pinned white so the
    # over/under boundary stays truthful.
    v = d["div"].to_numpy()
    vmin = float(np.nanpercentile(v, 1))                        # real negative reach
    vmax = float(np.nanpercentile(v, 95))                       # avoid the extreme + tail flattening everything
    vmin = min(vmin, -1e-3); vmax = max(vmax, 1e-3)             # guard: keep 0 interior
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    d = d.reindex(d["div"].abs().sort_values().index)          # strong divergence on top
    sc = ax.scatter(d.lon, d.lat, c=d["div"], cmap="RdBu_r", norm=norm,
                    s=8, linewidths=0.2, edgecolors="0.5", alpha=0.9)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(d.lat.mean()))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("(a)", fontsize=12, loc="left", fontweight="bold", pad=10)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, extend="both")
    cb.set_label("pred \u2212 benchmark  (>0 over, <0 under)", fontsize=8)


def panel_b(ax):
    report = {f"L{lv}": analyse_level_ci(lv) for lv in LEVELS}
    bands = list(next(iter(report.values())).keys())
    mids = {f"[{lo}, {hi})": (lo + min(hi, 1.0)) / 2.0
            for lo, hi in zip(BANDS[:-1], BANDS[1:])}
    cmap = matplotlib.colormaps["RdBu"]
    x = np.arange(len(LEVELS))
    ax.axhline(0, color="0.5", lw=0.9, ls="--")
    for b in bands:
        m = np.array([report[f"L{lv}"][b]["cellmean_div"]["mean"] for lv in LEVELS])
        lo = np.array([report[f"L{lv}"][b]["cellmean_div"]["lo2.5"] for lv in LEVELS])
        hi = np.array([report[f"L{lv}"][b]["cellmean_div"]["hi97.5"] for lv in LEVELS])
        ax.errorbar(x, m, yerr=np.vstack([m - lo, hi - m]), marker="o", ms=5,
                    lw=1.8, capsize=3, color=cmap(1.0 - mids.get(b, 0.5)),
                    markeredgecolor="0.4", markeredgewidth=0.6, label=b)
    ax.set_xticks(x); ax.set_xticklabels([f"L{lv}\n({lv}%)" for lv in LEVELS])
    ax.set_xlabel("Occurrence-data contamination")
    ax.set_ylabel("Over-prediction  (prediction \u2212 clean benchmark)")
    ax.set_title("(b)", fontsize=12, loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=7.5, title="benchmark band", title_fontsize=8)
    ax.margins(x=0.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=20, choices=(3, 10, 20))
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 5.8), constrained_layout=True)
    panel_a(axA, args.level)
    panel_b(axB)
    # constrained_layout handles the fixed-aspect map + colorbar without clipping;
    # do NOT combine with bbox_inches='tight' (they conflict and re-introduce clipping)
    fig.savefig(OUT / "Fig5.png", dpi=300)   # submission drop-in
    fig.savefig(OUT / "Fig5.pdf", dpi=300)   # vector, if JAE asks later
    print(f"wrote {OUT}/Fig5.png (300 dpi) and {OUT}/Fig5.pdf")


if __name__ == "__main__":
    main()
