#!/usr/bin/env python
"""
make_fig5_panelA.py -- Fig 5 panel (a): contamination-induced divergence surface
for P. leniusculus on the Sava-upper-Danube corridor.

Colored scatter of subcatchment representative points (from the master's
network-snapped coordinates), on a diverging palette centered at zero so BOTH
signs are visible: over-prediction in marginal habitat (warm) and under-
prediction in core habitat (cool), as Lucian requested -- core deflation is not
hidden.

The script prints the FULL coordinate extent before clipping, so you can confirm
or tune the corridor bbox with Lucian.

Run from repo root:
    python make_fig5_panelA.py --level 20            # clipped to corridor
    python make_fig5_panelA.py --level 20 --no-clip  # full domain, to see extent
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

MASTER = Path("/Users/kristianmiok/Desktop/Papers/sdm-robustness/data/raw/"
              "combined_data_true_master.csv")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

# Sava-upper-Danube corridor bbox (lon_min, lon_max, lat_min, lat_max).
# TUNE with Lucian once you see the printed full extent.
CORRIDOR = (13.0, 17.5, 45.3, 49.2)


def coords_by_subc() -> pd.DataFrame:
    """subc_id -> representative (lon, lat) from network-snapped coordinates."""
    m = pd.read_csv(MASTER, usecols=["subc_id", "lat_snap", "long_snap"]).dropna()
    return m.groupby("subc_id").agg(lat=("lat_snap", "median"),
                                    lon=("long_snap", "median"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=20, choices=(3, 10, 20))
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()

    src = OUT / f"overpred_pleniusculus_L{args.level}.csv"
    df = pd.read_csv(src)                      # cell_id, p, p_bench, div, ...
    g = coords_by_subc()
    d = df.merge(g, left_on="cell_id", right_index=True, how="inner")
    print(f"joined {len(d)}/{len(df)} surface cells to coordinates")
    print(f"FULL extent: lon [{d.lon.min():.2f}, {d.lon.max():.2f}]  "
          f"lat [{d.lat.min():.2f}, {d.lat.max():.2f}]")

    if not args.no_clip:
        lo_lon, hi_lon, lo_lat, hi_lat = CORRIDOR
        d = d[d.lon.between(lo_lon, hi_lon) & d.lat.between(lo_lat, hi_lat)]
        print(f"after corridor clip {CORRIDOR}: {len(d)} cells")
        if len(d) == 0:
            print("  !! 0 cells after clip -- adjust CORRIDOR to the printed extent.")
            return

    # diverging norm centered at 0, symmetric robust limits
    vmax = float(np.nanpercentile(np.abs(d["div"]), 98)) or 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    # sort so the strongest divergences plot on top
    d = d.reindex(d["div"].abs().sort_values().index)
    sc = ax.scatter(d["lon"], d["lat"], c=d["div"], cmap="RdBu_r", norm=norm,
                    s=7, linewidths=0, alpha=0.9)

    mean_lat = float(d["lat"].mean())
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))   # equirectangular correction
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Contamination-induced divergence (prediction \u2212 clean benchmark)\n"
                 f"P. leniusculus, L{args.level}, Sava\u2013upper-Danube corridor",
                 fontsize=11)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, extend="both")
    cb.set_label("prediction \u2212 clean benchmark\n(>0 over-prediction, <0 under-prediction)")
    fig.tight_layout()
    tag = "full" if args.no_clip else "corridor"
    fig.savefig(OUT / f"fig5_panelA_divergence_L{args.level}_{tag}.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"fig5_panelA_divergence_L{args.level}_{tag}.png",
                dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/fig5_panelA_divergence_L{args.level}_{tag}.pdf")


if __name__ == "__main__":
    main()
