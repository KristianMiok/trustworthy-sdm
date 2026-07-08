#!/usr/bin/env python
"""
make_fig5_panelB.py -- Fig 5 panel (b): directional miscalibration by benchmark
suitability band, with 30-replicate intervals. Recomputes from the actual data
via run_overpred_ci.analyse_level_ci (no hardcoded numbers).

Run from repo root:  python make_fig5_panelB.py
Writes figures/fig5_panelB_dose_response.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_overpred_ci import analyse_level_ci, LEVELS, BANDS

OUT = Path("figures"); OUT.mkdir(exist_ok=True)


def band_midpoints() -> dict[str, float]:
    mids = {}
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        label = f"[{lo}, {hi})" if hi < 1.0 else f"[{lo}, 1.0)"
        mids[f"[{lo}, {hi})"] = (lo + min(hi, 1.0)) / 2.0
    return mids


def main() -> None:
    report = {f"L{lv}": analyse_level_ci(lv) for lv in LEVELS}
    bands = list(next(iter(report.values())).keys())     # band label strings
    mids = band_midpoints()

    cmap = matplotlib.colormaps["RdBu"]                   # red=low mid, blue=high mid
    x = np.arange(len(LEVELS))

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0, color="0.5", lw=0.9, ls="--", zorder=1)

    for b in bands:
        m = np.array([report[f"L{lv}"][b]["cellmean_div"]["mean"] for lv in LEVELS])
        lo = np.array([report[f"L{lv}"][b]["cellmean_div"]["lo2.5"] for lv in LEVELS])
        hi = np.array([report[f"L{lv}"][b]["cellmean_div"]["hi97.5"] for lv in LEVELS])
        mid = mids.get(b, 0.5)
        ax.errorbar(x, m, yerr=np.vstack([m - lo, hi - m]), marker="o", ms=5,
                    lw=1.8, capsize=3, color=cmap(1.0 - mid),
                    label=f"benchmark {b}", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{lv}\n({lv}%)" for lv in LEVELS])
    ax.set_xlabel("Occurrence-data contamination")
    ax.set_ylabel("Over-prediction  (prediction \u2212 clean benchmark)")
    ax.set_title("Directional miscalibration by suitability band\n"
                 "(cell-mean, 30-replicate 2.5\u201397.5% intervals)", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_panelB_dose_response.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig5_panelB_dose_response.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/fig5_panelB_dose_response.pdf")


if __name__ == "__main__":
    main()
