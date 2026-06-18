"""Protocol B directional over-prediction (keystone Fig 6 analogue).

Replicates asymmetry_by_benchmark_decile EXACTLY, sourcing the clean reference
from the consensus benchmark ensemble (median across the 4 algorithms) instead
of the rf/xgboost-only benchmark_for() path.

For each cell: bin pixels by clean-benchmark suitability, and per bin compute
the fraction of contaminated-consensus means ABOVE the benchmark (over-prediction).
Produces headline 5-bin and supplementary 10-bin tables, entity-by-entity
(no cross-entity averaging — the overlay IS the replication argument).

Output: figures/asymmetry_protocol_b_5bin.csv, figures/asymmetry_protocol_b_10bin.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustworthy_sdm.analysis import DUAL_AXIS_ENTITIES, TRACKS  # noqa: E402
from trustworthy_sdm.conformal import CellID, load_ensemble  # noqa: E402

AXIS = "lowacc"


def consensus_benchmark(entity: str, track: str, surfaces_root: Path) -> pd.Series:
    cell = CellID(entity, "consensus", track, axis="benchmark", level=0)
    return load_ensemble(cell, surfaces_root).median(axis=1)


def asymmetry_by_benchmark_decile_consensus(entity, track, level, surfaces_root, n_bins):
    """Exact replica of asymmetry_by_benchmark_decile, consensus benchmark."""
    cell = CellID(entity, "consensus", track, axis=AXIS, level=level)
    ens = load_ensemble(cell, surfaces_root)
    bench = consensus_benchmark(entity, track, surfaces_root)
    shared = ens.index.intersection(bench.index)
    ens_mean = ens.loc[shared].mean(axis=1)
    bench_a = bench.loc[shared]
    bins = pd.qcut(bench_a, q=n_bins, duplicates="drop")
    over = (ens_mean > bench_a).groupby(bins, observed=True).mean()
    median_diff = (ens_mean - bench_a).groupby(bins, observed=True).median()
    bin_centres = pd.Series(bins.cat.categories.map(lambda iv: iv.mid),
                            index=over.index, name="bench_mid")
    return pd.DataFrame({
        "entity": entity, "track": track, "level": level,
        "bench_decile_mid": bin_centres.values,
        "frac_over_predicted": over.values,
        "median_diff": median_diff.values,
        "n_pixels": ens_mean.groupby(bins, observed=True).size().values,
    })


def asymmetry_panel_consensus(surfaces_root, level, n_bins):
    rows = []
    for entity in DUAL_AXIS_ENTITIES:
        for track in TRACKS:
            try:
                df = asymmetry_by_benchmark_decile_consensus(
                    entity, track, level, surfaces_root, n_bins)
                rows.append(df)
            except Exception as e:  # noqa: BLE001
                print(f"asymmetry FAIL {entity}|{track}|L{level}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=10)
    args = ap.parse_args()
    surfaces_root = REPO_ROOT / "data" / "replicate_surfaces_protocol_b"

    five = asymmetry_panel_consensus(surfaces_root, args.level, n_bins=5)
    ten = asymmetry_panel_consensus(surfaces_root, args.level, n_bins=10)

    out = REPO_ROOT / "figures"
    out.mkdir(exist_ok=True)
    five.to_csv(out / f"asymmetry_protocol_b_5bin_L{args.level}.csv", index=False)
    ten.to_csv(out / f"asymmetry_protocol_b_10bin_L{args.level}.csv", index=False)
    print(f"wrote 5-bin ({len(five)} rows) and 10-bin ({len(ten)} rows), level={args.level}")

    # Summary: is over-prediction concentrated in LOW benchmark-suitability bins?
    # (the directional signature: contamination inflates suitability where true
    #  suitability is low)
    print("\n=== 5-bin: frac_over_predicted by suitability bin (mean across entities/tracks) ===")
    if len(five):
        # order bins low->high by decile mid, summarise
        five = five.copy()
        five["bin_rank"] = five.groupby(["entity", "track"])["bench_decile_mid"].rank()
        summ = five.groupby("bin_rank").agg(
            mean_bench_mid=("bench_decile_mid", "mean"),
            mean_frac_over=("frac_over_predicted", "mean"),
            mean_median_diff=("median_diff", "mean"),
        )
        print(summ.round(3).to_string())
        print()
        print("Reading: bin_rank 1 = lowest benchmark suitability. If frac_over is")
        print("HIGH in low bins and drops in high bins, that's directional over-prediction:")
        print("contamination inflates predicted suitability where true suitability is low.")


if __name__ == "__main__":
    main()
