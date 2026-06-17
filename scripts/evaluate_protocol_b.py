"""Protocol B coverage + conformal evaluation.

Replicates analyse_cell and evaluate_cell_conformal EXACTLY, with one change:
the clean benchmark reference is the consensus benchmark ensemble
(median across the 4 algorithms on uncontaminated data), loaded from
{entity}__consensus__{track}__benchmark__L0/, instead of the rf/xgboost-only
benchmark_for() path.

All conformal math reuses the public helpers from trustworthy_sdm.conformal
(nonconformity_scores, conformal_quantile), so Protocol B's panel_conformal is
computed identically to Protocol A's. Shared code is untouched.

Output schemas match Protocol A's panel_summary.csv and panel_conformal.csv,
with algorithm='consensus'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustworthy_sdm.analysis import (  # noqa: E402
    ENTITY_NAME_TO_DIR, DUAL_AXIS_ENTITIES, TRACKS, FULL_PANEL_LOWACC_LEVELS,
)
from trustworthy_sdm.conformal import (  # noqa: E402
    CellID, GridBPaths, load_ensemble, ALPHA,
    nonconformity_scores, conformal_quantile,
)

AXIS = "lowacc"


def consensus_benchmark(entity: str, track: str, surfaces_root: Path) -> pd.Series:
    """Clean consensus reference: per-pixel median across the 4 algorithm
    surfaces in the consensus benchmark__L0 ensemble dir."""
    cell = CellID(entity, "consensus", track, axis="benchmark", level=0)
    ens = load_ensemble(cell, surfaces_root)  # wide: subc_id x 4 algos
    return ens.median(axis=1)


def coverage_row(entity, track, level, surfaces_root):
    cell = CellID(entity, "consensus", track, axis=AXIS, level=level)
    ens = load_ensemble(cell, surfaces_root)
    bench = consensus_benchmark(entity, track, surfaces_root)
    shared = ens.index.intersection(bench.index)
    ens_a = ens.loc[shared]; bench_a = bench.loc[shared]
    lo = ens_a.quantile(ALPHA / 2, axis=1)
    hi = ens_a.quantile(1 - ALPHA / 2, axis=1)
    width = hi - lo
    inside = (bench_a >= lo) & (bench_a <= hi)
    ens_mean = ens_a.mean(axis=1)
    diff = bench_a - ens_mean
    return {
        "entity": entity, "entity_dir": ENTITY_NAME_TO_DIR[entity],
        "algorithm": "consensus", "track": track, "level": level,
        "n_replicates": ens.shape[1], "n_pixels": len(shared),
        "coverage": float(inside.mean()),
        "median_width": float(width.median()), "mean_width": float(width.mean()),
        "median_signed_diff": float(diff.median()),
        "mean_abs_diff": float(diff.abs().mean()),
        "frac_diff_gt_0p1": float((diff.abs() > 0.1).mean()),
    }


def conformal_row(entity, track, level, surfaces_root, basin_lookup, alpha=ALPHA):
    cell = CellID(entity, "consensus", track, axis=AXIS, level=level)
    ens = load_ensemble(cell, surfaces_root)
    bench = consensus_benchmark(entity, track, surfaces_root)
    basins = basin_lookup(entity)
    shared = ens.index.intersection(bench.index).intersection(basins.index)
    if len(shared) == 0:
        raise RuntimeError(f"no shared subc_ids for {cell.short()}")
    ens_a = ens.loc[shared]; bench_a = bench.loc[shared]; basin_a = basins.loc[shared]
    lo = ens_a.quantile(alpha / 2, axis=1)
    hi = ens_a.quantile(1 - alpha / 2, axis=1)
    unique_basins = basin_a.dropna().unique()
    if len(unique_basins) < 2:
        raise RuntimeError(f"{cell.short()} has <2 basins")
    fold_results = []
    for test_basin in unique_basins:
        is_test = (basin_a == test_basin); cal_mask = ~is_test
        cal_scores = nonconformity_scores(bench_a[cal_mask], lo[cal_mask], hi[cal_mask])
        if len(cal_scores) == 0:
            continue
        q_hat = conformal_quantile(cal_scores, alpha=alpha)
        lo_c = lo[is_test] - q_hat; hi_c = hi[is_test] + q_hat
        bt = bench_a[is_test]
        inside_unc = (bt >= lo[is_test]) & (bt <= hi[is_test])
        inside_corr = (bt >= lo_c) & (bt <= hi_c)
        fold_results.append({
            "basin": test_basin, "n_test": int(is_test.sum()), "q_hat": q_hat,
            "cov_unc": float(inside_unc.mean()), "cov_corr": float(inside_corr.mean()),
            "width_unc": float((hi[is_test] - lo[is_test]).median()),
            "width_corr": float((hi_c - lo_c).median()),
        })
    fold_df = pd.DataFrame(fold_results)
    if len(fold_df) == 0:
        raise RuntimeError(f"no folds for {cell.short()}")
    w = fold_df["n_test"]
    cov_unc = float((fold_df["cov_unc"] * w).sum() / w.sum())
    cov_corr = float((fold_df["cov_corr"] * w).sum() / w.sum())
    width_unc = float(np.average(fold_df["width_unc"], weights=w))
    width_corr = float(np.average(fold_df["width_corr"], weights=w))
    cov_gap_pre = abs(0.95 - cov_unc)
    cov_gap_post = abs(0.95 - cov_corr)
    wif = width_corr / width_unc if width_unc > 0 else float("nan")
    return {
        "entity": entity, "algorithm": "consensus", "track": track, "level": level,
        "n_pixels_total": int(len(shared)), "n_basins": int(len(unique_basins)),
        "coverage_uncorrected": cov_unc, "coverage_conformal": cov_corr,
        "median_width_uncorrected": width_unc, "median_width_conformal": width_corr,
        "median_q_hat": float(fold_df["q_hat"].median()),
        "coverage_gap_pre": cov_gap_pre, "coverage_gap_post": cov_gap_post,
        "width_inflation_factor": wif,
    }


def main():
    surfaces_root = REPO_ROOT / "data" / "replicate_surfaces_protocol_b"

    # Reuse the upstream basin lookup that evaluate_cell_conformal uses.
    from trustworthy_sdm.conformal import _basin_id_lookup as basin_lookup

    cov_rows, conf_rows = [], []
    for entity in DUAL_AXIS_ENTITIES:
        for track in TRACKS:
            for level in FULL_PANEL_LOWACC_LEVELS:
                try:
                    cov_rows.append(coverage_row(entity, track, level, surfaces_root))
                except Exception as e:  # noqa: BLE001
                    print(f"coverage FAIL {entity}|{track}|L{level}: {e}")
                try:
                    conf_rows.append(conformal_row(entity, track, level, surfaces_root, basin_lookup))
                except Exception as e:  # noqa: BLE001
                    print(f"conformal FAIL {entity}|{track}|L{level}: {e}")

    cov = pd.DataFrame(cov_rows)
    conf = pd.DataFrame(conf_rows)
    out = REPO_ROOT / "figures"
    out.mkdir(exist_ok=True)
    cov.to_csv(out / "panel_summary_protocol_b.csv", index=False)
    conf.to_csv(out / "panel_conformal_protocol_b.csv", index=False)
    print(f"\nwrote {len(cov)} coverage rows, {len(conf)} conformal rows")
    print("\n=== COVERAGE (uncorrected, by level) ===")
    if len(cov):
        piv = cov.pivot_table(index=["entity", "track"], columns="level", values="coverage")
        print(piv.round(3).to_string())
    print("\n=== CONFORMAL (uncorrected -> corrected coverage) ===")
    if len(conf):
        for _, r in conf.iterrows():
            print(f"  {r['entity'][:22]:22s} {r['track']:13s} L{int(r['level']):2d}: "
                  f"{r['coverage_uncorrected']:.3f} -> {r['coverage_conformal']:.3f}  "
                  f"(WIF {r['width_inflation_factor']:.2f})")


if __name__ == "__main__":
    main()
