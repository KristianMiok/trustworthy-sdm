#!/usr/bin/env python
"""
run_flip.py -- worked-case flip analysis for Pacifastacus leniusculus (JAE paper).

Uses the project's OWN conformal machinery (trustworthy_sdm.conformal /
trustworthy_sdm.analysis) so the per-pixel corrected intervals are IDENTICAL in
convention to the panel numbers. Reproduces evaluate_cell_conformal's LOBO loop
but retains per-pixel [lo, hi] and [lo_cor, hi_cor], then applies flip_analysis
to quantify management-decision flips (Flip A withdrawn / Flip B recovered).

Honesty notes baked into what this reports:
  * Coverage/flips are defined against the benchmark MODEL surface, not ground
    truth (M2). Read a flip as "training on contaminated vs clean data changes a
    confident decision", not "matches reality".
  * q_hat is per-basin (LOBO), so the correction magnitude is basin-granular.
    The bridge is reported at BOTH pixel and basin granularity so we can see
    which framing is honest before writing the deployment (M1) paragraph.

Run from the repo root:
    python run_flip.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# make the src-layout package importable when run as a plain script from the
# repo root (PyCharm marks src/ as a source root; a bare terminal does not)
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

# The project's load_ensemble / benchmark_for call pd.read_parquet without an
# engine, and this pyarrow build throws "Repetition level histogram size
# mismatch" on the replicate surfaces. fastparquet reads them fine, so make it
# the process-wide default before any project code runs.
try:
    import fastparquet  # noqa: F401
    pd.options.io.parquet.engine = "fastparquet"
except Exception:  # noqa: BLE001
    pass

from trustworthy_sdm.io import CellID, GridBPaths
from trustworthy_sdm.analysis import ALPHA, load_ensemble, benchmark_for
from trustworthy_sdm.conformal import nonconformity_scores, conformal_quantile
import flip_analysis as fa

# --- config -----------------------------------------------------------------
SURFACES_ROOT = Path("data/replicate_surfaces").resolve()
PATHS = GridBPaths(root=Path("data/results").resolve())
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# subc_id -> basin_id map, built once from the master CSV (see build step).
# Reading it directly keeps the deposit free of the upstream sdm_robustness
# dependency, and is bit-identical to what _basin_id_lookup would have returned.
BASIN_LOOKUP_PATH = Path("data/results/basin_lookup.parquet")


def basin_lookup() -> pd.Series:
    """subc_id -> basin_id as a Series indexed by subc_id."""
    if not BASIN_LOOKUP_PATH.is_file():
        raise FileNotFoundError(
            f"{BASIN_LOOKUP_PATH} missing -- run the one-off build step first "
            "(reads subc_id/basin_id from combined_data_true_master.csv)."
        )
    m = pd.read_parquet(BASIN_LOOKUP_PATH)
    return m.drop_duplicates("subc_id").set_index("subc_id")["basin_id"]

ENTITY = "Pacifastacus leniusculus (alien)"
ALGO = "random_forest"
TRACK = "combined"                 # matches panel rows 78-80
LEVELS = (3, 10, 20)               # dose-response
TAU = 0.5                          # suitability threshold for the decision
TAU_SWEEP = (0.4, 0.5, 0.6)        # sensitivity of the flip counts to tau


def per_pixel_conformal(cell: CellID, alpha: float = ALPHA) -> pd.DataFrame:
    """LOBO conformal exactly as in evaluate_cell_conformal, but keep per-pixel
    point, uncorrected & corrected bounds, benchmark, basin and q_hat.

    All arithmetic on numpy arrays to avoid pandas index-alignment surprises.
    """
    ens = load_ensemble(cell, SURFACES_ROOT)
    bench = benchmark_for(cell.entity, cell.algorithm, cell.track, PATHS)
    basins = basin_lookup()

    shared = ens.index.intersection(bench.index).intersection(basins.index)
    if len(shared) == 0:
        raise RuntimeError(f"no shared subc_ids for {cell.short()}")
    # report how much of the prediction surface survives the basin join
    n_surface = len(ens.index.intersection(bench.index))
    print(f"  [{cell.short()}] basin coverage: {len(shared)}/{n_surface} "
          f"surface pixels ({100*len(shared)/max(n_surface,1):.1f}%) have a basin")

    ens_a = ens.loc[shared]
    subc = np.asarray(shared)
    p_arr = ens_a.mean(axis=1).to_numpy()
    lo_arr = ens_a.quantile(alpha / 2, axis=1).to_numpy()
    hi_arr = ens_a.quantile(1 - alpha / 2, axis=1).to_numpy()
    bench_arr = bench.loc[shared].to_numpy()
    basin_arr = np.asarray(basins.loc[shared])

    lo_cor = np.full(len(shared), np.nan)
    hi_cor = np.full(len(shared), np.nan)
    qhat = np.full(len(shared), np.nan)

    valid_basins = pd.unique(basin_arr[~pd.isna(basin_arr)])
    if len(valid_basins) < 2:
        raise RuntimeError(
            f"{cell.short()} has {len(valid_basins)} basin(s); LOBO needs >= 2"
        )

    for b in valid_basins:
        is_test = basin_arr == b
        cal = ~is_test
        scores = nonconformity_scores(
            pd.Series(bench_arr[cal]), pd.Series(lo_arr[cal]), pd.Series(hi_arr[cal])
        )
        if len(scores) == 0:
            continue
        q = conformal_quantile(scores, alpha=alpha)
        lo_cor[is_test] = lo_arr[is_test] - q
        hi_cor[is_test] = hi_arr[is_test] + q
        qhat[is_test] = q

    df = pd.DataFrame({
        "cell_id": subc,               # subc_id
        "p": p_arr,
        "lo_unc": lo_arr, "hi_unc": hi_arr,
        "lo_cor": lo_cor, "hi_cor": hi_cor,
        "p_bench": bench_arr,
        "basin_id": basin_arr,
        "q_hat": qhat,
    }).dropna(subset=["lo_cor", "hi_cor"]).reset_index(drop=True)
    return df


def analyse_level(level: int, tau: float):
    cell = CellID(ENTITY, ALGO, TRACK, axis="lowacc", level=level)
    df = per_pixel_conformal(cell)
    flips = fa.compute_flips(df, tau=tau)
    summ = fa.summarise(flips, tau=tau)
    bridge_pixel = fa.benchmark_bridge(flips)          # width-inflation vs divergence (pixel)

    # periphery / overprediction signature: signed divergence at Flip A cells.
    # >0 means the contaminated point over-predicts vs the clean benchmark,
    # which is Finding 1 -- ties the management flip to the directional mechanism.
    div = flips["p"] - flips["p_bench"]
    A = flips["flip_A_withdrawn"].to_numpy()
    signature = {
        "signed_div_median_all": float(div.median()),
        "signed_div_median_flipA": float(div[A].median()) if A.any() else float("nan"),
        "bench_suit_median_all": float(flips["p_bench"].median()),
        "bench_suit_median_flipA": float(flips.loc[A, "p_bench"].median()) if A.any() else float("nan"),
    }

    # honest basin-level bridge: does per-basin q_hat track per-basin divergence?
    basin_tbl = (flips.assign(absdiv=div.abs())
                 .groupby("basin_id")
                 .agg(q_hat=("q_hat", "first"),
                      mean_absdiv=("absdiv", "mean"),
                      n=("cell_id", "size")))
    basin_corr = float(basin_tbl["q_hat"].corr(basin_tbl["mean_absdiv"], method="spearman"))

    return df, flips, summ, bridge_pixel, signature, basin_corr


def main() -> None:
    # preflight: confirm the basin lookup parquet is present and sane
    nb = basin_lookup()
    print(f"[preflight] basin lookup OK: {len(nb)} subc_ids, "
          f"{nb.nunique()} basins (from {BASIN_LOOKUP_PATH})\n")

    report: dict = {}
    for level in LEVELS:
        df, flips, summ, bridge_pixel, sig, basin_corr = analyse_level(level, TAU)
        flips.to_csv(OUT / f"flips_pleniusculus_L{level}.csv", index=False)
        report[f"L{level}"] = {
            "summary": summ,
            "bridge_pixel_widthinfl_vs_divergence": bridge_pixel.get(
                "rank_corr_widthinflation_vs_divergence"),
            "bridge_basin_qhat_vs_divergence_spearman": basin_corr,
            "signature": sig,
        }

    # tau sensitivity at the headline level (L10)
    tau_tbl = {}
    for tau in TAU_SWEEP:
        _, _, summ, *_ = analyse_level(10, tau)
        tau_tbl[f"tau={tau}"] = {
            "n_flip_A_withdrawn": summ["flip_A_withdrawn"]["n"],
            "n_flip_B_recovered": summ["flip_B_recovered"]["n"],
            "n_cells_domain": summ["n_cells_domain"],
        }
    report["tau_sensitivity_L10"] = tau_tbl

    print("=" * 70)
    pprint(report, sort_dicts=False)
    print("=" * 70)
    print(f"\nPer-pixel flip CSVs written to {OUT}/flips_pleniusculus_L*.csv")
    print("Join cell_id (subc_id) to your grid geometry for the Fig 5 map;")
    print("clip to the Sava-upper-Danube domain there (geometry not needed for counts).")


if __name__ == "__main__":
    main()
