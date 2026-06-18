"""Compute the four Protocol B headline numbers Lucian requested, from the CSVs.

1. worst-case uncorrected coverage
2. post-conformal coverage range
3. width inflation range (where correction applied)
4. fraction-of-pixels over-predicted at the lowest suitability bin under L10
"""
from pathlib import Path
import pandas as pd

FIG = Path("figures")
conf = pd.read_csv(FIG / "panel_conformal_protocol_b.csv")
asym5 = pd.read_csv(FIG / "asymmetry_protocol_b_5bin_L10.csv")

print("=" * 64)
print("PROTOCOL B — HEADLINE NUMBERS")
print("=" * 64)

# 1. worst-case uncorrected coverage
w = conf.loc[conf.coverage_uncorrected.idxmin()]
print(f"\n1. Worst-case uncorrected coverage: {w.coverage_uncorrected:.3f}")
print(f"   ({w.entity}, {w.track}, L{int(w.level)})")

# 2. post-conformal coverage range
print(f"\n2. Post-conformal coverage range: "
      f"{conf.coverage_conformal.min():.3f} – {conf.coverage_conformal.max():.3f}")
print(f"   (mean {conf.coverage_conformal.mean():.3f})")

# 3. width inflation range, where correction was actually applied (WIF > 1.001)
applied = conf[conf.width_inflation_factor > 1.001]
print(f"\n3. Width inflation range (where correction applied, n={len(applied)}): "
      f"{applied.width_inflation_factor.min():.2f}× – {applied.width_inflation_factor.max():.2f}×")
print(f"   (median {applied.width_inflation_factor.median():.2f}×; "
      f"cells needing no correction: {int((conf.width_inflation_factor <= 1.001).sum())})")

# 4. frac over-predicted at lowest suitability bin under L10 (mean across entities/tracks)
l10 = asym5[asym5.level == 10].copy()
l10["bin_rank"] = l10.groupby(["entity", "track"])["bench_decile_mid"].rank()
lowest = l10[l10.bin_rank == 1.0]
print(f"\n4. Over-prediction at lowest-suitability bin (L10): "
      f"mean {lowest.frac_over_predicted.mean():.3f}")
print(f"   (range across entities/tracks "
      f"{lowest.frac_over_predicted.min():.3f} – {lowest.frac_over_predicted.max():.3f}; "
      f"mean bin suitability {lowest.bench_decile_mid.mean():.3f})")

# bonus: highest bin for the contrast Lucian will use
highest = l10[l10.bin_rank == l10.bin_rank.max()]
print(f"\n   For contrast — highest-suitability bin (L10): "
      f"mean {highest.frac_over_predicted.mean():.3f} "
      f"(suitability {highest.bench_decile_mid.mean():.3f})")
print("\n" + "=" * 64)
