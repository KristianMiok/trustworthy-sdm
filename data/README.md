# data/

This directory is gitignored. It is populated by rsync from VEGA after the
regeneration job completes.

## Expected layout (after first pilot rsync)

```
data/
├── README.md                            # this file (committed)
├── replicate_surfaces/                  # gitignored
│   ├── Austropotamobius_torrentium_pooled__random_forest__combined__benchmark__L0/
│   │   ├── rep_00.parquet
│   │   ├── rep_01.parquet
│   │   ├── ... (30 files)
│   │   └── rep_29.parquet
│   ├── Austropotamobius_torrentium_pooled__random_forest__combined__lowacc__L20/
│   ├── Austropotamobius_torrentium_pooled__xgboost__combined__benchmark__L0/
│   └── Austropotamobius_torrentium_pooled__xgboost__combined__lowacc__L20/
└── results/                             # gitignored; companion paper output
    ├── grid_b_full/                     # ~66 MB
    ├── grid_b_merged/
    │   └── grid_b_results_raw_merged.parquet
    └── task5c_benchmark_stability_array/
```

## Pulling data from VEGA (one-time)

```bash
# Companion paper's Grid B output (~70 MB total)
rsync -avz miokk@vega:/ceph/hpc/home/miokk/sdm-robustness/results/grid_b_full/        ./data/results/grid_b_full/
rsync -avz miokk@vega:/ceph/hpc/home/miokk/sdm-robustness/results/grid_b_merged/      ./data/results/grid_b_merged/
rsync -avz miokk@vega:/ceph/hpc/home/miokk/sdm-robustness/results/task5c_benchmark_stability_array/  \
                      ./data/results/task5c_benchmark_stability_array/

# Replicate surfaces (after the Slurm job runs)
rsync -avz miokk@vega:/ceph/hpc/home/miokk/trustworthy-sdm/data/replicate_surfaces/  ./data/replicate_surfaces/
```
