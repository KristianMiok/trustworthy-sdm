#!/usr/bin/env bash
#SBATCH --job-name=ts_pilot_torrentium
#SBATCH --output=logs/ts_pilot_torrentium_%j.out
#SBATCH --error=logs/ts_pilot_torrentium_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#
# Pilot regeneration for the trustworthy-sdm follow-up paper.
#
# Regenerates per-replicate suitability surfaces for one entity (Austropotamobius
# torrentium pooled) on the combined-track, with both RF and XGBoost, at
# benchmark (level 0) and lowacc max (level 20).
#
# Pilot scope (Path 2): A. torrentium combined RF+XGB, lowacc L3+L10+L20.
# Output: 6 cells x 30 replicates = 180 parquet files in
#   /ceph/hpc/home/miokk/trustworthy-sdm/data/replicate_surfaces/
#
# Estimated wall time: ~105 minutes (180 fits x ~35 sec each plus overhead).
# The 3 h limit gives headroom. Single-job execution; full-panel parallelism
# happens in slurm/full_panel_array.sbatch (separate, not yet built).
#
# After this completes successfully, scale up by adapting this script for
# the full 8-entity panel (96 cells) — likely as a Slurm array.

set -euo pipefail

REPO=${REPO:-/ceph/hpc/home/miokk/trustworthy-sdm}
UPSTREAM=${UPSTREAM:-/ceph/hpc/home/miokk/sdm-robustness}

cd "$REPO"
mkdir -p logs

# Use the venv's Python explicitly so we never depend on PATH resolution
# inside the Slurm batch shell. Match the user-site isolation we tested
# manually with PYTHONNOUSERSITE=1.
export PYTHONNOUSERSITE=1
PY="$REPO/.venv/bin/python"

echo "==> environment"
module load Python/3.12.3-GCCcore-13.3.0
"$PY" --version
"$PY" -c "import sdm_robustness; print('sdm_robustness:', sdm_robustness.__file__)"
"$PY" -c "import trustworthy_sdm; print('trustworthy_sdm:', trustworthy_sdm.__version__)"
"$PY" -c "import xgboost; print('xgboost:', xgboost.__version__)"

echo "==> inspect (sanity check)"
"$REPO/.venv/bin/trustworthy-sdm-inspect" --results-root "$UPSTREAM/results" -v

echo "==> regenerate (pilot)"
export TS_RESULTS_ROOT="$UPSTREAM/results"
export TS_MASTER_CSV="$UPSTREAM/data/combined_data_true_master.csv"
export TS_OUT_ROOT="$REPO/data/replicate_surfaces"
"$PY" scripts/pilot_regenerate.py

echo "==> done"
ls -la "$TS_OUT_ROOT"
find "$TS_OUT_ROOT" -name '*.parquet' | wc -l
