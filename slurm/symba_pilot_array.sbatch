#!/usr/bin/env bash
#SBATCH --job-name=ts_symba_pilot
#SBATCH --output=logs/ts_symba_pilot_%A_%a.out
#SBATCH --error=logs/ts_symba_pilot_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --array=0-59%32
#
# Symba pilot: Austropotamobius torrentium (pooled), combined track,
# benchmark (L0) + lowacc max (L20), 30 replicates each.
#
# Array spans 60 indices (0..59), one per (cell, replicate) work unit.
#   index 0..29  -> cell 0 (benchmark, L0), replicates 0..29
#   index 30..59 -> cell 1 (lowacc, L20),   replicates 0..29
#
# %32 limits to 32 concurrent tasks. With ~5 min per task at Mac speed
# (likely faster on VEGA), wall time is roughly max(per-task time)
# = ~5-10 min when 32 slots are available simultaneously.
#
# Per task this writes:
#   data/replicate_surfaces/{cell}/rep_NN.parquet  (1 file)
#   data/replicate_surfaces/{cell}/seeds.csv        (overwritten each task; benign)
#
# Test a single index locally before submitting:
#   python scripts/run_symba_pilot.py \
#       --master-csv .../combined_data_true_master.csv \
#       --out-root data/replicate_surfaces \
#       --array-index 0 -v

set -euo pipefail

REPO=${REPO:-/ceph/hpc/home/miokk/trustworthy-sdm}
UPSTREAM=${UPSTREAM:-/ceph/hpc/home/miokk/sdm-robustness}

cd "$REPO"
mkdir -p logs

# Use the Symba-specific venv (Python 3.12 + Symba) explicitly. The
# vanilla .venv used for RF/XGBoost panels does NOT have Symba installed.
export PYTHONNOUSERSITE=1
PY="$REPO/.venv-symba/bin/python"

echo "==> array task ${SLURM_ARRAY_TASK_ID:?must be set}"
module load Python/3.12.3-GCCcore-13.3.0
"$PY" --version
"$PY" -c "import sdm_robustness; print('sdm_robustness:', sdm_robustness.__file__)"
"$PY" -c "import trustworthy_sdm; print('trustworthy_sdm:', trustworthy_sdm.__version__)"
"$PY" -c "from symba import SymbaClassifier; print('symba: ok')"

echo "==> run_symba_pilot --array-index ${SLURM_ARRAY_TASK_ID}"
"$PY" scripts/run_symba_pilot.py \
    --master-csv "$UPSTREAM/data/raw/combined_data_true_master.csv" \
    --out-root "$REPO/data/replicate_surfaces" \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    -v

echo "==> done (index ${SLURM_ARRAY_TASK_ID})"