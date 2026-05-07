#!/usr/bin/env bash
#SBATCH --job-name=ts_rfxgb_l0
#SBATCH --output=logs/ts_rfxgb_l0_%A_%a.out
#SBATCH --error=logs/ts_rfxgb_l0_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --array=0-59%32
#
# RF + XGBoost benchmark/L0 surfaces for A. torrentium pooled, combined track.
# Fills the gap not covered by iter_pilot_cells() so we have a clean L0
# baseline to anchor the calibration analysis against.
#
# 2 algorithms x 30 replicates = 60 array tasks. RF/XGBoost are fast
# (~30-60 sec per fit), so 10-min walltime is generous.

set -euo pipefail

REPO=${REPO:-/ceph/hpc/home/miokk/trustworthy-sdm}
UPSTREAM=${UPSTREAM:-/ceph/hpc/home/miokk/sdm-robustness}

cd "$REPO"
mkdir -p logs

export PYTHONNOUSERSITE=1
PY="$REPO/.venv-symba/bin/python"

echo "==> array task ${SLURM_ARRAY_TASK_ID:?must be set}"
module load Python/3.12.3-GCCcore-13.3.0
"$PY" --version
"$PY" -c "import sdm_robustness; print('sdm_robustness:', sdm_robustness.__file__)"
"$PY" -c "import trustworthy_sdm; print('trustworthy_sdm:', trustworthy_sdm.__version__)"
"$PY" -c "import xgboost; print('xgboost:', xgboost.__version__)"

echo "==> run_rfxgb_l0 --array-index ${SLURM_ARRAY_TASK_ID}"
"$PY" scripts/run_rfxgb_l0.py \
    --master-csv "$UPSTREAM/data/combined_data_true_master.csv" \
    --out-root "$REPO/data/replicate_surfaces" \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    -v

echo "==> done (index ${SLURM_ARRAY_TASK_ID})"
