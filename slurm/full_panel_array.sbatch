#!/usr/bin/env bash
#SBATCH --job-name=ts_full
#SBATCH --output=logs/ts_full_%A_%a.out
#SBATCH --error=logs/ts_full_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --array=0-143%32
#
# Full-panel regeneration for trustworthy-sdm (Paper A).
#
# Array spans 144 indices (0..143), one per cell:
#   8 dual-axis entities x 2 algorithms x 3 tracks x 3 lowacc levels.
#
# %32 limits to 32 concurrent tasks. Adjust to fit cluster availability
# and your fairshare allocation. Each task does 30 replicates (~6.5 min
# at the pilot's 13 sec/fit throughput); the whole panel finishes in
# under an hour wall time when cluster has 32 free slots.
#
# Each task writes:
#   data/replicate_surfaces/{cell}/rep_NN.parquet  (30 files)
#   data/replicate_surfaces/summaries/summary_NNNN.csv (1 file)
#
# Inspect manifest before running:
#   trustworthy-sdm-inspect --print-manifest full --results-root <root>
# Or test a single index locally:
#   trustworthy-sdm-regenerate --cells full --array-index 0 --dry-run

set -euo pipefail

REPO=${REPO:-/ceph/hpc/home/miokk/trustworthy-sdm}
UPSTREAM=${UPSTREAM:-/ceph/hpc/home/miokk/sdm-robustness}

cd "$REPO"
mkdir -p logs

export PYTHONNOUSERSITE=1
PY="$REPO/.venv/bin/python"

echo "==> array task ${SLURM_ARRAY_TASK_ID:?must be set}"
module load Python/3.12.3-GCCcore-13.3.0
"$PY" --version
"$PY" -c "import sdm_robustness; print('sdm_robustness:', sdm_robustness.__file__)"
"$PY" -c "import trustworthy_sdm; print('trustworthy_sdm:', trustworthy_sdm.__version__)"

export TS_RESULTS_ROOT="$UPSTREAM/results"
export TS_MASTER_CSV="$UPSTREAM/data/combined_data_true_master.csv"
export TS_OUT_ROOT="$REPO/data/replicate_surfaces"

echo "==> regenerate (full panel, index ${SLURM_ARRAY_TASK_ID})"
"$REPO/.venv/bin/trustworthy-sdm-regenerate" \
    --results-root "$TS_RESULTS_ROOT" \
    --master-csv "$TS_MASTER_CSV" \
    --out-root "$TS_OUT_ROOT" \
    --cells full \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    -v

echo "==> done (index ${SLURM_ARRAY_TASK_ID})"
