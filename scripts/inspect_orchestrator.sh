#!/usr/bin/env bash
# inspect_orchestrator.sh — locate the upstream input-preparation code.
#
# The sdm_robustness package contains only core.py with fit_cv_cell. The code
# that splits combined_data_true_master.csv into (benchmark, contamination
# pool, accessible area) for one entity must live elsewhere — most likely
# in scripts/ or analysis/, or in the run_*.sh wrappers at the repo root.
#
# Run on VEGA, paste the entire output into the planning thread.

set -u

UPSTREAM=${UPSTREAM:-/ceph/hpc/home/miokk/sdm-robustness}

if [[ ! -d "$UPSTREAM" ]]; then
    echo "ERROR: upstream repo not found at $UPSTREAM" >&2
    echo "       set UPSTREAM=/path/to/sdm-robustness and retry" >&2
    exit 2
fi

cd "$UPSTREAM"

banner() { printf '\n=== %s ===\n' "$1"; }

banner "1. top-level layout"
ls -la

banner "2. python entry points (anything with __main__ or argparse)"
grep -rln '__main__\|argparse\|click' --include='*.py' . 2>/dev/null \
    | grep -v '\.venv' | grep -v '__pycache__' | head -30

banner "3. callers of fit_cv_cell"
grep -rn 'fit_cv_cell' --include='*.py' --include='*.sh' --include='*.ipynb' . 2>/dev/null \
    | grep -v '\.venv' | grep -v '__pycache__'

banner "4. callers of prepare_accessible_area"
grep -rn 'prepare_accessible_area\|contaminate_presence_set' --include='*.py' . 2>/dev/null \
    | grep -v '\.venv' | grep -v '__pycache__'

banner "5. references to combined_data_true_master.csv"
grep -rn 'combined_data_true_master\|true_master' --include='*.py' --include='*.sh' --include='*.yaml' . 2>/dev/null \
    | grep -v '\.venv' | grep -v '__pycache__' | head -20

banner "6. one of the run_*.sh shell wrappers (sample)"
ls run_*.sh 2>/dev/null | head -3
echo "--- contents of first run_*.sh ---"
first_run=$(ls run_*.sh 2>/dev/null | head -1)
[[ -n "$first_run" ]] && cat "$first_run"

banner "7. scripts/ directory"
ls scripts/ 2>/dev/null
echo "--- python files under scripts/ ---"
find scripts -name '*.py' 2>/dev/null | head -20

banner "8. analysis/ directory"
ls analysis/ 2>/dev/null
find analysis -name '*.py' 2>/dev/null | head -20

banner "9. signature of any function that builds (benchmark, pool, accessible)"
grep -rn 'def.*benchmark.*pool\|def.*entity_panel\|def.*build_inputs\|def.*prepare_entity\|def.*accessible' \
    --include='*.py' . 2>/dev/null \
    | grep -v '\.venv' | grep -v '__pycache__' | head -20

banner "10. configs/ contents"
ls configs/ 2>/dev/null
echo "--- one config file ---"
first_cfg=$(find configs -maxdepth 2 -type f 2>/dev/null | head -1)
[[ -n "$first_cfg" ]] && head -40 "$first_cfg"

banner "done"
