# trustworthy-sdm

Calibrated uncertainty for species distribution models under occurrence-data contamination.

Methodological follow-up to Pârvulescu, Petko & Miok (in submission). The companion paper documents that under realistic occurrence-data contamination, SDMs pass standard validation while predicting 25–30% inflated ranges. This repository develops calibrated-uncertainty methods (ensemble variance baseline, split conformal prediction, basin-Mondrian conformal) that detect this hidden spatial bias.

## Status

**Pre-pilot, Step 0.** We are regenerating per-replicate suitability surfaces by replaying the companion paper's `fit_cv_cell` with each of the 30 logged seeds. The companion paper's saved surfaces are deterministic single-prediction surfaces — no replicate dimension — so pixel-wise ensemble uncertainty cannot be computed from them.

**One blocking task before regeneration runs:** locating the orchestrator that splits the master CSV into `(benchmark, contamination_pool, accessible_area)` for one entity. The `sdm_robustness` package itself contains only `core.py`; the orchestration logic lives in shell wrappers or scripts at the repo root that we have not yet inspected. See `scripts/inspect_orchestrator.sh`.

## Repository layout

```
trustworthy-sdm/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/trustworthy_sdm/
│   ├── __init__.py
│   ├── io.py                     # Loaders for Grid B output
│   ├── regen.py                  # Per-replicate surface regeneration
│   └── cli.py                    # Command-line entry points
├── scripts/
│   ├── inspect_orchestrator.sh   # Probe upstream repo to locate input prep
│   └── pilot_regenerate.py       # Pilot: torrentium combined RF+XGB, bench+L20
├── slurm/
│   └── pilot_torrentium.sbatch   # Slurm job for the pilot
├── notebooks/
│   └── 01_pilot_coverage.py      # First analysis (after regeneration)
├── tests/
├── configs/
│   └── pilot.yaml
└── data/                         # Gitignored; populated by rsync from VEGA
    └── README.md                 # Tells you what should be here
```

## Setup — Mac (development)

> **Either pip+venv or uv works.** The instructions below use pip+venv to match the previous repo. If you prefer uv, the equivalent commands are: `uv venv && uv pip install -e ../sdm-robustness && uv pip install -e ".[dev]"`. The committed `uv.lock` is harmless when not using uv.


```bash
# 1. Clone this repo and the previous repo side-by-side
git clone git@github.com:KristianMiok/trustworthy-sdm.git
git clone git@github.com:KristianMiok/sdm-robustness.git   # if not already cloned

# 2. Create venv
cd trustworthy-sdm
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install the previous package as editable, then this one
pip install -e ../sdm-robustness
pip install -e ".[dev]"

# 4. Verify imports
python -c "from sdm_robustness.pipeline.core import fit_cv_cell; print('OK')"
python -c "import trustworthy_sdm; print(trustworthy_sdm.__version__)"
```

In PyCharm: open this directory as a project, point the project interpreter at `.venv/bin/python`, and PyCharm will index both packages (because `sdm_robustness` is editable-installed it shows up in External Libraries with full source navigation).

## Setup — VEGA (compute)

```bash
ssh miokk@vega
cd /ceph/hpc/home/miokk

# 1. Clone trustworthy-sdm next to sdm-robustness
git clone git@github.com:KristianMiok/trustworthy-sdm.git
cd trustworthy-sdm

# 2. Use the same Python module the previous project uses
module load Python/3.12.3-GCCcore-13.3.0
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install the previous package as editable from its absolute path
pip install -e /ceph/hpc/home/miokk/sdm-robustness
pip install -e .

# 4. Smoketest
python -c "from sdm_robustness.pipeline.core import fit_cv_cell; print('OK')"
trustworthy-sdm-inspect --help
```

## Step 0 — locate the orchestrator (do this once on VEGA)

Before any regeneration can run, we need to find how the previous project splits the master CSV into the three DataFrames `fit_cv_cell` consumes. Run this on VEGA:

```bash
cd /ceph/hpc/home/miokk/trustworthy-sdm
bash scripts/inspect_orchestrator.sh
```

Paste the output back into the planning thread. We expect to find a function or script in `sdm-robustness/scripts/`, `sdm-robustness/analysis/`, or one of the `run_*.sh` files at the repo root, that reads `combined_data_true_master.csv` and produces the entity-specific benchmark / contamination pool / accessible area DataFrames. Once located, `assemble_inputs()` in `regen.py` becomes a thin wrapper around it.

## Pilot run (after orchestrator is wired)

```bash
# On VEGA, after Step 0
cd /ceph/hpc/home/miokk/trustworthy-sdm
sbatch slurm/pilot_torrentium.sbatch
```

This regenerates *Austropotamobius torrentium* (pooled) combined-track surfaces for RF and XGBoost, at benchmark and lowacc-max levels. Output: 4 cells × 30 replicates = 120 parquet files in `data/replicate_surfaces/`.

After the job completes, rsync to your Mac:

```bash
# On Mac
rsync -avz miokk@vega:/ceph/hpc/home/miokk/trustworthy-sdm/data/replicate_surfaces/ \
            ./data/replicate_surfaces/
```

Then open `notebooks/01_pilot_coverage.py` (a percent-script that PyCharm will treat as a notebook).

## What this paper claims (Paper A)

1. **Negative result.** Pixel-wise ensemble variance from RF/XGBoost replicate ensembles produces 95% intervals that under-cover dramatically when occurrence data are contaminated, while Tier 1 metrics (AUC, TSS, Boyce) and Tier 2 importance rankings remain in their stability envelopes. Standard validation cannot detect what standard uncertainty cannot represent.

2. **Positive result.** Basin-Mondrian split conformal prediction restores nominal coverage on benchmark data and degrades gracefully under contamination. Conformal interval width increases monotonically with contamination level — the intervals know the model is in trouble even when the practitioner does not.

3. **Diagnostic.** Within-entity, conformal interval width on held-out spatial folds is monotonic in contamination level. Proposed as an ad-hoc red flag for practitioners without ground truth.

See `docs/concept_notes.docx` for the full plan, and `docs/paper_a_pilot_additions.md` for operational additions agreed with Lucian after his review.

## Methodological note on the ensemble baseline

We use the `final_model` surface returned by `fit_cv_cell(return_artifacts=True)` as each replicate's ensemble member. This is a model trained on the full contaminated dataset (no holdout). We chose this over per-fold OOF ensembles because (a) it is the surface a practitioner would naturally compute when refitting on all data after CV, and (b) the negative result is therefore framed against the most natural baseline rather than a methodologically refined one. Per-fold OOF analysis is deferred to Paper B.
