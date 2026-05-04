"""Pilot regeneration: torrentium combined RF+XGB at benchmark and lowacc L20.

Invoked by ``slurm/pilot_torrentium.sbatch``. This is a thin wrapper around
``trustworthy_sdm.cli.regenerate_main`` so we can keep environment paths
out of the Slurm script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from trustworthy_sdm.cli import regenerate_main

# Defaults for VEGA. Override with environment variables if needed.
RESULTS_ROOT = Path(os.environ.get(
    "TS_RESULTS_ROOT", "/ceph/hpc/home/miokk/sdm-robustness/results"
))
MASTER_CSV = Path(os.environ.get(
    "TS_MASTER_CSV",
    "/ceph/hpc/home/miokk/sdm-robustness/data/combined_data_true_master.csv",
))
OUT_ROOT = Path(os.environ.get(
    "TS_OUT_ROOT",
    "/ceph/hpc/home/miokk/trustworthy-sdm/data/replicate_surfaces",
))


if __name__ == "__main__":
    sys.exit(regenerate_main([
        "--results-root", str(RESULTS_ROOT),
        "--master-csv", str(MASTER_CSV),
        "--out-root", str(OUT_ROOT),
        "--cells", "pilot",
        "--verbose",
    ]))
