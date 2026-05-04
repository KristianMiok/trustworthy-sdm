"""Smoke test for assemble_inputs.

This test is skipped if sdm_robustness or the master CSV is not available
(typical on Mac before VEGA-side data is rsynced). On VEGA inside the .venv,
it should pass after `pip install -e /ceph/hpc/home/miokk/sdm-robustness`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _master_csv_path() -> Path | None:
    candidates = [
        Path(os.environ.get("TS_MASTER_CSV", "")),
        Path("/ceph/hpc/home/miokk/sdm-robustness/data/combined_data_true_master.csv"),
        Path.cwd() / "data" / "master" / "combined_data_true_master.csv",
    ]
    for c in candidates:
        if c and c.is_file():
            return c
    return None


@pytest.fixture(scope="module")
def master_csv() -> Path:
    p = _master_csv_path()
    if p is None:
        pytest.skip("master CSV not found; set TS_MASTER_CSV or run on VEGA")
    return p


@pytest.fixture(scope="module")
def sdm_robustness_available() -> bool:
    try:
        import sdm_robustness  # noqa: F401
        return True
    except ImportError:
        pytest.skip("sdm_robustness not installed")
        return False


def test_assemble_inputs_torrentium_lowacc(master_csv: Path, sdm_robustness_available: bool) -> None:
    """assemble_inputs should produce non-empty DataFrames for the pilot entity."""
    from trustworthy_sdm.regen import assemble_inputs

    inputs = assemble_inputs(
        entity="Austropotamobius torrentium (pooled)",
        track="combined",
        axis="lowacc",
        master_csv=master_csv,
    )

    assert len(inputs.benchmark) > 0, "benchmark is empty"
    assert len(inputs.contamination_pool) > 0, "lowacc pool is empty"
    assert len(inputs.accessible_area) > 0, "accessible area is empty"
    assert "subc_id" in inputs.accessible_area.columns
    assert inputs.n_experiment == len(inputs.benchmark)


def test_assemble_inputs_torrentium_benchmark(master_csv: Path, sdm_robustness_available: bool) -> None:
    """Benchmark axis should map to snap_pool (mathematically inert at level=0)."""
    from trustworthy_sdm.regen import assemble_inputs

    inputs = assemble_inputs(
        entity="Austropotamobius torrentium (pooled)",
        track="combined",
        axis="benchmark",
        master_csv=master_csv,
    )

    assert len(inputs.benchmark) > 0
    # snap_pool may be empty for some entities, but accessible_area must not be
    assert len(inputs.accessible_area) > 0
