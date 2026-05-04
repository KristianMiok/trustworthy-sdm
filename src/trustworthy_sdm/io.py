"""Loaders for Grid B output produced by the previous project.

The companion paper's pipeline writes a directory structure like::

    {grid_b_root}/
        {entity_dir}_rfxgb/
            results_raw.parquet
            surfaces/
                {entity_dir}_random_forest_combined_benchmark.parquet
                {entity_dir}_random_forest_combined_lowacc_max.parquet
                ...
            variable_importance_vectors.parquet
        {entity_dir}_maxent_lowacc_combined_L10/
            results_raw.parquet
            surfaces/
                ...

This module provides typed accessors so we don't sprinkle path-string
manipulation throughout the analysis code.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Entity name conventions
#
# Entity directory names use underscores (e.g. "Austropotamobius_torrentium_pooled").
# Entity values inside results_raw.parquet use spaces and parentheses
# (e.g. "Austropotamobius torrentium (pooled)"). Both forms are needed.

ENTITY_NAME_TO_DIR: dict[str, str] = {
    "Astacus astacus": "Astacus_astacus",
    "Austropotamobius fulcisianus (pooled)": "Austropotamobius_fulcisianus_pooled",
    "Austropotamobius torrentium (pooled)": "Austropotamobius_torrentium_pooled",
    "Cambarus latimanus": "Cambarus_latimanus",
    "Cambarus striatus": "Cambarus_striatus",
    "Creaserinus fodiens": "Creaserinus_fodiens",
    "Faxonius limosus (alien)": "Faxonius_limosus_alien",
    "Faxonius limosus (native)": "Faxonius_limosus_native",
    "Lacunicambarus diogenes": "Lacunicambarus_diogenes",
    "Pacifastacus leniusculus (alien)": "Pacifastacus_leniusculus_alien",
    "Pontastacus leptodactylus (pooled)": "Pontastacus_leptodactylus_pooled",
    "Procambarus clarkii (alien)": "Procambarus_clarkii_alien",
    "Procambarus clarkii (native)": "Procambarus_clarkii_native",
}

# 8 dual-axis entities (testable on both contamination axes)
DUAL_AXIS_ENTITIES: tuple[str, ...] = (
    "Astacus astacus",
    "Austropotamobius fulcisianus (pooled)",
    "Austropotamobius torrentium (pooled)",
    "Faxonius limosus (alien)",
    "Pacifastacus leniusculus (alien)",
    "Pontastacus leptodactylus (pooled)",
    "Procambarus clarkii (alien)",
    "Procambarus clarkii (native)",
)

ALGORITHMS: tuple[str, ...] = ("random_forest", "xgboost", "maxent")
TRACKS: tuple[str, ...] = ("local_only", "upstream_only", "combined")
AXES: tuple[str, ...] = ("snapping", "lowacc")


# ---------------------------------------------------------------------------
# Cell descriptor

@dataclass(frozen=True)
class CellID:
    """Identifies one (entity, algorithm, track, axis, level) Grid B cell.

    A "benchmark" cell uses ``axis="benchmark"`` and ``level=0``; we canonicalise
    benchmark seed-lookup on ``axis="snapping", level=0`` rows in the metrics
    table because both axes share the same benchmark seeds.
    """

    entity: str           # canonical name, e.g. "Austropotamobius torrentium (pooled)"
    algorithm: str        # "random_forest" | "xgboost" | "maxent"
    track: str            # "local_only" | "upstream_only" | "combined"
    axis: str             # "snapping" | "lowacc" | "benchmark"
    level: int            # 0 for benchmark, otherwise contamination level

    @property
    def entity_dir(self) -> str:
        return ENTITY_NAME_TO_DIR[self.entity]

    def short(self) -> str:
        """Short identifier used as directory name for replicate surfaces."""
        return f"{self.entity_dir}__{self.algorithm}__{self.track}__{self.axis}__L{self.level}"


# ---------------------------------------------------------------------------
# Path resolution

@dataclass(frozen=True)
class GridBPaths:
    """Resolves filesystem paths for Grid B output.

    Parameters
    ----------
    root : Path
        Path to the directory that contains ``grid_b_full/``,
        ``grid_b_merged/``, and ``task5c_benchmark_stability_array/``.

        On VEGA: ``/ceph/hpc/home/miokk/sdm-robustness/results``.
        On Mac after rsync: ``./data/results``.
    """

    root: Path

    @property
    def grid_b_full(self) -> Path:
        return self.root / "grid_b_full"

    @property
    def grid_b_merged(self) -> Path:
        return self.root / "grid_b_merged" / "grid_b_results_raw_merged.parquet"

    @property
    def benchmark_stability(self) -> Path:
        return self.root / "task5c_benchmark_stability_array"

    def rfxgb_cell_dir(self, entity: str) -> Path:
        """Directory holding RF and XGBoost output for one entity."""
        return self.grid_b_full / f"{ENTITY_NAME_TO_DIR[entity]}_rfxgb"

    def existing_surface(
        self,
        cell: CellID,
        kind: str = "benchmark",  # "benchmark" | "lowacc_max" | "snapping_max"
    ) -> Path:
        """Path to the single-prediction surface saved by the companion pipeline."""
        if cell.algorithm not in {"random_forest", "xgboost"}:
            raise ValueError(
                f"existing_surface() only supports rf/xgboost, got {cell.algorithm}"
            )
        cell_dir = self.rfxgb_cell_dir(cell.entity)
        fname = (
            f"{ENTITY_NAME_TO_DIR[cell.entity]}_{cell.algorithm}_{cell.track}_{kind}.parquet"
        )
        return cell_dir / "surfaces" / fname


# ---------------------------------------------------------------------------
# Loaders

def load_merged_metrics(paths: GridBPaths) -> pd.DataFrame:
    """Load the full per-replicate metrics table (~17k rows × 42 columns)."""
    return pd.read_parquet(paths.grid_b_merged)


def load_existing_surface(paths: GridBPaths, cell: CellID, kind: str) -> pd.DataFrame:
    """Load one of the single-prediction surfaces from the companion pipeline."""
    return pd.read_parquet(paths.existing_surface(cell, kind=kind))


def load_benchmark_stability(
    paths: GridBPaths, entity: str, algorithm: str, track: str
) -> pd.DataFrame:
    """Load the Task 5c benchmark stability envelope for one (entity, algo, track)."""
    cell_dir = (
        paths.benchmark_stability
        / f"{ENTITY_NAME_TO_DIR[entity]}_{algorithm}_{track}"
        / "benchmark_stability.parquet"
    )
    return pd.read_parquet(cell_dir)


def replicate_seeds(metrics: pd.DataFrame, cell: CellID) -> pd.DataFrame:
    """Extract the 30 (replicate, seed) pairs for one cell from the metrics table.

    Returns a DataFrame with columns ['replicate', 'seed'], sorted by replicate.
    Raises KeyError if the cell is not found.
    """
    if cell.axis == "benchmark":
        # Benchmark rows have level=0 on either axis; we canonicalise on snapping.
        mask = (
            (metrics["entity"] == cell.entity)
            & (metrics["algorithm"] == cell.algorithm)
            & (metrics["track"] == cell.track)
            & (metrics["axis"] == "snapping")
            & (metrics["level"] == 0)
        )
    else:
        mask = (
            (metrics["entity"] == cell.entity)
            & (metrics["algorithm"] == cell.algorithm)
            & (metrics["track"] == cell.track)
            & (metrics["axis"] == cell.axis)
            & (metrics["level"] == cell.level)
        )

    sub = (
        metrics.loc[mask, ["replicate", "seed"]]
        .drop_duplicates()
        .sort_values("replicate")
        .reset_index(drop=True)
    )
    if len(sub) == 0:
        raise KeyError(f"No replicates found for cell {cell.short()}")
    return sub


# ---------------------------------------------------------------------------
# Iteration helpers

# Pilot: full lowacc contamination curve for one entity.
# Six cells = 2 algorithms (RF, XGBoost) x 3 contamination levels (3, 10, 20).
# We do NOT regenerate benchmark replicates in the pilot — Path 2 uses the
# companion paper's saved deterministic benchmark surface from disk as the
# reference target. Benchmark regeneration is a full-panel concern.
PILOT_LOWACC_LEVELS: tuple[int, ...] = (3, 10, 20)


def iter_pilot_cells() -> Iterator[CellID]:
    """Pilot: A. torrentium combined RF+XGB at lowacc levels 3, 10, 20.

    Six cells x 30 replicates = 180 fits. ~105 minutes wall on a single node
    in one Slurm job. Output: 180 parquet files in data/replicate_surfaces/.

    Benchmark surface comes from disk (companion paper's saved single-prediction
    surface) — see GridBPaths.existing_surface(cell, kind="benchmark").
    """
    entity = "Austropotamobius torrentium (pooled)"
    track = "combined"
    for algorithm in ("random_forest", "xgboost"):
        for level in PILOT_LOWACC_LEVELS:
            yield CellID(entity, algorithm, track, axis="lowacc", level=level)


# ---------------------------------------------------------------------------
# Full-panel cell iteration

# Contamination levels for the full Grid B sweep on the lowacc axis.
# Level 0 (benchmark) is excluded — Path 2 uses the companion paper's
# saved deterministic benchmark surface from disk as the reference target.
FULL_PANEL_LOWACC_LEVELS: tuple[int, ...] = (3, 10, 20)


def iter_full_panel_cells() -> Iterator[CellID]:
    """Full panel: 8 dual-axis entities x 2 algorithms x 3 tracks x 3 lowacc levels.

    Total: 144 cells x 30 replicates = 4,320 fits. With one cell per Slurm
    array task and ~13 sec/fit throughput, each array task takes ~6.5 min;
    144 array tasks finish in well under an hour given enough cluster slots.

    Cell ordering is deterministic so that array index N maps to the same
    cell across every invocation. Order: entity (alphabetical) -> algorithm ->
    track -> level.
    """
    for entity in DUAL_AXIS_ENTITIES:
        for algorithm in ("random_forest", "xgboost"):
            for track in TRACKS:
                for level in FULL_PANEL_LOWACC_LEVELS:
                    yield CellID(entity, algorithm, track, axis="lowacc", level=level)


def cell_at_index(cells: list[CellID], index: int) -> CellID:
    """Return cells[index], raising a clear error if out of range."""
    if index < 0 or index >= len(cells):
        raise IndexError(
            f"array index {index} out of range; "
            f"valid indices are 0..{len(cells) - 1} for this cell list"
        )
    return cells[index]
