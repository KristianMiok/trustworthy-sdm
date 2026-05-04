"""Smoke tests for trustworthy_sdm.io.

These do not require the full Grid B output to be present; they only verify
the metadata mappings and CellID semantics.
"""

from __future__ import annotations

import pytest

from trustworthy_sdm.io import (
    ALGORITHMS,
    AXES,
    DUAL_AXIS_ENTITIES,
    ENTITY_NAME_TO_DIR,
    TRACKS,
    CellID,
    iter_pilot_cells,
)


def test_entity_mapping_is_bijective() -> None:
    dirs = list(ENTITY_NAME_TO_DIR.values())
    assert len(dirs) == len(set(dirs)), "duplicate entity_dir values"


def test_dual_axis_entities_known() -> None:
    for entity in DUAL_AXIS_ENTITIES:
        assert entity in ENTITY_NAME_TO_DIR, f"unknown entity: {entity}"


def test_algorithms_canonical() -> None:
    assert ALGORITHMS == ("random_forest", "xgboost", "maxent")


def test_tracks_canonical() -> None:
    assert TRACKS == ("local_only", "upstream_only", "combined")


def test_axes_canonical() -> None:
    assert AXES == ("snapping", "lowacc")


def test_cellid_short_is_distinctive() -> None:
    a = CellID("Astacus astacus", "random_forest", "combined", "snapping", 5)
    b = CellID("Astacus astacus", "random_forest", "combined", "lowacc", 5)
    assert a.short() != b.short()


def test_cellid_unknown_entity_raises() -> None:
    cell = CellID("Made up species", "random_forest", "combined", "lowacc", 20)
    with pytest.raises(KeyError):
        _ = cell.entity_dir


def test_pilot_cells_count_and_uniqueness() -> None:
    cells = list(iter_pilot_cells())
    assert len(cells) == 4
    assert len({c.short() for c in cells}) == 4
    # All four cells should be the same entity and track
    assert all(c.entity == "Austropotamobius torrentium (pooled)" for c in cells)
    assert all(c.track == "combined" for c in cells)
