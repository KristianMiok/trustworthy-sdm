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
    # 2 algorithms x 3 lowacc levels = 6 cells
    assert len(cells) == 6
    assert len({c.short() for c in cells}) == 6
    # All cells are A. torrentium combined lowacc
    assert all(c.entity == "Austropotamobius torrentium (pooled)" for c in cells)
    assert all(c.track == "combined" for c in cells)
    assert all(c.axis == "lowacc" for c in cells)
    # Levels are 3, 10, 20
    assert {c.level for c in cells} == {3, 10, 20}
    # Both algorithms appear
    assert {c.algorithm for c in cells} == {"random_forest", "xgboost"}
