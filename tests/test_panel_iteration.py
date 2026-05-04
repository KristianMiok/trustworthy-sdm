"""Tests for the full-panel cell iterator and array-index dispatch."""

from __future__ import annotations

import pytest

from trustworthy_sdm.io import (
    DUAL_AXIS_ENTITIES,
    FULL_PANEL_LOWACC_LEVELS,
    cell_at_index,
    iter_full_panel_cells,
)


def test_full_panel_count() -> None:
    cells = list(iter_full_panel_cells())
    expected = len(DUAL_AXIS_ENTITIES) * 2 * 3 * len(FULL_PANEL_LOWACC_LEVELS)
    assert len(cells) == expected
    assert expected == 144


def test_full_panel_uniqueness() -> None:
    cells = list(iter_full_panel_cells())
    shorts = [c.short() for c in cells]
    assert len(set(shorts)) == len(shorts), "duplicate cells in full panel"


def test_full_panel_deterministic_ordering() -> None:
    """Same iterator call must produce same ordering — required for array indices."""
    a = [c.short() for c in iter_full_panel_cells()]
    b = [c.short() for c in iter_full_panel_cells()]
    assert a == b


def test_full_panel_only_lowacc() -> None:
    cells = list(iter_full_panel_cells())
    assert all(c.axis == "lowacc" for c in cells)
    assert {c.level for c in cells} == set(FULL_PANEL_LOWACC_LEVELS)


def test_cell_at_index_bounds() -> None:
    cells = list(iter_full_panel_cells())
    assert cell_at_index(cells, 0) == cells[0]
    assert cell_at_index(cells, len(cells) - 1) == cells[-1]
    with pytest.raises(IndexError):
        cell_at_index(cells, len(cells))
    with pytest.raises(IndexError):
        cell_at_index(cells, -1)
