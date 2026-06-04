"""End-to-end pipeline tests for fdup.utils (Phase 3, Step 3.9)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.utils import (
    d8,
    delineate_watershed,
    disaggregate_mask,
    flow_accumulation,
    mask_area,
    match_grids,
    river_tree,
    snap_pour_cell,
)

# 1° pixels, origin at (0°E, 50°N) — valid geographic coordinates throughout,
# so mask_area and geodesy helpers return positive areas.  spherical=False is
# still used for D8 so results are fully deterministic.
TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 50.0)

# 4×4 float32 DEM sloping toward the SE corner (row=3, col=3).
# For every interior cell the steepest D8 descent is SE (diagonal slope
# > cardinal slope), so all 16 cells ultimately drain to (3,3).
_DEM_ARR = np.array(
    [
        [6.0, 5.0, 4.0, 3.0],
        [5.0, 4.0, 3.0, 2.0],
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# World-coordinate centre of cell (3, 3):  x = 0 + 3.5*1 = 3.5,  y = 50 + 3.5*(-1) = 46.5
_MOUTH_X = 3.5
_MOUTH_Y = 46.5
_CELL_SIZE = 1.0  # degrees (= pixel spacing = search radius of exactly one cell)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dem() -> Grid:
    return Grid.create(
        array=_DEM_ARR.copy(),
        type=GridType.DEM,
        transform=TRANSFORM,
        crs=None,
        nodata=float("nan"),
    )


# ---------------------------------------------------------------------------
# Main end-to-end pipeline test
# ---------------------------------------------------------------------------


def test_utils_pipeline() -> None:
    """4×4 synthetic-DEM pipeline: d8 → flowacc → snap → watershed → tree."""

    # ------------------------------------------------------------------
    # 1. Build the DEM Grid
    # ------------------------------------------------------------------
    dem = _make_dem()

    # ------------------------------------------------------------------
    # 2. D8 flow directions (projected-style kernel for determinism)
    # ------------------------------------------------------------------
    fd = d8(dem, spherical=False)

    assert fd.meta.type == GridType.FlowDir
    assert fd.array.dtype == np.uint8
    assert fd.shape == (4, 4)

    # ------------------------------------------------------------------
    # 3. Flow accumulation (cell-count mode)
    # ------------------------------------------------------------------
    fa = flow_accumulation(fd, area=False)

    assert fa.array.dtype == np.uint32, (
        f"expected uint32 for small grid, got {fa.array.dtype}"
    )
    assert fa.array[3, 3] == 16, (
        f"all 16 cells should drain to (3,3), got {fa.array[3, 3]}"
    )

    # ------------------------------------------------------------------
    # 4. Snap pour cell — should land on (3, 3)
    # ------------------------------------------------------------------
    r, c = snap_pour_cell(fa, x=_MOUTH_X, y=_MOUTH_Y, radius=_CELL_SIZE)

    assert (r, c) == (3, 3), f"snapped pour cell should be (3,3), got ({r},{c})"

    # ------------------------------------------------------------------
    # 5. Delineate watershed — the entire 4×4 grid is the basin
    # ------------------------------------------------------------------
    ws = delineate_watershed(fd, r, c)

    assert ws.meta.type == GridType.Mask
    assert ws.array.dtype == np.bool_
    assert ws.array.all(), "all 16 cells should be in the watershed"

    # ------------------------------------------------------------------
    # 6. mask_area — must be positive and match the geographic expectation
    # ------------------------------------------------------------------
    area_km2 = mask_area(ws)

    assert area_km2 > 0, f"expected positive area, got {area_km2}"

    # Independent computation using the spherical-trapezoid formula.
    _EARTH_R = 6_371_000.0
    t = TRANSFORM
    dy_m = abs(t.e) * math.pi / 180.0 * _EARTH_R
    expected_area = 0.0
    for row in range(4):
        lat = t.f + (row + 0.5) * t.e
        dx_m = abs(t.a) * math.pi / 180.0 * _EARTH_R * math.cos(math.radians(lat))
        expected_area += 4 * dx_m * dy_m / 1e6  # 4 True cells per row
    assert math.isclose(area_km2, expected_area, rel_tol=1e-10), (
        f"area {area_km2:.4f} km² does not match expected {expected_area:.4f} km²"
    )

    # ------------------------------------------------------------------
    # 7. disaggregate_mask — doubles the resolution
    # ------------------------------------------------------------------
    ws_fine = disaggregate_mask(ws, k=2)

    assert ws_fine.shape == (8, 8), f"expected (8,8), got {ws_fine.shape}"
    assert ws_fine.array.all(), "all pixels should remain True after disaggregation"
    assert math.isclose(ws_fine.meta.transform.a, t.a / 2, rel_tol=1e-12), (
        "transform.a should be halved"
    )
    assert math.isclose(ws_fine.meta.transform.e, t.e / 2, rel_tol=1e-12), (
        "transform.e should be halved"
    )

    # ------------------------------------------------------------------
    # 8. match_grids — identity case: result equals input
    # ------------------------------------------------------------------
    ws_matched = match_grids(reference=ws_fine, other=ws_fine)

    assert ws_matched.shape == ws_fine.shape, (
        f"expected shape {ws_fine.shape}, got {ws_matched.shape}"
    )
    assert np.array_equal(ws_matched.array, ws_fine.array), (
        "match_grids identity should produce an identical array"
    )

    # ------------------------------------------------------------------
    # 9. river_tree (no mask)
    # ------------------------------------------------------------------
    tree_grid, seeds = river_tree(fd, fa)

    assert tree_grid.meta.type == GridType.Tree
    assert tree_grid.array.dtype == np.uint32
    assert seeds.dtype.names == (
        "mouth_row",
        "mouth_col",
        "headwater_row",
        "headwater_col",
        "ncells",
        "acc",
        "length_m",
    )
    assert len(seeds) >= 1, "expected at least one river segment"
    assert seeds["length_m"].dtype == np.float64

    # ------------------------------------------------------------------
    # 10. river_tree with mask pruning — re-densification invariant
    # ------------------------------------------------------------------
    tree_grid_m, seeds_m = river_tree(fd, fa, mask=ws)

    assert len(seeds_m) <= len(seeds), (
        "masked river_tree should not add new segments"
    )
    if len(seeds_m) > 0:
        assert tree_grid_m.array.max() == len(seeds_m), (
            "tree IDs must be re-densified: max ID should equal segment count"
        )


# ---------------------------------------------------------------------------
# Edge-case: snap_pour_cell raises when no valid cell is within radius
# ---------------------------------------------------------------------------


def test_snap_pour_cell_no_candidate_raises() -> None:
    """snap_pour_cell raises ValueError when the window is all nodata."""
    arr = np.full((4, 4), np.iinfo(np.uint32).max, dtype=np.uint32)
    fa_nd = Grid.create(
        array=arr,
        type=GridType.FlowAcc,
        transform=TRANSFORM,
        crs=None,
        nodata=int(np.iinfo(np.uint32).max),
    )
    with pytest.raises(ValueError, match="no valid pour cell"):
        snap_pour_cell(fa_nd, x=1.5, y=48.5, radius=0.5)


# ---------------------------------------------------------------------------
# Edge-case: disaggregate_mask rejects k <= 0
# ---------------------------------------------------------------------------


def test_disaggregate_mask_bad_k() -> None:
    dem = _make_dem()
    fd = d8(dem, spherical=False)
    ws = delineate_watershed(fd, pour_row=3, pour_col=3)
    with pytest.raises(ValueError):
        disaggregate_mask(ws, k=0)
    with pytest.raises(ValueError):
        disaggregate_mask(ws, k=-1)
