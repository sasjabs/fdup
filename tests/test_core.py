"""Smoke tests for fdup._core (Phase 1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridMeta, GridType
from fdup._core.validation import check_aligned
from fdup._core.geodesy import get_cell_areas
from fdup._core.slicing import cell_slice
from fdup._core.warmup import warmup, _FLOWACC_DTYPES

# A small geographic-style transform: 0.01° pixels, origin at (0°E, 10°N).
GEO_TRANSFORM = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)


# ---------------------------------------------------------------------------
# Grid.create — dtype / nodata enforcement per GridType
# ---------------------------------------------------------------------------

class TestGridCreate:
    def _arr(self, dtype, shape=(4, 4)):
        return np.ones(shape, dtype=dtype)

    def test_flowdir_cast_to_uint8(self):
        arr = self._arr(np.int16)
        g = Grid.create(arr, GridType.FlowDir, GEO_TRANSFORM)
        assert g.array.dtype == np.uint8
        assert g.meta.nodata == 255

    def test_flowdir_nodata_remapped(self):
        arr = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        g = Grid.create(arr, GridType.FlowDir, GEO_TRANSFORM, nodata=0)
        # Cells that were 0 should be remapped to 255
        assert g.array[0, 1] == 255
        assert g.array[1, 0] == 255
        assert g.meta.nodata == 255

    def test_mask_cast_to_bool(self):
        arr = np.array([[0, 1], [2, 0]], dtype=np.int32)
        g = Grid.create(arr, GridType.Mask, GEO_TRANSFORM)
        assert g.array.dtype == np.bool_
        assert g.array[0, 1] is np.bool_(True)
        assert g.array[0, 0] is np.bool_(False)
        assert g.meta.nodata is None

    def test_tree_cast_to_uint32(self):
        arr = self._arr(np.int64)
        g = Grid.create(arr, GridType.Tree, GEO_TRANSFORM)
        assert g.array.dtype == np.uint32
        assert g.meta.nodata == 0

    def test_flowacc_keeps_int32(self):
        arr = self._arr(np.int32)
        g = Grid.create(arr, GridType.FlowAcc, GEO_TRANSFORM)
        assert g.array.dtype == np.int32
        assert g.meta.nodata == np.iinfo(np.int32).max

    def test_flowacc_keeps_float32(self):
        arr = self._arr(np.float32)
        g = Grid.create(arr, GridType.FlowAcc, GEO_TRANSFORM)
        assert g.array.dtype == np.float32
        assert math.isnan(g.meta.nodata)

    def test_flowacc_rejects_bad_dtype(self):
        arr = self._arr(np.int8)
        with pytest.raises(TypeError):
            Grid.create(arr, GridType.FlowAcc, GEO_TRANSFORM)

    def test_dem_keeps_float64(self):
        arr = self._arr(np.float64)
        g = Grid.create(arr, GridType.DEM, GEO_TRANSFORM)
        assert g.array.dtype == np.float64
        assert math.isnan(g.meta.nodata)

    def test_dem_keeps_int32(self):
        arr = self._arr(np.int32)
        g = Grid.create(arr, GridType.DEM, GEO_TRANSFORM)
        assert g.array.dtype == np.int32

    def test_dem_rejects_uint8(self):
        arr = self._arr(np.uint8)
        with pytest.raises(TypeError):
            Grid.create(arr, GridType.DEM, GEO_TRANSFORM)

    def test_shape_property(self):
        arr = np.zeros((6, 8), dtype=np.float32)
        g = Grid.create(arr, GridType.FlowAcc, GEO_TRANSFORM)
        assert g.shape == (6, 8)


# ---------------------------------------------------------------------------
# check_aligned
# ---------------------------------------------------------------------------

class TestCheckAligned:
    def _make_coarse_fine(self, k=4, fine_shape=(8, 8), fine_origin=(0.0, 10.0)):
        """Return (coarse, fine) grids where coarse pixel = k × fine pixel."""
        lon0, lat0 = fine_origin
        fine_transform = Affine(0.01, 0.0, lon0, 0.0, -0.01, lat0)
        coarse_transform = Affine(k * 0.01, 0.0, lon0, 0.0, -k * 0.01, lat0)

        fine = Grid.create(
            np.zeros(fine_shape, dtype=np.uint8),
            GridType.FlowDir,
            fine_transform,
        )
        coarse_shape = (fine_shape[0] // k, fine_shape[1] // k)
        coarse = Grid.create(
            np.zeros(coarse_shape, dtype=np.uint8),
            GridType.FlowDir,
            coarse_transform,
        )
        return coarse, fine

    def test_aligned_returns_correct_tuple(self):
        coarse, fine = self._make_coarse_fine(k=4, fine_shape=(8, 8))
        result = check_aligned(coarse, fine)
        assert result == (4, 4, 0, 0)

    def test_aligned_k2(self):
        coarse, fine = self._make_coarse_fine(k=2, fine_shape=(8, 8))
        kx, ky, off_r, off_c = check_aligned(coarse, fine)
        assert (kx, ky) == (2, 2)
        assert (off_r, off_c) == (0, 0)

    def test_aligned_with_nonzero_offset(self):
        # Fine grid starts 2 fine pixels left/above the coarse grid origin.
        k = 4
        fine_transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)
        # Coarse origin is offset by 2 fine pixels in both directions.
        coarse_transform = Affine(k * 0.01, 0.0, 2 * 0.01, 0.0, -k * 0.01, 10.0 - 2 * 0.01)

        fine = Grid.create(
            np.zeros((16, 16), dtype=np.uint8), GridType.FlowDir, fine_transform
        )
        coarse = Grid.create(
            np.zeros((3, 3), dtype=np.uint8), GridType.FlowDir, coarse_transform
        )
        kx, ky, off_r, off_c = check_aligned(coarse, fine)
        assert kx == 4
        assert ky == 4
        assert off_r == 2
        assert off_c == 2

    def test_non_integer_ratio_raises(self):
        fine_transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)
        # Ratio 3.5 — not an integer
        coarse_transform = Affine(0.035, 0.0, 0.0, 0.0, -0.035, 10.0)
        fine = Grid.create(np.zeros((8, 8), dtype=np.uint8), GridType.FlowDir, fine_transform)
        coarse = Grid.create(np.zeros((2, 2), dtype=np.uint8), GridType.FlowDir, coarse_transform)
        with pytest.raises(ValueError):
            check_aligned(coarse, fine)

    def test_misaligned_origin_raises(self):
        fine_transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)
        # Origin shifted by half a fine pixel — not on a boundary
        coarse_transform = Affine(0.04, 0.0, 0.005, 0.0, -0.04, 10.0)
        fine = Grid.create(np.zeros((8, 8), dtype=np.uint8), GridType.FlowDir, fine_transform)
        coarse = Grid.create(np.zeros((2, 2), dtype=np.uint8), GridType.FlowDir, coarse_transform)
        with pytest.raises(ValueError):
            check_aligned(coarse, fine)

    def test_negative_offset_raises(self):
        # Coarse origin is north of the fine grid (negative off_r).
        fine_transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)
        coarse_transform = Affine(0.04, 0.0, 0.0, 0.0, -0.04, 10.04)  # lat above fine top
        fine = Grid.create(np.zeros((8, 8), dtype=np.uint8), GridType.FlowDir, fine_transform)
        coarse = Grid.create(np.zeros((2, 2), dtype=np.uint8), GridType.FlowDir, coarse_transform)
        with pytest.raises(ValueError):
            check_aligned(coarse, fine)


# ---------------------------------------------------------------------------
# get_cell_areas
# ---------------------------------------------------------------------------

class TestGetCellAreas:
    def test_geographic_returns_correct_length(self):
        areas = get_cell_areas(GEO_TRANSFORM, 8, geographic=True)
        assert areas.shape == (8,)
        assert areas.dtype == np.float64

    def test_geographic_positive_areas(self):
        areas = get_cell_areas(GEO_TRANSFORM, 8, geographic=True)
        assert np.all(areas > 0)

    def test_geographic_decreases_toward_poles(self):
        # With origin at 10°N and negative dy, rows go southward — area
        # increases as we approach the equator (cosine gets larger).
        # GEO_TRANSFORM: f=10.0, e=-0.01 → lat_centers decrease each row.
        areas = get_cell_areas(GEO_TRANSFORM, 8, geographic=True)
        # Rows go south → cos(lat) increases → area should increase.
        assert np.all(np.diff(areas) > 0), "Areas should increase moving equatorward"

    def test_projected_constant(self):
        # A projected transform with 100 m pixels (metre-based CRS).
        proj_transform = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 0.0)
        areas = get_cell_areas(proj_transform, 10, geographic=False)
        assert areas.shape == (10,)
        assert np.all(areas == areas[0]), "Projected areas must be constant across rows"
        # 100 × 100 = 10 000 CRS units² (m² for a metre-based CRS; no /1e6 conversion)
        assert math.isclose(areas[0], 10_000.0, rel_tol=1e-9)

    def test_geographic_vs_projected_differ_at_midlat(self):
        geo = get_cell_areas(GEO_TRANSFORM, 4, geographic=True)
        # Same pixel size but interpreted as metres
        proj_transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.0)
        proj = get_cell_areas(proj_transform, 4, geographic=False)
        assert not np.allclose(geo, proj)


# ---------------------------------------------------------------------------
# cell_slice
# ---------------------------------------------------------------------------

class TestCellSlice:
    def setup_method(self):
        # 6×6 array filled with sequential integers for easy identification
        self.arr = np.arange(36, dtype=np.int32).reshape(6, 6)

    def test_interior_no_fill(self):
        window = cell_slice(self.arr, 2, 2, 3, fill=-1)
        expected = self.arr[2:5, 2:5]
        np.testing.assert_array_equal(window, expected)

    def test_interior_is_view(self):
        window = cell_slice(self.arr, 1, 1, 2, fill=0)
        # Interior result must share memory (view)
        assert np.shares_memory(window, self.arr)

    def test_top_left_corner_negative_indices(self):
        # i=-1, j=-1: top-left 3×3 window starts 1 row/col outside
        window = cell_slice(self.arr, -1, -1, 3, fill=99)
        assert window.shape == (3, 3)
        assert window[0, 0] == 99  # fully outside
        assert window[0, 1] == 99  # still first row, outside
        assert window[0, 2] == 99  # still first row, outside
        assert window[1, 0] == 99  # first col, outside
        assert window[1, 1] == self.arr[0, 0]  # first valid cell
        assert window[2, 1] == self.arr[1, 0]

    def test_bottom_right_overflow(self):
        # Start at (4, 4) with k=3 → window [4:7, 4:7] overflows the 6×6 array
        window = cell_slice(self.arr, 4, 4, 3, fill=-1)
        assert window.shape == (3, 3)
        # Only [4:6, 4:6] is valid (2×2 in the top-left corner of the window)
        np.testing.assert_array_equal(window[:2, :2], self.arr[4:6, 4:6])
        # The rest is fill
        assert window[2, 0] == -1
        assert window[0, 2] == -1

    def test_fully_outside_all_fill(self):
        # i=-5, j=-5, k=2 — entirely outside (can't reach row/col 0)
        window = cell_slice(self.arr, -5, -5, 2, fill=7)
        assert window.shape == (2, 2)
        np.testing.assert_array_equal(window, np.full((2, 2), 7, dtype=np.int32))

    def test_fill_dtype_cast(self):
        arr = np.zeros((4, 4), dtype=np.float32)
        window = cell_slice(arr, -1, 0, 3, fill=999)
        assert window.dtype == np.float32
        assert window[0, 0] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# warmup (trivial smoke — it is a no-op in Phase 1)
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_runs_without_error(self):
        warmup()

    def test_flowacc_dtypes_tuple(self):
        assert len(_FLOWACC_DTYPES) == 6
        expected = {np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64}
        assert {np.dtype(d) for d in _FLOWACC_DTYPES} == {np.dtype(t) for t in expected}
