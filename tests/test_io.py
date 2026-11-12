"""Smoke tests for fdup.io (Phase 2): round-trip write→read for every GridType."""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.io import read, write

TRANSFORM = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(grid_type: GridType, dtype, value=5) -> Grid:
    """Build a 4×4 synthetic Grid of the given type and dtype."""
    if grid_type == GridType.FlowDir:
        arr = np.full((4, 4), 4, dtype=np.uint8)   # code 4 = S
    elif grid_type == GridType.Mask:
        arr = np.ones((4, 4), dtype=np.bool_)
    elif grid_type == GridType.Tree:
        arr = np.full((4, 4), value, dtype=np.uint32)
    else:
        arr = np.full((4, 4), value, dtype=dtype)

    return Grid.create(array=arr, type=grid_type, transform=TRANSFORM, crs=None)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def _check(self, grid: Grid, tmp_path, grid_type: GridType):
        p = tmp_path / f"{grid_type.name}.tif"
        write(grid, p)
        g2 = read(p, grid_type=grid_type)

        assert g2.meta.type == grid_type
        assert g2.array.dtype == grid.array.dtype, (
            f"dtype mismatch: wrote {grid.array.dtype}, read back {g2.array.dtype}"
        )
        assert g2.meta.transform == grid.meta.transform

        # nodata check (NaN-aware)
        if grid.meta.nodata is None:
            assert g2.meta.nodata is None
        elif isinstance(grid.meta.nodata, float) and math.isnan(grid.meta.nodata):
            assert isinstance(g2.meta.nodata, float) and math.isnan(g2.meta.nodata)
        else:
            assert g2.meta.nodata == grid.meta.nodata

        # array equality (NaN-aware for floats)
        if np.issubdtype(grid.array.dtype, np.floating):
            np.testing.assert_array_equal(grid.array, g2.array, strict=False)
        else:
            np.testing.assert_array_equal(grid.array, g2.array)

    def test_flowdir_roundtrip(self, tmp_path):
        g = _make_grid(GridType.FlowDir, np.uint8)
        self._check(g, tmp_path, GridType.FlowDir)

    def test_mask_roundtrip(self, tmp_path):
        g = _make_grid(GridType.Mask, np.bool_)
        self._check(g, tmp_path, GridType.Mask)

    def test_tree_roundtrip(self, tmp_path):
        g = _make_grid(GridType.Tree, np.uint32)
        self._check(g, tmp_path, GridType.Tree)

    def test_flowacc_uint32_roundtrip(self, tmp_path):
        g = _make_grid(GridType.FlowAcc, np.uint32)
        self._check(g, tmp_path, GridType.FlowAcc)

    def test_flowacc_int32_roundtrip(self, tmp_path):
        g = _make_grid(GridType.FlowAcc, np.int32)
        self._check(g, tmp_path, GridType.FlowAcc)

    def test_flowacc_float32_roundtrip(self, tmp_path):
        arr = np.full((4, 4), 3.14, dtype=np.float32)
        g = Grid.create(array=arr, type=GridType.FlowAcc, transform=TRANSFORM, crs=None)
        p = tmp_path / "flowacc_f32.tif"
        write(g, p)
        g2 = read(p, grid_type=GridType.FlowAcc)
        assert g2.array.dtype == np.float32
        np.testing.assert_allclose(g2.array, arr, rtol=1e-5)

    def test_dem_int16_roundtrip(self, tmp_path):
        arr = np.full((4, 4), 100, dtype=np.int16)
        g = Grid.create(array=arr, type=GridType.DEM, transform=TRANSFORM, crs=None)
        p = tmp_path / "dem_i16.tif"
        write(g, p)
        g2 = read(p, grid_type=GridType.DEM)
        assert g2.array.dtype == np.int16
        np.testing.assert_array_equal(g2.array, arr)

    def test_dem_float64_roundtrip(self, tmp_path):
        arr = np.full((4, 4), 250.5, dtype=np.float64)
        g = Grid.create(array=arr, type=GridType.DEM, transform=TRANSFORM, crs=None)
        p = tmp_path / "dem_f64.tif"
        write(g, p)
        g2 = read(p, grid_type=GridType.DEM)
        assert g2.array.dtype == np.float64
        np.testing.assert_allclose(g2.array, arr, rtol=1e-9)


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------

class TestOverwriteProtection:

    def test_write_refuses_overwrite_by_default(self, tmp_path):
        g = _make_grid(GridType.FlowDir, np.uint8)
        p = tmp_path / "fd.tif"
        write(g, p)
        with pytest.raises(FileExistsError):
            write(g, p)  # second write without overwrite=True

    def test_write_allows_overwrite_when_flag_set(self, tmp_path):
        g = _make_grid(GridType.FlowDir, np.uint8)
        p = tmp_path / "fd.tif"
        write(g, p)
        write(g, p, overwrite=True)  # must not raise


# ---------------------------------------------------------------------------
# Multi-band rejection
# ---------------------------------------------------------------------------

class TestMultiBandRejection:

    def test_read_raises_on_multiband(self, tmp_path):
        p = tmp_path / "multiband.tif"
        profile = {
            "driver": "GTiff",
            "count": 2,
            "height": 4,
            "width": 4,
            "dtype": "uint8",
            "transform": TRANSFORM,
            "crs": None,
        }
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(np.zeros((4, 4), dtype=np.uint8), 1)
            dst.write(np.zeros((4, 4), dtype=np.uint8), 2)

        with pytest.raises(ValueError, match="single-band"):
            read(p, grid_type=GridType.FlowDir)


def _write_raw(path, array: np.ndarray, nodata) -> None:
    profile = {
        "driver": "GTiff",
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": array.dtype.name,
        "transform": TRANSFORM,
        "crs": None,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


class TestReadNodataRemap:
    """Nodata remap must happen before a narrowing cast wraps the sentinel."""

    def test_flowdir_uint8_nodata_remapped_in_place(self, tmp_path):
        src = np.array([[1, 0], [4, 255]], dtype=np.uint8)
        p = tmp_path / "fd_u8.tif"
        _write_raw(p, src, nodata=0)
        g = read(p, grid_type=GridType.FlowDir)
        np.testing.assert_array_equal(
            g.array, np.array([[1, 255], [4, 255]], dtype=np.uint8)
        )
        assert g.meta.nodata == 255

    def test_flowdir_int16_nodata_masked_before_cast(self, tmp_path):
        # 256 wraps to 0 in uint8. A real sink (0) must stay 0; only the
        # file-nodata pixel becomes the canonical sentinel 255.
        src = np.array([[1, 256], [0, 4]], dtype=np.int16)
        p = tmp_path / "fd_i16.tif"
        _write_raw(p, src, nodata=256)
        g = read(p, grid_type=GridType.FlowDir)
        np.testing.assert_array_equal(
            g.array, np.array([[1, 255], [0, 4]], dtype=np.uint8)
        )

    def test_strahler_int16_nodata_masked_before_cast(self, tmp_path):
        # 257 wraps to 1. The valid order-1 cell must stay 1.
        src = np.array([[1, 257], [2, 3]], dtype=np.int16)
        p = tmp_path / "st_i16.tif"
        _write_raw(p, src, nodata=257)
        g = read(p, grid_type=GridType.Strahler)
        np.testing.assert_array_equal(
            g.array, np.array([[1, 0], [2, 3]], dtype=np.uint8)
        )
        assert g.meta.nodata == 0

    def test_mask_nodata_cleared_without_extra_cast(self, tmp_path):
        src = np.array([[0, 1], [255, 2]], dtype=np.uint8)
        p = tmp_path / "mask.tif"
        _write_raw(p, src, nodata=255)
        g = read(p, grid_type=GridType.Mask)
        np.testing.assert_array_equal(
            g.array, np.array([[False, True], [False, True]])
        )
        assert g.array.dtype == np.bool_
        assert g.meta.nodata is None
