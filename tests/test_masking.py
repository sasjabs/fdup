"""Tests for fdup.utils.masking (masking toolkit).

Covers:
  - disaggregate_mask (moved from watershed.py)
  - mask_area (moved from watershed.py)
  - threshold_mask (FlowAcc and Strahler inputs)
  - mask_grid (any grid type)
  - crop_grid (integer, float, and Mask grids)
  - CLI subcommands threshold-mask, mask-grid, crop-grid
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.utils.masking import (
    crop_grid,
    disaggregate_mask,
    mask_area,
    mask_grid,
    threshold_mask,
)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

# 1° pixels at ~45°N — geographic CRS (None treated as geographic).
TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 50.0)


def _mask(arr: np.ndarray, transform: Affine = TRANSFORM) -> Grid:
    return Grid.create(
        array=arr.astype(np.bool_),
        type=GridType.Mask,
        transform=transform,
        crs=None,
    )


def _flowacc(arr: np.ndarray, nodata=None, transform: Affine = TRANSFORM) -> Grid:
    g = Grid.create(
        array=arr,
        type=GridType.FlowAcc,
        transform=transform,
        crs=None,
    )
    if nodata is not None:
        from fdup._core.types import GridMeta
        meta = GridMeta(
            type=g.meta.type,
            transform=g.meta.transform,
            crs=g.meta.crs,
            nodata=nodata,
        )
        return Grid(array=g.array, meta=meta)
    return g


def _strahler(arr: np.ndarray, transform: Affine = TRANSFORM) -> Grid:
    return Grid.create(
        array=arr.astype(np.uint8),
        type=GridType.Strahler,
        transform=transform,
        crs=None,
    )


def _dem(arr: np.ndarray, nodata=None, transform: Affine = TRANSFORM) -> Grid:
    g = Grid.create(
        array=arr,
        type=GridType.DEM,
        transform=transform,
        crs=None,
    )
    if nodata is not None:
        from fdup._core.types import GridMeta
        meta = GridMeta(
            type=g.meta.type,
            transform=g.meta.transform,
            crs=g.meta.crs,
            nodata=nodata,
        )
        return Grid(array=g.array, meta=meta)
    return g


# ---------------------------------------------------------------------------
# disaggregate_mask
# ---------------------------------------------------------------------------


class TestDisaggregateMask:

    def test_shape_doubled(self):
        arr = np.ones((3, 4), dtype=np.bool_)
        out = disaggregate_mask(_mask(arr), k=2)
        assert out.shape == (6, 8)

    def test_all_true_stays_true(self):
        arr = np.ones((2, 2), dtype=np.bool_)
        out = disaggregate_mask(_mask(arr), k=3)
        assert out.array.all()

    def test_false_cell_expands_to_block(self):
        arr = np.array([[True, False], [True, True]], dtype=np.bool_)
        out = disaggregate_mask(_mask(arr), k=2)
        # top-right 2×2 block should be False
        assert not out.array[0:2, 2:4].any()
        # other three 2×2 blocks should be True
        assert out.array[0:2, 0:2].all()

    def test_transform_pixel_spacing_halved(self):
        arr = np.ones((4, 4), dtype=np.bool_)
        t = Affine(0.5, 0.0, 10.0, 0.0, -0.5, 45.0)
        out = disaggregate_mask(_mask(arr, transform=t), k=2)
        assert math.isclose(out.meta.transform.a, 0.25)
        assert math.isclose(out.meta.transform.e, -0.25)

    def test_transform_origin_unchanged(self):
        arr = np.ones((4, 4), dtype=np.bool_)
        t = Affine(1.0, 0.0, 5.0, 0.0, -1.0, 52.0)
        out = disaggregate_mask(_mask(arr, transform=t), k=4)
        assert math.isclose(out.meta.transform.c, t.c)
        assert math.isclose(out.meta.transform.f, t.f)

    def test_output_type_is_mask(self):
        arr = np.ones((2, 2), dtype=np.bool_)
        out = disaggregate_mask(_mask(arr), k=2)
        assert out.meta.type == GridType.Mask

    def test_bad_k_zero_raises(self):
        arr = np.ones((2, 2), dtype=np.bool_)
        with pytest.raises(ValueError):
            disaggregate_mask(_mask(arr), k=0)

    def test_bad_k_negative_raises(self):
        arr = np.ones((2, 2), dtype=np.bool_)
        with pytest.raises(ValueError):
            disaggregate_mask(_mask(arr), k=-1)

    def test_bad_k_float_raises(self):
        arr = np.ones((2, 2), dtype=np.bool_)
        with pytest.raises(ValueError):
            disaggregate_mask(_mask(arr), k=2.0)  # type: ignore[arg-type]

    def test_wrong_input_type_raises(self):
        arr = np.ones((3, 3), dtype=np.float32)
        fa = _flowacc(arr)
        with pytest.raises(ValueError):
            disaggregate_mask(fa, k=2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# mask_area
# ---------------------------------------------------------------------------


class TestMaskArea:

    def test_all_true_positive_area(self):
        arr = np.ones((4, 4), dtype=np.bool_)
        area = mask_area(_mask(arr))
        assert area > 0

    def test_all_false_zero_area(self):
        arr = np.zeros((4, 4), dtype=np.bool_)
        area = mask_area(_mask(arr))
        assert area == 0.0

    def test_area_proportional_to_true_count(self):
        """For a uniform-latitude transform, area ∝ number of True cells."""
        # Use projected (non-geographic) transform so all cells have equal area.
        t = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 0.0)
        arr_full = np.ones((4, 4), dtype=np.bool_)
        arr_half = arr_full.copy()
        arr_half[:, 2:] = False

        from fdup._core.types import GridMeta
        def _proj_mask(a):
            try:
                from rasterio.crs import CRS
                crs = CRS.from_epsg(32632)
            except Exception:
                crs = None
            meta = GridMeta(
                type=GridType.Mask,
                transform=t,
                crs=crs,
                nodata=None,
            )
            return Grid(array=a.astype(np.bool_), meta=meta)

        full_area = mask_area(_proj_mask(arr_full))
        half_area = mask_area(_proj_mask(arr_half))
        assert math.isclose(half_area, full_area / 2, rel_tol=1e-9)

    def test_wrong_input_type_raises(self):
        arr = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            mask_area(_flowacc(arr))


# ---------------------------------------------------------------------------
# threshold_mask
# ---------------------------------------------------------------------------


class TestThresholdMask:

    def test_flowacc_basic(self):
        arr = np.array([[1, 5, 10], [2, 7, 3]], dtype=np.uint32)
        out = threshold_mask(_flowacc(arr), cutoff=5.0)
        expected = np.array([[False, True, True], [False, True, False]])
        np.testing.assert_array_equal(out.array, expected)

    def test_strahler_basic(self):
        arr = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.uint8)
        out = threshold_mask(_strahler(arr), cutoff=2.0)
        expected = np.array([[False, False, True], [False, True, False]])
        np.testing.assert_array_equal(out.array, expected)

    def test_output_type_is_mask(self):
        arr = np.ones((3, 3), dtype=np.uint32)
        out = threshold_mask(_flowacc(arr), cutoff=1.0)
        assert out.meta.type == GridType.Mask

    def test_transform_and_crs_preserved(self):
        arr = np.ones((3, 4), dtype=np.uint32)
        g = _flowacc(arr)
        out = threshold_mask(g, cutoff=1.0)
        assert out.meta.transform == g.meta.transform
        assert out.meta.crs == g.meta.crs

    def test_strahler_nodata_zero_is_false(self):
        """Strahler nodata=0 should map to False even at cutoff=0."""
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        out = threshold_mask(_strahler(arr), cutoff=0.0)
        # nodata=0 → False; value 1 >= 0 but nodata wins; value 2 >= 0
        # Strahler nodata=0: cell with value 0 is nodata → False
        assert not out.array[0, 0]

    def test_float_flowacc_nan_is_false(self):
        arr = np.array([[float("nan"), 5.0, 10.0]], dtype=np.float32)
        fa = Grid.create(
            array=arr,
            type=GridType.FlowAcc,
            transform=TRANSFORM,
            crs=None,
        )
        out = threshold_mask(fa, cutoff=5.0)
        assert not out.array[0, 0]  # NaN → False
        assert out.array[0, 1]      # 5.0 >= 5.0 → True
        assert out.array[0, 2]      # 10.0 >= 5.0 → True

    def test_cutoff_equal_to_value_is_true(self):
        arr = np.array([[4, 5, 6]], dtype=np.uint32)
        out = threshold_mask(_flowacc(arr), cutoff=5.0)
        assert not out.array[0, 0]
        assert out.array[0, 1]
        assert out.array[0, 2]

    def test_wrong_type_raises(self):
        arr = np.ones((3, 3), dtype=np.uint8)
        fd = Grid.create(
            array=arr,
            type=GridType.FlowDir,
            transform=TRANSFORM,
            crs=None,
        )
        with pytest.raises(ValueError):
            threshold_mask(fd, cutoff=1.0)

    def test_all_below_cutoff_all_false(self):
        arr = np.ones((4, 4), dtype=np.uint32)
        out = threshold_mask(_flowacc(arr), cutoff=100.0)
        assert not out.array.any()


# ---------------------------------------------------------------------------
# mask_grid
# ---------------------------------------------------------------------------


class TestMaskGrid:

    def test_basic_integer_flowacc(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint32)
        mask_arr = np.array([[True, False], [True, False]], dtype=np.bool_)
        fa = _flowacc(arr)
        m = _mask(mask_arr)
        out = mask_grid(fa, m)
        assert out.array[0, 0] == 1
        assert out.array[1, 0] == 3
        # False cells → nodata
        nd = fa.meta.nodata
        assert out.array[0, 1] == nd
        assert out.array[1, 1] == nd

    def test_output_same_type_as_input(self):
        arr = np.ones((3, 3), dtype=np.uint32)
        m = _mask(np.ones((3, 3), dtype=np.bool_))
        out = mask_grid(_flowacc(arr), m)
        assert out.meta.type == GridType.FlowAcc

    def test_mask_grid_on_mask_sets_false(self):
        """mask_grid on a Mask input sets False where the mask is False."""
        arr = np.ones((3, 3), dtype=np.bool_)
        mask_arr = np.array([[True, False, True]] * 3, dtype=np.bool_)
        g = _mask(arr)
        m = _mask(mask_arr)
        out = mask_grid(g, m)
        assert out.meta.type == GridType.Mask
        assert not out.array[:, 1].any()
        assert out.array[:, 0].all()
        assert out.array[:, 2].all()

    def test_shape_mismatch_raises(self):
        arr = np.ones((3, 3), dtype=np.uint32)
        mask_arr = np.ones((4, 4), dtype=np.bool_)
        with pytest.raises(ValueError):
            mask_grid(_flowacc(arr), _mask(mask_arr))

    def test_transform_mismatch_raises(self):
        arr = np.ones((3, 3), dtype=np.uint32)
        t2 = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 50.0)
        mask_arr = np.ones((3, 3), dtype=np.bool_)
        with pytest.raises(ValueError):
            mask_grid(_flowacc(arr), _mask(mask_arr, transform=t2))

    def test_wrong_mask_type_raises(self):
        arr = np.ones((3, 3), dtype=np.uint32)
        fa_as_mask = _flowacc(arr)  # Not a Mask type
        with pytest.raises(ValueError):
            mask_grid(_flowacc(arr), fa_as_mask)  # type: ignore[arg-type]

    def test_all_true_mask_leaves_data_unchanged(self):
        arr = np.array([[10, 20], [30, 40]], dtype=np.uint32)
        m = _mask(np.ones((2, 2), dtype=np.bool_))
        out = mask_grid(_flowacc(arr), m)
        np.testing.assert_array_equal(out.array, arr)

    def test_all_false_mask_sets_all_nodata(self):
        arr = np.array([[10, 20], [30, 40]], dtype=np.uint32)
        fa = _flowacc(arr)
        m = _mask(np.zeros((2, 2), dtype=np.bool_))
        out = mask_grid(fa, m)
        nd = fa.meta.nodata
        assert (out.array == nd).all()


# ---------------------------------------------------------------------------
# crop_grid
# ---------------------------------------------------------------------------


class TestCropGrid:

    def test_mask_grid_cropped_to_true_cells(self):
        arr = np.array(
            [[False, False, False],
             [False, True,  False],
             [False, False, False]],
            dtype=np.bool_,
        )
        g = _mask(arr)
        out = crop_grid(g)
        assert out.shape == (1, 1)
        assert out.array[0, 0] == True  # noqa: E712

    def test_integer_grid_nodata_trimmed(self):
        nd = np.iinfo(np.uint32).max
        arr = np.full((5, 5), nd, dtype=np.uint32)
        arr[1:4, 1:4] = 7  # 3×3 data block
        fa = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None, nodata=int(nd),
        )
        out = crop_grid(fa)
        assert out.shape == (3, 3)
        assert (out.array == 7).all()

    def test_float_grid_nan_trimmed(self):
        arr = np.full((4, 4), float("nan"), dtype=np.float32)
        arr[0, 0] = 1.0
        arr[3, 3] = 2.0
        g = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None,
        )
        out = crop_grid(g)
        assert out.shape == (4, 4)  # corners → full extent

    def test_float_grid_all_nan_raises(self):
        arr = np.full((3, 3), float("nan"), dtype=np.float32)
        g = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None,
        )
        with pytest.raises(ValueError, match="all cells are nodata"):
            crop_grid(g)

    def test_integer_all_nodata_raises(self):
        nd = int(np.iinfo(np.uint32).max)
        arr = np.full((3, 3), nd, dtype=np.uint32)
        g = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None, nodata=nd,
        )
        with pytest.raises(ValueError, match="all cells are nodata"):
            crop_grid(g)

    def test_no_trimming_needed_returns_same(self):
        arr = np.ones((3, 3), dtype=np.uint32)
        g = _flowacc(arr)
        out = crop_grid(g)
        assert out is g

    def test_transform_origin_shifted_correctly(self):
        nd = int(np.iinfo(np.uint32).max)
        arr = np.full((5, 5), nd, dtype=np.uint32)
        arr[2, 3] = 1  # single data cell at row=2, col=3
        g = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None, nodata=nd,
        )
        out = crop_grid(g)
        t = TRANSFORM
        expected_c = t.c + 3 * t.a  # col0=3
        expected_f = t.f + 2 * t.e  # row0=2
        assert math.isclose(out.meta.transform.c, expected_c, abs_tol=1e-12)
        assert math.isclose(out.meta.transform.f, expected_f, abs_tol=1e-12)

    def test_pixel_spacing_unchanged(self):
        nd = int(np.iinfo(np.uint32).max)
        arr = np.full((5, 5), nd, dtype=np.uint32)
        arr[1:4, 1:4] = 1
        g = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None, nodata=nd,
        )
        out = crop_grid(g)
        assert math.isclose(out.meta.transform.a, TRANSFORM.a)
        assert math.isclose(out.meta.transform.e, TRANSFORM.e)

    def test_type_preserved(self):
        nd = int(np.iinfo(np.uint32).max)
        arr = np.full((5, 5), nd, dtype=np.uint32)
        arr[2, 2] = 5
        g = Grid.create(
            array=arr, type=GridType.FlowAcc,
            transform=TRANSFORM, crs=None, nodata=nd,
        )
        out = crop_grid(g)
        assert out.meta.type == GridType.FlowAcc

    def test_mask_all_false_raises(self):
        arr = np.zeros((3, 3), dtype=np.bool_)
        g = _mask(arr)
        with pytest.raises(ValueError, match="all cells are nodata"):
            crop_grid(g)

    def test_dem_with_nodata_trimmed(self):
        nd = -9999
        arr = np.full((6, 6), nd, dtype=np.int32)
        arr[1:5, 2:4] = 100
        g = Grid.create(
            array=arr, type=GridType.DEM,
            transform=TRANSFORM, crs=None, nodata=nd,
        )
        out = crop_grid(g)
        assert out.shape == (4, 2)
        assert (out.array == 100).all()


# ---------------------------------------------------------------------------
# Backwards compatibility: disaggregate_mask / mask_area still in fdup.utils
# ---------------------------------------------------------------------------


def test_public_api_exports():
    """disaggregate_mask, mask_area, threshold_mask, mask_grid, crop_grid are
    all importable from fdup.utils."""
    from fdup import utils  # noqa: F401
    assert hasattr(utils, "disaggregate_mask")
    assert hasattr(utils, "mask_area")
    assert hasattr(utils, "threshold_mask")
    assert hasattr(utils, "mask_grid")
    assert hasattr(utils, "crop_grid")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_threshold_mask_flowacc(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    arr = np.array([[1, 5, 10], [2, 7, 3]], dtype=np.uint32)
    fa = Grid.create(array=arr, type=GridType.FlowAcc, transform=TRANSFORM, crs=None)
    inp = str(tmp_path / "fa.tif")
    fdup_io.write(fa, inp, overwrite=True)
    out = str(tmp_path / "mask.tif")

    main(["threshold-mask", "--flowacc", inp, "--cutoff", "5", "-o", out])

    result = fdup_io.read(out, grid_type=GridType.Mask)
    assert result.meta.type == GridType.Mask
    expected = np.array([[False, True, True], [False, True, False]])
    np.testing.assert_array_equal(result.array, expected)


def test_cli_threshold_mask_strahler(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    arr = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    st = Grid.create(array=arr, type=GridType.Strahler, transform=TRANSFORM, crs=None)
    inp = str(tmp_path / "strahler.tif")
    fdup_io.write(st, inp, overwrite=True)
    out = str(tmp_path / "mask.tif")

    main(["threshold-mask", "--strahler", inp, "--cutoff", "2", "-o", out])

    result = fdup_io.read(out, grid_type=GridType.Mask)
    assert result.meta.type == GridType.Mask
    assert not result.array[0, 0]   # nodata=0
    assert not result.array[0, 1]   # 1 < 2
    assert result.array[0, 2]       # 2 >= 2
    assert result.array[0, 3]       # 3 >= 2


def test_cli_mask_grid(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    arr = np.array([[10, 20], [30, 40]], dtype=np.uint32)
    fa = Grid.create(array=arr, type=GridType.FlowAcc, transform=TRANSFORM, crs=None)
    mask_arr = np.array([[True, False], [True, True]], dtype=np.bool_)
    m = Grid.create(array=mask_arr, type=GridType.Mask, transform=TRANSFORM, crs=None)

    inp_grid = str(tmp_path / "fa.tif")
    inp_mask = str(tmp_path / "mask.tif")
    out = str(tmp_path / "masked.tif")
    fdup_io.write(fa, inp_grid, overwrite=True)
    fdup_io.write(m, inp_mask, overwrite=True)

    main(["mask-grid", "--grid", inp_grid, "--grid-type", "flowacc",
          "--mask", inp_mask, "-o", out])

    result = fdup_io.read(out, grid_type=GridType.FlowAcc)
    assert result.meta.type == GridType.FlowAcc
    assert result.array[0, 0] == 10
    assert result.array[1, 0] == 30
    assert result.array[1, 1] == 40
    nd = fa.meta.nodata
    assert result.array[0, 1] == nd


def test_cli_crop_grid(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    nd = int(np.iinfo(np.uint32).max)
    arr = np.full((5, 5), nd, dtype=np.uint32)
    arr[1:4, 1:4] = 42
    fa = Grid.create(
        array=arr, type=GridType.FlowAcc,
        transform=TRANSFORM, crs=None, nodata=nd,
    )
    inp = str(tmp_path / "fa.tif")
    out = str(tmp_path / "cropped.tif")
    fdup_io.write(fa, inp, overwrite=True)

    main(["crop-grid", "--grid", inp, "--grid-type", "flowacc", "-o", out])

    result = fdup_io.read(out, grid_type=GridType.FlowAcc)
    assert result.shape == (3, 3)
    assert (result.array == 42).all()
