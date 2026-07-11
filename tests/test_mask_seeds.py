"""Tests for fdup.utils.mask_seeds (tree.py)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.utils import d8, flow_accumulation, river_tree
from fdup.utils.tree import mask_seeds

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Projected CRS-style transform: 1 km pixels, simple origin.
TRANSFORM = Affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0)


def _make_grids(dem_arr: np.ndarray) -> tuple[Grid, Grid]:
    """Return (flowdir, flowacc) grids from a float32 DEM array."""
    dem = Grid.create(
        array=dem_arr.astype(np.float32),
        type=GridType.DEM,
        transform=TRANSFORM,
        crs=None,
        nodata=float("nan"),
    )
    fd = d8(dem, spherical=False)
    fa = flow_accumulation(fd, area=False)
    return fd, fa


def _mask_grid(shape: tuple[int, int], rows: slice, cols: slice) -> Grid:
    """Create a Mask Grid with True only in the given row/col slices."""
    arr = np.zeros(shape, dtype=np.bool_)
    arr[rows, cols] = True
    return Grid.create(
        array=arr,
        type=GridType.Mask,
        transform=TRANSFORM,
        crs=None,
    )


def _full_mask(shape: tuple[int, int]) -> Grid:
    arr = np.ones(shape, dtype=np.bool_)
    return Grid.create(
        array=arr,
        type=GridType.Mask,
        transform=TRANSFORM,
        crs=None,
    )


# ---------------------------------------------------------------------------
# Helper: build a DEM that drains in an S (south) chain
# ---------------------------------------------------------------------------

def _south_chain_dem(nrows: int, ncols: int) -> np.ndarray:
    """DEM with decreasing elevation toward the south (row = nrows-1, col = 0).

    The central column drains straight south; other cells drain eastward or
    southward.  Simple enough that flow direction is fully predictable.
    """
    arr = np.zeros((nrows, ncols), dtype=np.float32)
    for r in range(nrows):
        for c in range(ncols):
            arr[r, c] = (nrows - 1 - r) * 10.0 + (ncols - 1 - c) * 1.0
    return arr


# ---------------------------------------------------------------------------
# 1. Full-mask pass-through: output equals input seeds
# ---------------------------------------------------------------------------


def test_full_mask_preserves_seeds():
    """With an all-True mask, mask_seeds should return at least as many seeds
    as river_tree, all with length_m >= 0 and correct acc values."""
    nrows, ncols = 8, 8
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)

    _tree, seeds = river_tree(fd, fa)
    full = _full_mask((nrows, ncols))

    out = mask_seeds(seeds, fd, fa, full)

    assert out.dtype.names == (
        "mouth_row", "mouth_col",
        "headwater_row", "headwater_col",
        "ncells", "acc", "length_m",
    )
    # With all-True mask every original seed (>= 2 cells) should survive.
    assert len(out) >= len(seeds), (
        f"Expected at least {len(seeds)} sub-seeds, got {len(out)}"
    )
    assert np.all(out["ncells"] >= 2), "every sub-seed must have >= 2 cells"
    assert np.all(out["length_m"] >= 0.0), "length_m must be non-negative"
    assert out["length_m"].dtype == np.float64


# ---------------------------------------------------------------------------
# 2. Empty mask → zero output seeds
# ---------------------------------------------------------------------------


def test_empty_mask_returns_no_seeds():
    nrows, ncols = 6, 6
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)
    _tree, seeds = river_tree(fd, fa)

    empty_mask_arr = np.zeros((nrows, ncols), dtype=np.bool_)
    empty_mask = Grid.create(
        array=empty_mask_arr,
        type=GridType.Mask,
        transform=TRANSFORM,
        crs=None,
    )
    out = mask_seeds(seeds, fd, fa, empty_mask)
    assert len(out) == 0, f"Expected 0 sub-seeds with empty mask, got {len(out)}"


# ---------------------------------------------------------------------------
# 3. Empty seeds array → empty output
# ---------------------------------------------------------------------------


def test_empty_seeds_input():
    nrows, ncols = 4, 4
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)
    full = _full_mask((nrows, ncols))

    # Build an empty seeds array with the correct dtype
    from fdup.utils.tree import _make_seeds_dtype
    empty_seeds = np.empty(0, dtype=_make_seeds_dtype(fa.array.dtype))

    out = mask_seeds(empty_seeds, fd, fa, full)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# 4. Multi-entry/exit mask: single seed, mask splits into two sub-segments
# ---------------------------------------------------------------------------


def test_split_mask_two_subsegments():
    """A seed spanning 6 cells is masked so that only cells 0-1 and 4-5 are
    in-mask (with a gap at cells 2-3).  Expect exactly 2 sub-seeds."""

    # 1-D chain of 6 cells draining south: row 0 → 1 → 2 → 3 → 4 → 5
    nrows = 7   # extra row for sink
    ncols = 1

    dem_arr = np.zeros((nrows, ncols), dtype=np.float32)
    for r in range(nrows):
        dem_arr[r, 0] = float(nrows - 1 - r) * 10.0

    fd, fa = _make_grids(dem_arr)

    # river_tree should produce one segment spanning rows 0..5 (at least)
    _tree, seeds = river_tree(fd, fa)
    assert len(seeds) >= 1, "Need at least one seed for this test"

    # Mask: rows 0,1 True; rows 2,3 False; rows 4,5 True; row 6 False
    mask_arr = np.array([[True], [True], [False], [False], [True], [True], [False]])
    assert mask_arr.shape == (nrows, ncols)
    split_mask = Grid.create(
        array=mask_arr,
        type=GridType.Mask,
        transform=TRANSFORM,
        crs=None,
    )

    out = mask_seeds(seeds, fd, fa, split_mask)

    # We expect exactly 2 sub-seeds (one per contiguous in-mask block of >= 2 cells)
    assert len(out) == 2, (
        f"Expected 2 sub-seeds from split mask, got {len(out)}: {out}"
    )

    # The first sub-seed (most upstream run) has its mouth in rows 0-1
    # and the second sub-seed's mouth in rows 4-5.
    # sub-seeds are in emission order (upstream to downstream):
    mouth_rows = sorted(int(r) for r in out["mouth_row"])
    # Both sub-seeds should have ncells = 2
    assert np.all(out["ncells"] == 2), f"Expected ncells=2, got {out['ncells']}"
    # acc at each mouth should be read from flowacc
    for i in range(len(out)):
        mr = int(out["mouth_row"][i])
        mc = int(out["mouth_col"][i])
        assert out["acc"][i] == fa.array[mr, mc], (
            f"acc mismatch for sub-seed {i}: expected {fa.array[mr, mc]}, got {out['acc'][i]}"
        )

    # length_m for each 2-cell run is one step distance (1000 m for S direction,
    # projected CRS with 1000 m pixel spacing, so step dist S = abs(e) = 1000)
    for i in range(len(out)):
        assert out["length_m"][i] > 0.0, "length_m must be positive"


# ---------------------------------------------------------------------------
# 5. Validation errors
# ---------------------------------------------------------------------------


def test_mask_seeds_wrong_type_flowdir():
    nrows, ncols = 4, 4
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)
    full = _full_mask((nrows, ncols))
    _tree, seeds = river_tree(fd, fa)

    # Pass flowacc where flowdir expected
    with pytest.raises(ValueError):
        mask_seeds(seeds, fa, fa, full)


def test_mask_seeds_wrong_type_mask():
    nrows, ncols = 4, 4
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)
    _tree, seeds = river_tree(fd, fa)

    # Pass flowacc where mask expected
    with pytest.raises(ValueError):
        mask_seeds(seeds, fd, fa, fa)


def test_mask_seeds_missing_seed_fields():
    nrows, ncols = 4, 4
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)
    full = _full_mask((nrows, ncols))

    bad_seeds = np.array([(1, 2)], dtype=[("mouth_row", np.uint32), ("foo", np.uint32)])
    with pytest.raises(ValueError, match="missing required fields"):
        mask_seeds(bad_seeds, fd, fa, full)


def test_mask_seeds_shape_mismatch():
    nrows, ncols = 4, 4
    dem_arr = _south_chain_dem(nrows, ncols)
    fd, fa = _make_grids(dem_arr)
    _tree, seeds = river_tree(fd, fa)

    # Mask of wrong shape
    wrong_mask = Grid.create(
        array=np.ones((3, 3), dtype=np.bool_),
        type=GridType.Mask,
        transform=TRANSFORM,
        crs=None,
    )
    with pytest.raises(ValueError):
        mask_seeds(seeds, fd, fa, wrong_mask)
