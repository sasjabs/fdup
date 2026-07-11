"""Tests for fdup.utils.strahler_order (Strahler stream orders).

Grids use Esri D8 encoding: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE.
0 = sink, 255 = nodata (all border cells).
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType, _default_nodata
from fdup.utils.strahler import strahler_order


TRANSFORM = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flowdir(arr: np.ndarray) -> Grid:
    return Grid.create(
        array=arr.astype(np.uint8),
        type=GridType.FlowDir,
        transform=TRANSFORM,
        crs=None,
    )


# ---------------------------------------------------------------------------
# Fixture grids (hand-verified)
# ---------------------------------------------------------------------------

# Y-shaped confluence (5×4):
#
#   col:  0    1    2    3
#   row 0: 255 255  255  255
#   row 1: 255   1    4  255    (1,1)→E  (1,2)→S
#   row 2: 255   1    4  255    (2,1)→E  (2,2)→S
#   row 3: 255 255    0  255    sink
#   row 4: 255 255  255  255
#
# Trace:
#   Headwaters: (1,1), (2,1) → order 1
#   (1,2): 1 upstream [(1,1)] → order 1
#   (2,2): 2 order-1 inputs [(1,2), (2,1)] → order 2  (Strahler rule: increment)
_SIMPLE_ARR = np.array(
    [[255, 255, 255, 255],
     [255,   1,   4, 255],
     [255,   1,   4, 255],
     [255, 255,   0, 255],
     [255, 255, 255, 255]],
    dtype=np.uint8,
)
_SIMPLE_EXPECTED = np.array(
    [[0, 0, 0, 0],
     [0, 1, 1, 0],
     [0, 1, 2, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],
    dtype=np.uint8,
)

# Straight E-flowing chain (3×5): single headwater, no confluences → all order 1.
_CHAIN_ARR = np.array(
    [[255, 255, 255, 255, 255],
     [255,   1,   1,   1,   0],
     [255, 255, 255, 255, 255]],
    dtype=np.uint8,
)

# Two Y-shapes draining into a common trunk → order 3 at the confluence (7×6):
#
#   col:  0    1    2    3    4    5
#   row 0: 255 255  255  255  255  255
#   row 1: 255   1    4    1    4  255    HW-L→E, col2→S, HW-R→E, col4→S
#   row 2: 255   1    4    1    4  255    HW-L2→E, col2→S, HW-R2→E, col4→S
#   row 3: 255 255    1    4   16  255    left-trunk→E, confluence→S, right-trunk→W
#   row 4: 255 255  255    4  255  255    (4,3)→S
#   row 5: 255 255  255    0  255  255    sink
#   row 6: 255 255  255  255  255  255
#
# Left branch:
#   (1,1),(2,1) → headwaters order 1
#   (1,2): 1 upstream → order 1
#   (2,2): 2 order-1 inputs → order 2
#   (3,2): 1 order-2 input → order 2; dir=E→(3,3)
# Right branch (symmetric):
#   (1,3),(2,3) → headwaters order 1
#   (1,4): 1 upstream → order 1
#   (2,4): 2 order-1 inputs → order 2
#   (3,4): 1 order-2 input → order 2; dir=W→(3,3)
# Confluence:
#   (3,3): 2 order-2 inputs [(3,2),(3,4)] → order 3
#   (4,3): 1 order-3 input → order 3
#   (5,3) sink → 0
_ORDER3_ARR = np.array(
    [[255, 255, 255, 255, 255, 255],
     [255,   1,   4,   1,   4, 255],
     [255,   1,   4,   1,   4, 255],
     [255, 255,   1,   4,  16, 255],
     [255, 255, 255,   4, 255, 255],
     [255, 255, 255,   0, 255, 255],
     [255, 255, 255, 255, 255, 255]],
    dtype=np.uint8,
)
_ORDER3_EXPECTED = np.array(
    [[0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 0],
     [0, 1, 2, 1, 2, 0],
     [0, 0, 2, 3, 2, 0],
     [0, 0, 0, 3, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# Tests: simple Y confluence
# ---------------------------------------------------------------------------

class TestStrahlerSimpleConfluence:
    """Verify orders on a Y-shaped confluence grid (expected max order = 2)."""

    def test_output_type_is_strahler(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        assert out.meta.type == GridType.Strahler

    def test_output_dtype_is_uint8(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        assert out.array.dtype == np.uint8

    def test_shape_and_transform_preserved(self):
        fd = _flowdir(_SIMPLE_ARR)
        out = strahler_order(fd)
        assert out.shape == fd.shape
        assert out.meta.transform == fd.meta.transform
        assert out.meta.crs is None

    def test_nodata_value_is_zero(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        assert out.meta.nodata == 0

    def test_known_orders(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        np.testing.assert_array_equal(out.array, _SIMPLE_EXPECTED)

    def test_nodata_border_cells_are_zero(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        assert np.all(out.array[_SIMPLE_ARR == 255] == 0)

    def test_sink_cell_is_zero(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        assert np.all(out.array[_SIMPLE_ARR == 0] == 0)

    def test_max_order_is_2(self):
        out = strahler_order(_flowdir(_SIMPLE_ARR))
        assert int(out.array.max()) == 2


# ---------------------------------------------------------------------------
# Tests: straight chain → all order 1
# ---------------------------------------------------------------------------

class TestStrahlerHeadwaterChain:
    """Single headwater chain with no confluences → every stream cell order 1."""

    def test_all_stream_cells_order_1(self):
        fd = _flowdir(_CHAIN_ARR)
        out = strahler_order(fd)
        stream = (_CHAIN_ARR > 0) & (_CHAIN_ARR <= 128)
        assert np.all(out.array[stream] == 1)

    def test_max_order_is_1(self):
        out = strahler_order(_flowdir(_CHAIN_ARR))
        assert int(out.array.max()) == 1


# ---------------------------------------------------------------------------
# Tests: order-3 confluence (two order-2 branches meeting)
# ---------------------------------------------------------------------------

class TestStrahlerOrder3Confluence:
    """Two Y-shapes merge into a common trunk → order 3 at the main confluence."""

    def test_known_orders(self):
        out = strahler_order(_flowdir(_ORDER3_ARR))
        np.testing.assert_array_equal(out.array, _ORDER3_EXPECTED)

    def test_max_order_is_3(self):
        out = strahler_order(_flowdir(_ORDER3_ARR))
        assert int(out.array.max()) == 3

    def test_confluence_cell_is_order_3(self):
        out = strahler_order(_flowdir(_ORDER3_ARR))
        assert out.array[3, 3] == 3

    def test_trunk_downstream_of_confluence_is_order_3(self):
        out = strahler_order(_flowdir(_ORDER3_ARR))
        assert out.array[4, 3] == 3


# ---------------------------------------------------------------------------
# Tests: unequal-order merge does not increment
# ---------------------------------------------------------------------------

class TestStrahlerUnequalMerge:
    """When a lower-order tributary joins a higher-order trunk, order is unchanged.

    Grid (5×4): three E-flowing headwaters draining into a single S-flowing column.
    The column sees an order-1 input at every row, but confluences are always
    one order-2 + one order-1 → max stays 2, count of max = 1 → no increment.

      col:  0    1    2    3
      row 1: 255   1    4  255    (1,1)→E, col2→S
      row 2: 255   1    4  255    (2,1)→E, col2→S    ← confluence: two order-1 → order 2
      row 3: 255   1    4  255    (3,1)→E, col2→S    ← one order-2 + one order-1 → order 2
      row 4: 255 255    0  255    sink
    """

    _ARR = np.array(
        [[255, 255, 255, 255],
         [255,   1,   4, 255],
         [255,   1,   4, 255],
         [255,   1,   4, 255],
         [255, 255,   0, 255]],
        dtype=np.uint8,
    )
    _EXPECTED = np.array(
        [[0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 1, 2, 0],
         [0, 1, 2, 0],
         [0, 0, 0, 0]],
        dtype=np.uint8,
    )

    def test_known_orders(self):
        out = strahler_order(_flowdir(self._ARR))
        np.testing.assert_array_equal(out.array, self._EXPECTED)

    def test_max_order_stays_2(self):
        out = strahler_order(_flowdir(self._ARR))
        assert int(out.array.max()) == 2


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestStrahlerEdgeCases:

    def test_all_nodata_returns_zeros(self):
        arr = np.full((5, 5), 255, dtype=np.uint8)
        out = strahler_order(_flowdir(arr))
        np.testing.assert_array_equal(out.array, np.zeros((5, 5), dtype=np.uint8))

    def test_single_sink_returns_zero(self):
        arr = np.zeros((3, 3), dtype=np.uint8)
        arr[0, :] = arr[2, :] = arr[:, 0] = arr[:, 2] = 255
        out = strahler_order(_flowdir(arr))
        assert out.array[1, 1] == 0


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------

class TestStrahlerInputValidation:

    def test_wrong_grid_type_raises(self):
        arr = np.zeros((4, 4), dtype=np.float32)
        grid = Grid.create(
            array=arr,
            type=GridType.FlowAcc,
            transform=TRANSFORM,
            crs=None,
        )
        with pytest.raises((TypeError, ValueError)):
            strahler_order(grid)


# ---------------------------------------------------------------------------
# Tests: GridType.Strahler registration
# ---------------------------------------------------------------------------

class TestGridTypeStrahler:

    def test_strahler_in_gridtype_enum(self):
        assert hasattr(GridType, "Strahler")
        assert GridType.Strahler.value == "Strahler"

    def test_grid_create_coerces_to_uint8(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        g = Grid.create(
            array=arr,
            type=GridType.Strahler,
            transform=TRANSFORM,
            crs=None,
        )
        assert g.array.dtype == np.uint8
        assert g.meta.nodata == 0
        assert g.meta.type == GridType.Strahler

    def test_default_nodata_is_zero(self):
        assert _default_nodata(GridType.Strahler, np.dtype(np.uint8)) == 0

    def test_strahler_in_allowed_dtypes(self):
        from fdup._core.types import _ALLOWED_DTYPES
        assert GridType.Strahler in _ALLOWED_DTYPES
        assert np.uint8 in _ALLOWED_DTYPES[GridType.Strahler]
