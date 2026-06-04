"""Snap a pour point to the highest flow-accumulation cell within a radius.

Public API
----------
snap_pour_cell(flowacc, x, y, radius) -> tuple[int, int]
    Find the highest-accumulation valid cell within *radius* CRS units of
    the world coordinate ``(x, y)``.

_warmup(dtype) -> None
    Pre-compile numba kernels for all supported FlowAcc dtypes.

Notes
-----
``radius`` is **Euclidean distance in CRS units**, even for geographic
CRSes.  For EPSG:4326 this is degrees, not metres.  This is
intentional — to use metres, project the grid first.
"""

from __future__ import annotations

import math

import numpy as np
import numba

from fdup._core.types import Grid, GridType
from fdup._core.validation import check_type

_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _snap_nan(
    grid,
    pour_row: float,
    pour_col: float,
    pixel_width: float,
    pixel_height: float,
    snap_dist_sq: float,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
):
    """Find the highest-accumulation cell, treating NaN as nodata."""
    best_row = np.int64(-1)
    best_col = np.int64(-1)
    best_val = -np.inf
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            val = grid[r, c]
            if np.isnan(val):
                continue
            dr = ((r + 0.5) - pour_row) * pixel_height
            dc = ((c + 0.5) - pour_col) * pixel_width
            if dr * dr + dc * dc > snap_dist_sq:
                continue
            fval = float(val)
            if fval > best_val:
                best_val = fval
                best_row = r
                best_col = c
    return best_row, best_col


@numba.njit(cache=True)
def _snap_nodata(
    grid,
    nodata,
    pour_row: float,
    pour_col: float,
    pixel_width: float,
    pixel_height: float,
    snap_dist_sq: float,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
):
    """Find the highest-accumulation cell, treating *nodata* as invalid."""
    best_row = np.int64(-1)
    best_col = np.int64(-1)
    best_val = -np.inf
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            val = grid[r, c]
            if val == nodata:
                continue
            dr = ((r + 0.5) - pour_row) * pixel_height
            dc = ((c + 0.5) - pour_col) * pixel_width
            if dr * dr + dc * dc > snap_dist_sq:
                continue
            fval = float(val)
            if fval > best_val:
                best_val = fval
                best_row = r
                best_col = c
    return best_row, best_col


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snap_pour_cell(
    flowacc: Grid,
    x: float,
    y: float,
    radius: float,
) -> tuple[int, int]:
    """Snap a pour point to the highest flow-accumulation cell within *radius*.

    Parameters
    ----------
    flowacc :
        ``GridType.FlowAcc`` Grid.
    x, y :
        World coordinates of the nominal pour point in the grid's CRS.
    radius :
        **Euclidean distance in CRS units**, even for geographic CRSes.
        For EPSG:4326 this is **degrees**, not metres.  This is intentional
        — to use metres, project the grid first.

    Returns
    -------
    (row, col) : tuple[int, int]
        Row and column indices of the snapped pour-point cell.

    Raises
    ------
    ValueError
        When no valid (finite / non-nodata) flow-accumulation cell exists
        within *radius* of ``(x, y)``.
    """
    check_type(flowacc, GridType.FlowAcc)

    arr = flowacc.array
    transform = flowacc.meta.transform
    nrows, ncols = flowacc.shape

    pour_col_f, pour_row_f = ~transform * (float(x), float(y))

    pixel_width = float(transform.a)
    pixel_height = float(transform.e)
    snap_dist_sq = float(radius) ** 2

    col_radius = radius / abs(pixel_width)
    row_radius = radius / abs(pixel_height)
    col_min = max(0, int(math.floor(pour_col_f - col_radius)))
    col_max = min(ncols - 1, int(math.ceil(pour_col_f + col_radius)))
    row_min = max(0, int(math.floor(pour_row_f - row_radius)))
    row_max = min(nrows - 1, int(math.ceil(pour_row_f + row_radius)))

    dtype = arr.dtype
    use_nan = dtype.kind == "f"

    if use_nan:
        best_row, best_col = _snap_nan(
            arr,
            pour_row_f, pour_col_f,
            pixel_width, pixel_height, snap_dist_sq,
            row_min, row_max, col_min, col_max,
        )
    else:
        nodata = flowacc.meta.nodata
        if nodata is None:
            nodata = int(np.iinfo(dtype).max)
        nodata_typed = dtype.type(nodata)
        best_row, best_col = _snap_nodata(
            arr, nodata_typed,
            pour_row_f, pour_col_f,
            pixel_width, pixel_height, snap_dist_sq,
            row_min, row_max, col_min, col_max,
        )

    if best_row == -1:
        raise ValueError(
            f"snap_pour_cell: no valid pour cell within radius={radius} of ({x}, {y})"
        )

    return int(best_row), int(best_col)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(dtype: np.dtype | None = None) -> None:  # noqa: ARG001
    """Pre-compile both snap kernels for all supported FlowAcc dtypes.

    The *dtype* argument is accepted for registry uniformity but ignored.
    """
    ones = np.ones((3, 3))
    args = (1.5, 1.5, 1.0, -1.0, 4.0, 0, 2, 0, 2)

    for dt in (np.float32, np.float64):
        _snap_nan(ones.astype(dt), *args)

    for dt in (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64):
        arr = ones.astype(dt)
        _snap_nodata(arr, arr.dtype.type(0), *args)
