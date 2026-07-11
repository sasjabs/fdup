"""Mask creation and manipulation utilities.

Functions
---------
disaggregate_mask(mask, k) -> Grid
    Disaggregate a Mask Grid by integer factor *k* (moved from watershed.py).
mask_area(mask) -> float
    Total area of True-valued cells (moved from watershed.py).
threshold_mask(grid, cutoff) -> Grid
    Boolean mask where ``grid >= cutoff``; nodata cells map to False.
mask_grid(grid, mask) -> Grid
    Set nodata on *grid* wherever *mask* is False.
crop_grid(grid) -> Grid
    Trim *grid* to the minimal bounding box of data cells.
"""

from __future__ import annotations

import numpy as np
from affine import Affine

from fdup._core.geodesy import get_cell_areas
from fdup._core.types import Grid, GridType
from fdup._core.validation import (
    check_crs_match,
    check_shape_match,
    check_transform_match,
    check_type,
)


# ---------------------------------------------------------------------------
# Moved from watershed.py
# ---------------------------------------------------------------------------


def disaggregate_mask(mask: Grid, k: int) -> Grid:
    """Disaggregate a Mask Grid by factor *k* (pure numpy).

    Parameters
    ----------
    mask :
        ``GridType.Mask``, bool.
    k :
        Positive integer scale factor.  Each cell becomes a k×k block with
        the same value.

    Returns
    -------
    Grid
        ``GridType.Mask``, bool, shape ``(nrows*k, ncols*k)``.  The output
        transform has pixel spacing ``a/k`` and ``e/k``; the origin is
        unchanged.
    """
    check_type(mask, GridType.Mask)
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    out = np.repeat(np.repeat(mask.array, k, axis=0), k, axis=1)
    t = mask.meta.transform
    out_transform = Affine(t.a / k, t.b, t.c, t.d, t.e / k, t.f)
    return Grid.create(
        array=out,
        type=GridType.Mask,
        transform=out_transform,
        crs=mask.meta.crs,
    )


def mask_area(mask: Grid) -> float:
    """Compute the area of all True-valued cells in a mask.

    Parameters
    ----------
    mask :
        ``GridType.Mask``, bool.

    Returns
    -------
    float
        Total area of True cells.  Unit is **km²** for a geographic CRS
        (lat/lon); **CRS units²** (e.g. m²) for a projected CRS.
    """
    check_type(mask, GridType.Mask)
    cell_areas = get_cell_areas(
        mask.meta.transform,
        mask.shape[0],
        geographic=mask.meta.is_geographic,
    )
    return float(np.sum(mask.array * cell_areas[:, np.newaxis]))


# ---------------------------------------------------------------------------
# New utilities
# ---------------------------------------------------------------------------


def threshold_mask(grid: Grid, cutoff: float) -> Grid:
    """Create a boolean mask where *grid* values are >= *cutoff*.

    Parameters
    ----------
    grid :
        ``GridType.FlowAcc`` or ``GridType.Strahler``.
    cutoff :
        Threshold value (inclusive).

    Returns
    -------
    Grid
        ``GridType.Mask``, bool.  Cells with ``value >= cutoff`` are True;
        nodata and NaN cells are False.  Same transform and CRS as *grid*.
    """
    check_type(grid, (GridType.FlowAcc, GridType.Strahler))
    arr = grid.array
    nodata = grid.meta.nodata

    # Build a boolean mask of valid (non-nodata) cells.
    if np.issubdtype(arr.dtype, np.floating):
        valid = ~np.isnan(arr)
        if nodata is not None and not np.isnan(float(nodata)):
            valid &= arr != nodata
    else:
        valid = np.ones(arr.shape, dtype=np.bool_)
        if nodata is not None:
            valid &= arr != nodata

    out = valid & (arr >= cutoff)
    return Grid.create(
        array=out,
        type=GridType.Mask,
        transform=grid.meta.transform,
        crs=grid.meta.crs,
    )


def mask_grid(grid: Grid, mask: Grid) -> Grid:
    """Apply a boolean mask to *grid*, setting nodata where *mask* is False.

    Parameters
    ----------
    grid :
        Any ``Grid``.
    mask :
        ``GridType.Mask``, bool.  Must have the same shape, transform, and
        CRS as *grid*.

    Returns
    -------
    Grid
        A new Grid of the same type as *grid*.  Cells where *mask* is False
        are set to ``grid.meta.nodata``.  For ``GridType.Mask`` inputs
        (whose nodata is ``None``) those cells are set to ``False`` instead.

    Raises
    ------
    ValueError
        On shape, transform, or CRS mismatch; or if *grid* is a non-Mask
        type with ``nodata=None``.
    """
    check_type(mask, GridType.Mask)
    check_shape_match(grid, mask)
    check_transform_match(grid, mask)
    check_crs_match(grid, mask)

    out_arr = grid.array.copy()

    if grid.meta.type == GridType.Mask:
        out_arr[~mask.array] = False
        return Grid.create(
            array=out_arr,
            type=GridType.Mask,
            transform=grid.meta.transform,
            crs=grid.meta.crs,
        )

    nodata = grid.meta.nodata
    if nodata is None:
        raise ValueError(
            "Cannot apply mask_grid to a non-Mask grid with nodata=None."
        )
    out_arr[~mask.array] = nodata
    return Grid.create(
        array=out_arr,
        type=grid.meta.type,
        transform=grid.meta.transform,
        crs=grid.meta.crs,
        nodata=nodata,
    )


def crop_grid(grid: Grid) -> Grid:
    """Trim *grid* to the minimal bounding box of data cells.

    Data cells are defined as:

    * ``True`` for ``GridType.Mask``.
    * Non-NaN (and non-nodata-sentinel) for float arrays.
    * Non-nodata-sentinel for integer arrays (all cells if nodata is None).

    Parameters
    ----------
    grid :
        Any ``Grid``.

    Returns
    -------
    Grid
        A new Grid with the array sliced to the minimal bounding box and the
        transform origin shifted accordingly (``c += col0 * a``,
        ``f += row0 * e``).  Returned unchanged when no trimming is needed.

    Raises
    ------
    ValueError
        When all cells are nodata.
    """
    arr = grid.array
    nodata = grid.meta.nodata

    # ---- determine which cells are data ----
    if grid.meta.type == GridType.Mask:
        data_mask = arr.astype(np.bool_)
    elif np.issubdtype(arr.dtype, np.floating):
        data_mask = ~np.isnan(arr)
        if nodata is not None and not np.isnan(float(nodata)):
            data_mask &= arr != nodata
    else:
        if nodata is not None:
            data_mask = arr != nodata
        else:
            data_mask = np.ones(arr.shape, dtype=np.bool_)

    rows_with_data = np.any(data_mask, axis=1)
    cols_with_data = np.any(data_mask, axis=0)

    if not rows_with_data.any():
        raise ValueError(
            "crop_grid: all cells are nodata; cannot crop to an empty bounding box."
        )

    row0 = int(np.argmax(rows_with_data))
    row1 = int(len(rows_with_data) - 1 - np.argmax(rows_with_data[::-1])) + 1
    col0 = int(np.argmax(cols_with_data))
    col1 = int(len(cols_with_data) - 1 - np.argmax(cols_with_data[::-1])) + 1

    nrows, ncols = arr.shape
    if row0 == 0 and row1 == nrows and col0 == 0 and col1 == ncols:
        return grid

    out_arr = arr[row0:row1, col0:col1]

    t = grid.meta.transform
    new_c = t.c + col0 * t.a
    new_f = t.f + row0 * t.e
    new_transform = Affine(t.a, t.b, new_c, t.d, t.e, new_f)

    return Grid.create(
        array=out_arr,
        type=grid.meta.type,
        transform=new_transform,
        crs=grid.meta.crs,
        nodata=nodata,
    )
