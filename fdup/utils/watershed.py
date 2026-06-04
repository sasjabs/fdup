"""Watershed delineation utilities.

Functions
---------
delineate_watershed(flowdir, pour_row, pour_col) -> Grid
disaggregate_mask(mask, k) -> Grid
mask_area(mask) -> float
_warmup(dtype) -> None

Note: Implemented in Phase 3, Step 3.5.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fdup._core.geodesy import get_cell_areas
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_type
from affine import Affine


# ---------------------------------------------------------------------------
# BFS kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bfs(flowdir: np.ndarray, pour_row: int, pour_col: int) -> np.ndarray:
    """Upstream BFS from a pour cell.

    Traces backwards through the D8 grid to find all cells that drain
    through ``(pour_row, pour_col)``.

    The reverse-direction lookup: a neighbour at offset (dr, dc) drains
    *into* the current cell if its flow code equals the opposite direction.
    Direction table (ESRI codes): 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE.
    Opposite of direction d is: E<->W, SE<->NW, S<->N, SW<->NE.
    """
    nrows, ncols = flowdir.shape

    # 8 neighbours and the ESRI code that means "this neighbour flows HERE"
    # (i.e. the code in the neighbour cell that points back to the current cell)
    # Neighbour at (dr, dc) points back to us with the *opposite* code.
    # Opposites: 1<->16, 2<->32, 4<->64, 8<->128
    NR   = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int64)
    NC   = np.array([ 1,  1,  0, -1, -1, -1,  0,  1], dtype=np.int64)
    # Code that neighbour at offset NR[i], NC[i] must have to drain to (r,c)
    BACK = np.array([16, 32, 64, 128, 1, 2, 4, 8], dtype=np.uint8)

    mask = np.zeros((nrows, ncols), dtype=np.bool_)
    queue = np.empty(nrows * ncols, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    mask[pour_row, pour_col] = True
    queue[tail] = np.int64(pour_row) * ncols + pour_col
    tail += 1

    while head < tail:
        flat = queue[head]
        head += 1
        r = flat // ncols
        c = flat % ncols
        for k in range(8):
            nr = r + NR[k]
            nc = c + NC[k]
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if not mask[nr, nc] and flowdir[nr, nc] == BACK[k]:
                    mask[nr, nc] = True
                    queue[tail] = np.int64(nr) * ncols + nc
                    tail += 1

    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delineate_watershed(
    flowdir: Grid,
    pour_row: int,
    pour_col: int,
) -> Grid:
    """Delineate the upstream watershed for a pour point.

    Parameters
    ----------
    flowdir :
        ``GridType.FlowDir``, uint8.
    pour_row, pour_col :
        Row/column indices of the pour cell.

    Returns
    -------
    Grid
        ``GridType.Mask``, bool, same transform and CRS as *flowdir*.
        ``True`` for cells that drain through the pour point.
    """
    check_type(flowdir, GridType.FlowDir)
    mask = _bfs(flowdir.array, int(pour_row), int(pour_col))
    return Grid.create(
        array=mask,
        type=GridType.Mask,
        transform=flowdir.meta.transform,
        crs=flowdir.meta.crs,
    )


def disaggregate_mask(mask: Grid, k: int) -> Grid:
    """Disaggregate a Mask Grid by factor *k* (pure numpy, no numba).

    Parameters
    ----------
    mask :
        ``GridType.Mask``, bool.
    k :
        Positive integer scale factor.  Each cell becomes a k×k block of
        cells with the same value.

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
    """Compute the geographic area of a True-valued mask in **km²**.

    Parameters
    ----------
    mask :
        ``GridType.Mask``, bool.

    Returns
    -------
    float
        Total area in km².
    """
    check_type(mask, GridType.Mask)
    cell_areas = get_cell_areas(
        mask.meta.transform,
        mask.shape[0],
        geographic=mask.meta.is_geographic,
    )
    return float(np.sum(mask.array * cell_areas[:, np.newaxis]))


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(dtype: np.dtype | None = None) -> None:  # noqa: ARG001
    """Pre-compile the BFS kernel.  Dtype-agnostic; *dtype* is ignored."""
    tiny = np.array([[4, 4], [0, 0]], dtype=np.uint8)
    _bfs(tiny, 1, 0)
