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

from fdup._core.d8 import DIR_DROW, DIR_DCOL, ENCODE_DIR
from fdup._core.geodesy import get_cell_areas
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_type
from affine import Affine


# ---------------------------------------------------------------------------
# BFS neighbour tables (module-level so Numba sees pre-known types)
# ---------------------------------------------------------------------------

# Row/col offsets for the 8 D8 neighbours, in DIR iteration order.
_BFS_DR = DIR_DROW.astype(np.int32)
_BFS_DC = DIR_DCOL.astype(np.int32)

# For each direction i the "back" code: the D8 code a neighbour at offset
# (_BFS_DR[i], _BFS_DC[i]) must carry to flow *into* the current cell.
# Derived from ENCODE_DIR by looking up the opposite offset (-dr, -dc).
_BFS_BACK = np.array(
    [ENCODE_DIR[-int(r) + 1, -int(c) + 1] for r, c in zip(DIR_DROW, DIR_DCOL)],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# BFS kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bfs(
    flowdir: np.ndarray,
    pour_row: np.int32,
    pour_col: np.int32,
    drs: np.ndarray,
    dcs: np.ndarray,
    back: np.ndarray,
) -> np.ndarray:
    """Upstream BFS from a pour cell.

    Traces backwards through the D8 grid to find all cells that drain
    through ``(pour_row, pour_col)``.
    """
    nrows, ncols = flowdir.shape

    visited = np.zeros((nrows, ncols), dtype=np.bool_)
    mask    = np.zeros((nrows, ncols), dtype=np.bool_)
    queue_r = np.empty(nrows * ncols, dtype=np.int32)
    queue_c = np.empty(nrows * ncols, dtype=np.int32)
    head = np.int32(0)
    tail = np.int32(0)

    visited[pour_row, pour_col] = True
    mask[pour_row, pour_col]    = True
    queue_r[tail] = pour_row
    queue_c[tail] = pour_col
    tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        for k in range(8):
            nr = r + drs[k]
            nc = c + dcs[k]
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                continue
            if visited[nr, nc]:
                continue
            if flowdir[nr, nc] == back[k]:
                visited[nr, nc] = True
                mask[nr, nc]    = True
                queue_r[tail]   = nr
                queue_c[tail]   = nc
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
    mask = _bfs(
        flowdir.array,
        np.int32(pour_row), np.int32(pour_col),
        _BFS_DR, _BFS_DC, _BFS_BACK,
    )
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
    _bfs(tiny, np.int32(1), np.int32(0), _BFS_DR, _BFS_DC, _BFS_BACK)
