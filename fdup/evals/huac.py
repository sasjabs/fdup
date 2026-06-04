"""Hierarchical upstream area comparison (HUAC).

Public API
----------
huac(flowacc_fine, flowdir_coarse, flowacc_coarse, pour_row, pour_col, *,
     upstream_area_threshold) -> tuple[Grid, pd.DataFrame]
    BFS from *pour_row/pour_col* on the coarse grid.  For each coarse cell,
    the maximum fine accumulation within its footprint is compared to the
    coarse accumulation, yielding a relative-error raster and a per-cell table.

    Port of ``run_hwac`` from the raw ``hwac.py`` script with the following
    changes:
    * ``validate_grids_and_alignment`` replaced by ``check_aligned`` (returns
      ``off_r, off_c`` in addition to ``kx, ky``, enabling correct mapping
      when the grids are not co-registered at (0, 0)).
    * ``coarse_cell_to_fine_bounds`` replaced by closed-form integer arithmetic
      ``r0 = off_r + cr * ky`` (no floating-point world-coordinate round-trip).
    * Nodata sentinels on integer FlowAcc inputs are converted to NaN before
      the numba kernels so all FlowAcc dtypes are handled correctly.

_warmup(dtype)
    Pre-compile ``collect_upstream_neighbors`` and ``max_and_argmax_2d_finite``
    for the warmup registry.  The *dtype* argument is accepted for uniformity;
    both kernels operate on float64 / uint8 internally and do not specialize
    per FlowAcc dtype.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from fdup._core.types import Grid, GridType
from fdup._core.validation import (
    check_aligned,
    check_crs_match,
    check_shape_match,
    check_transform_match,
    check_type,
)

# ---------------------------------------------------------------------------
# Column names — match the raw hwac.py exactly (no renaming).
# ---------------------------------------------------------------------------

_TABLE_COLUMNS = (
    "cell_row",
    "cell_col",
    "max_pixel_row",
    "max_pixel_col",
    "coarse_upstream_area",
    "fine_max_upstream_area",
    "relative_upstream_area_error",
)


# ---------------------------------------------------------------------------
# Numba kernels (ported from hwac.py — unchanged)
# ---------------------------------------------------------------------------


@njit(cache=True)
def max_and_argmax_2d_finite(arr: np.ndarray):
    """Maximum over a 2-D float array, ignoring NaN.

    Returns
    -------
    (max_val, arg_row, arg_col)
        If no finite values exist, returns ``(nan, -1, -1)``.
        Row/column indices are relative to *arr* (0-based).
    """
    h = arr.shape[0]
    w = arr.shape[1]
    found = False
    best = 0.0
    best_r = -1
    best_c = -1
    for i in range(h):
        for j in range(w):
            v = arr[i, j]
            if not np.isnan(v):
                if not found or v > best:
                    best = v
                    best_r = i
                    best_c = j
                    found = True
    if not found:
        return np.nan, -1, -1
    return best, best_r, best_c


@njit(cache=True)
def collect_upstream_neighbors(
    dir_arr: np.ndarray,
    cr: int,
    cc: int,
    dir_nodata: np.int64,
    out_nr: np.ndarray,
    out_nc: np.ndarray,
) -> int:
    """Find coarse cells in the 8-neighbourhood of ``(cr, cc)`` that drain into it.

    Parameters
    ----------
    dir_arr :
        ESRI D8 codes (any integer dtype); invalid/nodata/sink cells are skipped.
    out_nr, out_nc :
        Pre-allocated length-8 int64 buffers; first *n* entries are valid.

    Returns
    -------
    n : int
        Number of upstream neighbours written (0..8).
    """
    nrows = dir_arr.shape[0]
    ncols = dir_arr.shape[1]

    DI = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    DJ = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    NBR_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
    NBR_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)

    ctd = np.full(256, np.int8(-1), dtype=np.int8)
    ctd[1]   = np.int8(0)
    ctd[2]   = np.int8(1)
    ctd[4]   = np.int8(2)
    ctd[8]   = np.int8(3)
    ctd[16]  = np.int8(4)
    ctd[32]  = np.int8(5)
    ctd[64]  = np.int8(6)
    ctd[128] = np.int8(7)

    n = 0
    for k in range(8):
        nr = cr + NBR_DR[k]
        nc = cc + NBR_DC[k]
        if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
            continue
        code = np.int64(dir_arr[nr, nc])
        if code == dir_nodata or code == 0:
            continue
        if code < 1 or code > 255:
            continue
        d = np.int64(ctd[code])
        if d < 0:
            continue
        dr = nr + DI[d]
        dc = nc + DJ[d]
        if dr == cr and dc == cc:
            out_nr[n] = nr
            out_nc[n] = nc
            n += 1
    return n


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float64_with_nan(grid: Grid) -> np.ndarray:
    """Cast a FlowAcc grid's array to float64, replacing integer nodata with NaN."""
    arr = grid.array.astype(np.float64)
    nd = grid.meta.nodata
    if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
        arr[arr == float(nd)] = np.nan
    return arr


def _fine_window(cr: int, cc: int, kx: int, ky: int, off_r: int, off_c: int,
                 fine_h: int, fine_w: int) -> tuple[int, int, int, int]:
    """Return the fine-pixel half-open slice ``[r0:r1, c0:c1]`` for coarse cell ``(cr, cc)``.

    Uses closed-form integer arithmetic derived from ``check_aligned`` offsets,
    replacing the world-coordinate round-trip in the raw ``coarse_cell_to_fine_bounds``.
    The result is clipped to ``[0, fine_h)`` × ``[0, fine_w)``.
    """
    r0 = max(0, off_r + cr * ky)
    r1 = min(fine_h, off_r + (cr + 1) * ky)
    c0 = max(0, off_c + cc * kx)
    c1 = min(fine_w, off_c + (cc + 1) * kx)
    return r0, r1, c0, c1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def huac(
    flowacc_fine: Grid,
    flowdir_coarse: Grid,
    flowacc_coarse: Grid,
    pour_row: int,
    pour_col: int,
    *,
    upstream_area_threshold: float = 0.0,
) -> tuple[Grid, pd.DataFrame]:
    """Hierarchical upstream area comparison (HUAC).

    Traverses upstream from *pour_row/pour_col* on the coarse grid via BFS.
    For each visited coarse cell the function locates the maximum fine
    accumulation within that cell's fine-pixel footprint and computes a
    relative error:

    .. math::

        e = \\frac{A_{\\text{coarse}} - A_{\\text{fine,max}}}{A_{\\text{fine,max}}}

    Parameters
    ----------
    flowacc_fine :
        ``GridType.FlowAcc`` fine-resolution flow accumulation.
    flowdir_coarse :
        ``GridType.FlowDir`` coarse-resolution flow direction (ESRI D8 uint8).
    flowacc_coarse :
        ``GridType.FlowAcc`` coarse-resolution flow accumulation.  Must share
        the same shape, transform, and CRS as *flowdir_coarse*.
    pour_row, pour_col :
        Pour-point cell indices on the coarse grid (0-based row, column).
    upstream_area_threshold :
        Coarse cells whose accumulation is below this value are not written
        to the error raster or output table, but upstream BFS continues through
        them so that small tributaries do not block traversal.

    Returns
    -------
    error_grid : Grid
        ``GridType.FlowAcc`` float32 relative-error raster with coarse shape.
        NaN at unvisited or unevaluable cells.  ``nodata=nan``.
    df : pd.DataFrame
        Per-cell table with columns
        ``(cell_row, cell_col, max_pixel_row, max_pixel_col,
        coarse_upstream_area, fine_max_upstream_area,
        relative_upstream_area_error)``.
        Index is the default RangeIndex.

    Raises
    ------
    ValueError
        On type/shape/transform/CRS/alignment mismatch or out-of-bounds pour
        point.
    """
    check_type(flowacc_fine, GridType.FlowAcc)
    check_type(flowdir_coarse, GridType.FlowDir)
    check_type(flowacc_coarse, GridType.FlowAcc)

    check_shape_match(flowdir_coarse, flowacc_coarse)
    check_transform_match(flowdir_coarse, flowacc_coarse)
    check_crs_match(flowdir_coarse, flowacc_coarse)
    check_crs_match(flowacc_fine, flowdir_coarse)

    kx, ky, off_r, off_c = check_aligned(coarse=flowdir_coarse, fine=flowacc_fine)

    coarse_h, coarse_w = flowdir_coarse.shape
    fine_h, fine_w = flowacc_fine.shape

    if not (0 <= pour_row < coarse_h and 0 <= pour_col < coarse_w):
        raise ValueError(
            f"Pour point ({pour_row}, {pour_col}) is outside coarse grid "
            f"shape ({coarse_h}, {coarse_w})."
        )

    # Convert accumulation arrays to float64 with nodata → NaN so that
    # max_and_argmax_2d_finite correctly skips nodata cells for all dtypes.
    coarse_acc_f64 = _to_float64_with_nan(flowacc_coarse)
    fine_acc_f64 = _to_float64_with_nan(flowacc_fine)

    dir_arr = flowdir_coarse.array
    dir_nodata_i64 = np.int64(255)  # FlowDir canonical nodata

    error_grid = np.full((coarse_h, coarse_w), np.nan, dtype=np.float32)
    visited = np.zeros((coarse_h, coarse_w), dtype=bool)
    rows: list[dict[str, Any]] = []

    queue: deque[tuple[int, int]] = deque()
    queue.append((pour_row, pour_col))
    visited[pour_row, pour_col] = True

    out_nr = np.empty(8, dtype=np.int64)
    out_nc = np.empty(8, dtype=np.int64)

    while queue:
        cr, cc = queue.popleft()

        if dir_arr[cr, cc] == 255:
            continue
        coarse_area = coarse_acc_f64[cr, cc]
        if np.isnan(coarse_area):
            continue

        # Enqueue upstream neighbors then skip cells below threshold
        # (traversal must continue through small tributaries).
        if coarse_area < upstream_area_threshold:
            n_up = collect_upstream_neighbors(
                dir_arr, cr, cc, dir_nodata_i64, out_nr, out_nc
            )
            for i in range(n_up):
                nr, nc = int(out_nr[i]), int(out_nc[i])
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            continue

        r0, r1, c0, c1 = _fine_window(cr, cc, kx, ky, off_r, off_c, fine_h, fine_w)

        if r0 >= r1 or c0 >= c1:
            fine_max = np.nan
            max_pr: float = np.nan
            max_pc: float = np.nan
            rel_err: float = np.nan
        else:
            window = fine_acc_f64[r0:r1, c0:c1]
            fine_max, ar, ac = max_and_argmax_2d_finite(window)
            if ar < 0:
                max_pr = np.nan
                max_pc = np.nan
            else:
                max_pr = float(r0 + ar)
                max_pc = float(c0 + ac)

            if np.isnan(fine_max) or fine_max <= 0.0:
                rel_err = np.nan
            else:
                rel_err = float((coarse_area - fine_max) / fine_max)

        error_grid[cr, cc] = np.float32(rel_err)

        rows.append(
            {
                "cell_row": cr,
                "cell_col": cc,
                "max_pixel_row": max_pr,
                "max_pixel_col": max_pc,
                "coarse_upstream_area": float(coarse_area),
                "fine_max_upstream_area": (
                    float(fine_max) if np.isfinite(fine_max) else np.nan
                ),
                "relative_upstream_area_error": rel_err,
            }
        )

        n_up = collect_upstream_neighbors(
            dir_arr, cr, cc, dir_nodata_i64, out_nr, out_nc
        )
        for i in range(n_up):
            nr, nc = int(out_nr[i]), int(out_nc[i])
            if not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    df = pd.DataFrame(rows, columns=list(_TABLE_COLUMNS))

    err_grid = Grid.create(
        array=error_grid,
        type=GridType.FlowAcc,
        transform=flowdir_coarse.meta.transform,
        crs=flowdir_coarse.meta.crs,
        nodata=float("nan"),
    )
    return err_grid, df


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup(dtype: object) -> None:  # noqa: ARG001
    """Pre-compile ``max_and_argmax_2d_finite`` and ``collect_upstream_neighbors``.

    The *dtype* argument is accepted for warmup-registry uniformity; both
    kernels always operate on float64 / uint8 and do not specialise per
    FlowAcc dtype.
    """
    dummy_f = np.array([[1.0, np.nan], [3.0, 2.0]], dtype=np.float64)
    max_and_argmax_2d_finite(dummy_f)

    dgrid = np.zeros((3, 3), dtype=np.uint8)
    dgrid[0, 1] = 4  # S: (0,1) → (1,1)
    out_r = np.empty(8, dtype=np.int64)
    out_c = np.empty(8, dtype=np.int64)
    collect_upstream_neighbors(dgrid, 1, 1, np.int64(255), out_r, out_c)
