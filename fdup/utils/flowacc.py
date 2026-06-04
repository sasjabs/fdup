"""Flow accumulation from a FlowDir Grid.

Public API
----------
flow_accumulation(flowdir, *, area=True) -> Grid
    Compute upstream flow accumulation.

_warmup(dtype) -> None
    Pre-compile numba kernels.  The *dtype* argument is accepted for
    registry uniformity but ignored.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fdup._core.geodesy import get_cell_areas
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_dtype, check_type


# ---------------------------------------------------------------------------
# Numba kernel  (Kahn's topological-sort BFS — ported from get_flowacc.py)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _flow_accumulation_kernel(
    dir_arr: np.ndarray,   # int32[:, :]
    weights: np.ndarray,   # float64[:, :]
    dir_nodata: np.int64,
) -> np.ndarray:
    """Kahn's topological-sort flow accumulation.

    Returns a float64 array with accumulated weights.  Cells where
    ``dir_arr == dir_nodata`` are set to NaN.
    """
    nrows = dir_arr.shape[0]
    ncols = dir_arr.shape[1]

    DI = np.array([0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int64)
    DJ = np.array([1,  1,  0, -1, -1, -1,  0,  1], dtype=np.int64)

    ctd = np.full(256, np.int8(-1), dtype=np.int8)
    ctd[1]   = np.int8(0)
    ctd[2]   = np.int8(1)
    ctd[4]   = np.int8(2)
    ctd[8]   = np.int8(3)
    ctd[16]  = np.int8(4)
    ctd[32]  = np.int8(5)
    ctd[64]  = np.int8(6)
    ctd[128] = np.int8(7)

    in_deg = np.zeros((nrows, ncols), dtype=np.int32)
    for i in range(nrows):
        for j in range(ncols):
            code = np.int64(dir_arr[i, j])
            if code == dir_nodata or code == 0:
                continue
            if code < 1 or code > 255:
                continue
            d = np.int64(ctd[code])
            if d < 0:
                continue
            ni = i + DI[d]
            nj = j + DJ[d]
            if 0 <= ni < nrows and 0 <= nj < ncols:
                in_deg[ni, nj] += 1

    acc = np.empty((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            if np.int64(dir_arr[i, j]) == dir_nodata:
                acc[i, j] = np.nan
            else:
                w = weights[i, j]
                acc[i, j] = 0.0 if np.isnan(w) else w

    queue = np.empty(nrows * ncols, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for i in range(nrows):
        for j in range(ncols):
            if np.int64(dir_arr[i, j]) != dir_nodata and in_deg[i, j] == 0:
                queue[tail] = np.int64(i) * ncols + j
                tail += 1

    while head < tail:
        flat = queue[head]
        head += 1
        i = flat // ncols
        j = flat % ncols

        code = np.int64(dir_arr[i, j])
        if code == dir_nodata or code == 0:
            continue
        if code < 1 or code > 255:
            continue
        d = np.int64(ctd[code])
        if d < 0:
            continue

        ni = i + DI[d]
        nj = j + DJ[d]
        if 0 <= ni < nrows and 0 <= nj < ncols:
            if np.int64(dir_arr[ni, nj]) != dir_nodata:
                acc[ni, nj] += acc[i, j]
                in_deg[ni, nj] -= 1
                if in_deg[ni, nj] == 0:
                    queue[tail] = np.int64(ni) * ncols + nj
                    tail += 1

    return acc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flow_accumulation(flowdir: Grid, *, area: bool = True) -> Grid:
    """Compute flow accumulation for a D8 flow-direction Grid.

    Parameters
    ----------
    flowdir :
        ``GridType.FlowDir``, uint8.
    area :
        When ``True`` (default), each cell accumulates the geographic area
        of its upstream cells in **km²**; output dtype is ``float32``,
        nodata is ``NaN``.
        When ``False``, each cell counts its upstream cells (including
        itself); output dtype is ``uint32`` if the maximum fits, else
        ``uint64``; nodata is ``iinfo(dtype).max``.

    Returns
    -------
    Grid
        ``GridType.FlowAcc``, same transform and CRS as *flowdir*.
    """
    check_type(flowdir, GridType.FlowDir)
    check_dtype(flowdir, (np.uint8,))

    nrows, ncols = flowdir.shape
    dir_i32 = flowdir.array.astype(np.int32, copy=False)
    dir_nodata = np.int64(255)

    if area:
        cell_areas = get_cell_areas(
            flowdir.meta.transform,
            nrows,
            geographic=flowdir.meta.is_geographic,
        )
        weights = np.broadcast_to(cell_areas[:, np.newaxis], (nrows, ncols)).astype(np.float64)
        weights = np.ascontiguousarray(weights)
        acc_f64 = _flow_accumulation_kernel(dir_i32, weights, dir_nodata)
        out = acc_f64.astype(np.float32)
        nodata_val = float("nan")
        return Grid.create(
            array=out,
            type=GridType.FlowAcc,
            transform=flowdir.meta.transform,
            crs=flowdir.meta.crs,
            nodata=nodata_val,
        )
    else:
        weights = np.ones((nrows, ncols), dtype=np.float64)
        acc_f64 = _flow_accumulation_kernel(dir_i32, weights, dir_nodata)

        # Replace NaN (nodata cells) with 0 for max inspection, then check range
        valid_mask = ~np.isnan(acc_f64)
        max_acc = int(acc_f64[valid_mask].max()) if valid_mask.any() else 0

        if max_acc <= int(np.iinfo(np.uint32).max) - 1:
            out_dtype = np.uint32
        else:
            out_dtype = np.uint64

        sentinel = int(np.iinfo(out_dtype).max)
        out = np.where(valid_mask, acc_f64.astype(out_dtype), np.iinfo(out_dtype).max).astype(out_dtype)

        return Grid.create(
            array=out,
            type=GridType.FlowAcc,
            transform=flowdir.meta.transform,
            crs=flowdir.meta.crs,
            nodata=sentinel,
        )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(dtype: np.dtype | None = None) -> None:  # noqa: ARG001
    """Pre-compile the flow-accumulation kernel.

    The *dtype* argument is accepted for registry uniformity but ignored.
    """
    tiny_dir = np.array([[4, 4], [4, 0]], dtype=np.int32)
    tiny_w = np.ones((2, 2), dtype=np.float64)
    _flow_accumulation_kernel(tiny_dir, tiny_w, np.int64(255))
