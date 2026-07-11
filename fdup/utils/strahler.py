"""Strahler stream order from a FlowDir Grid.

Public API
----------
strahler_order(flowdir) -> Grid
    Compute Strahler stream orders using BFS in topological order.
    Headwater cells (valid stream cells with no upstream neighbour) receive
    order 1; the Strahler merge rule increments the order at confluences
    where two or more tributaries of equal maximum order meet.

_warmup(dtype) -> None
    Pre-compile numba kernels.  The *dtype* argument is accepted for
    registry uniformity but ignored (FlowDir is always uint8).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fdup._core.types import Grid, GridType
from fdup._core.validation import check_dtype, check_type


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _strahler_bfs(dir_arr: np.ndarray, order: np.ndarray, n_valid: int) -> None:
    """Compute Strahler stream order via BFS in topological order.

    Each valid stream cell is processed exactly once, after all of its
    upstream cells have been assigned their orders.  The Strahler rule is
    then applied locally:

        order = max_in + 1   if two or more incoming tributaries share
                              the maximum incoming order
        order = max_in       otherwise

    Cells in flow-direction cycles, sinks (dir == 0), and nodata cells
    (dir > 128) receive order == 0.

    Parameters
    ----------
    dir_arr : np.ndarray, uint8, 2-D
        Esri D8 flow direction raster (1, 2, 4, 8, 16, 32, 64, 128).
        0 = sink/flat.  Values > 128 treated as nodata.
    order : np.ndarray, uint8, 2-D
        Pre-allocated zero array; written in-place.
    n_valid : int
        Number of valid stream cells (used to size the BFS queue).
    """
    dj_arr = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    di_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

    # Direction-code → neighbour-offset index
    idx = np.zeros(256, dtype=np.int64)
    idx[1]   = np.int64(0)
    idx[2]   = np.int64(1)
    idx[4]   = np.int64(2)
    idx[8]   = np.int64(3)
    idx[16]  = np.int64(4)
    idx[32]  = np.int64(5)
    idx[64]  = np.int64(6)
    idx[128] = np.int64(7)

    nrow = np.int64(dir_arr.shape[0])
    ncol = np.int64(dir_arr.shape[1])

    # in_degree[i, j] = number of valid upstream cells that flow into (i, j).
    in_degree = np.zeros((int(nrow), int(ncol)), dtype=np.int8)

    for i in range(int(nrow)):
        for j in range(int(ncol)):
            d = dir_arr[i, j]
            if d == 0 or d > 128:
                continue
            k  = idx[int(d)]
            ni = np.int64(i) + di_arr[k]
            nj = np.int64(j) + dj_arr[k]
            if ni < 0 or ni >= nrow or nj < 0 or nj >= ncol:
                continue
            nd = dir_arr[int(ni), int(nj)]
            if nd == 0 or nd > 128:
                continue
            in_degree[int(ni), int(nj)] += np.int8(1)

    # Per-cell accumulators for the confluence merge rule.
    max_in  = np.zeros((int(nrow), int(ncol)), dtype=np.uint8)
    cnt_max = np.zeros((int(nrow), int(ncol)), dtype=np.uint8)

    # BFS queue pre-allocated to the number of valid stream cells.
    queue = np.empty(int(n_valid), dtype=np.int64)
    head  = np.int64(0)
    tail  = np.int64(0)

    # Seed: valid stream cells with no upstream neighbours (headwaters → order 1).
    for i in range(int(nrow)):
        for j in range(int(ncol)):
            d = dir_arr[i, j]
            if d == 0 or d > 128:
                continue
            if in_degree[i, j] == 0:
                order[i, j]  = np.uint8(1)
                queue[tail]  = np.int64(i) * ncol + np.int64(j)
                tail        += np.int64(1)

    # BFS — each cell dequeued exactly once, after all upstream cells are done.
    while head < tail:
        flat     = queue[head]
        head    += np.int64(1)
        i        = int(flat // ncol)
        j        = int(flat % ncol)
        curr_ord = int(order[i, j])

        d = dir_arr[i, j]
        if d == 0 or d > 128:
            continue

        k  = idx[int(d)]
        ni = np.int64(i) + di_arr[k]
        nj = np.int64(j) + dj_arr[k]

        if ni < 0 or ni >= nrow or nj < 0 or nj >= ncol:
            continue

        nd = dir_arr[int(ni), int(nj)]
        if nd == 0 or nd > 128:
            continue

        # Update downstream cell's merge accumulators.
        if curr_ord > int(max_in[int(ni), int(nj)]):
            max_in [int(ni), int(nj)] = np.uint8(curr_ord)
            cnt_max[int(ni), int(nj)] = np.uint8(1)
        elif curr_ord == int(max_in[int(ni), int(nj)]):
            cnt_max[int(ni), int(nj)] += np.uint8(1)

        in_degree[int(ni), int(nj)] -= np.int8(1)

        if in_degree[int(ni), int(nj)] == 0:
            m = int(max_in [int(ni), int(nj)])
            c = int(cnt_max[int(ni), int(nj)])
            order[int(ni), int(nj)] = np.uint8(m + 1) if c >= 2 else np.uint8(m)
            queue[tail]  = ni * ncol + nj
            tail        += np.int64(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strahler_order(flowdir: Grid) -> Grid:
    """Compute Strahler stream order for a D8 flow-direction Grid.

    Headwaters (valid stream cells with no upstream neighbour) receive order
    1.  At each confluence the Strahler rule is applied:

    * If two or more incoming tributaries share the maximum incoming order,
      the downstream cell receives that maximum + 1.
    * Otherwise the downstream cell inherits the maximum order unchanged.

    Sinks (dir == 0), nodata cells (dir == 255), and cells inside
    flow-direction cycles all receive order 0 (the Strahler nodata value).

    Time / memory complexity: O(N) — each cell is enqueued and dequeued once.

    Parameters
    ----------
    flowdir :
        ``GridType.FlowDir``, uint8.  Esri D8 encoding
        (1, 2, 4, 8, 16, 32, 64, 128); 0 = sink; 255 = nodata.

    Returns
    -------
    Grid
        ``GridType.Strahler``, uint8.  0 = nodata / sink / non-stream.
        Same transform and CRS as *flowdir*.
    """
    check_type(flowdir, GridType.FlowDir)
    check_dtype(flowdir, (np.uint8,))

    dir_arr = flowdir.array
    order   = np.zeros(dir_arr.shape, dtype=np.uint8)
    n_valid = int(np.count_nonzero((dir_arr > 0) & (dir_arr <= 128)))
    _strahler_bfs(dir_arr, order, n_valid)

    return Grid.create(
        array=order,
        type=GridType.Strahler,
        transform=flowdir.meta.transform,
        crs=flowdir.meta.crs,
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(dtype: object = None) -> None:  # noqa: ARG001
    """Pre-compile the Strahler BFS kernel.

    *dtype* is accepted for registry uniformity but ignored (FlowDir is
    always uint8).
    """
    _dir = np.array(
        [[255, 255, 255, 255, 255],
         [255,   1,   1,   0, 255],
         [255,   1,   1,   0, 255],
         [255, 255, 255, 255, 255]],
        dtype=np.uint8,
    )
    _order   = np.zeros(_dir.shape, dtype=np.uint8)
    _n_valid = int(np.count_nonzero((_dir > 0) & (_dir <= 128)))
    _strahler_bfs(_dir, _order, _n_valid)
