"""River-tree tracing from D8 flow direction and accumulation.

Public API
----------
river_tree(flowdir, flowacc, mask=None) -> tuple[Grid, np.ndarray]
    Trace river segments as a tree raster plus a structured array of seeds.

mask_seeds(seeds, flowdir, flowacc, mask) -> np.ndarray
    Prune a seeds array to only in-mask sub-segments, re-computing acc and
    length_m from the flowacc array and flow-direction grid.

_warmup(dtype) -> None
    Pre-compile numba kernels for each FlowAcc dtype.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fdup._core.geodesy import row_distance_table
from fdup._core.types import Grid, GridType
from fdup._core.validation import (
    check_crs_match,
    check_dtype,
    check_shape_match,
    check_transform_match,
    check_type,
)

_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _valid_acc(val, nodata_val):
    """Return True if *val* is a usable, positive accumulation value.

    Handles both float (NaN = nodata) and integer (sentinel = nodata) dtypes
    through numba's per-type specialisation.
    """
    if val != val:          # IEEE NaN test — True only for float NaN
        return False
    if val == nodata_val:   # exact sentinel match for int types (NaN≠NaN, so safe for float too)
        return False
    return val > val - val  # val > 0  (type-generic zero: val - val == 0 for any dtype)


@njit(cache=True)
def _nb_trace(acc_arr, nodata_val, dir_arr, row_dists, cell_i, cell_j, nvalid):
    """Trace all river-tree segments upstream from high→low accumulation cells.

    Parameters
    ----------
    acc_arr : 2-D numeric array
        Flow accumulation. Dtype is any member of ``_FLOWACC_DTYPES``; numba
        specialises this function per dtype so no compression is needed.
    nodata_val : scalar (same dtype as acc_arr)
        Nodata sentinel — NaN for float types, ``iinfo(dtype).max`` for ints.
    dir_arr : uint8 2-D
        ESRI D8 flow direction (1=E … 128=NE, 0/255=nodata).
    row_dists : float64 (nrows, 8)
        Per-row neighbour distances in metres, COMPASS_ORDER (E SE S SW W NW N NE).
    cell_i, cell_j : uint32 1-D
        Valid cell coordinates, **pre-sorted by accumulation descending**.
    nvalid : int64
        Length of cell_i / cell_j.

    Returns
    -------
    tree_raw : uint32 (nrows, ncols)
        1-based segment ID per cell; 0 = background or single-cell discard.
    seeds_geom : uint32 (buf_cap, 5)
        Buffer rows: [mouth_row, mouth_col, hw_row, hw_col, ncells].
        Only indices ``0 .. nseed-1`` are populated.
    acc_buf : 1-D (buf_cap), same dtype as acc_arr
        Flow accumulation at mouth per segment (indices 0..nseed-1 valid).
    len_buf : float64 1-D (buf_cap)
        Path length in metres per segment (indices 0..nseed-1 valid).
    nseed : int64
        Number of valid (≥ 2 cell) segments written into the buffers.
    """
    # D8 offsets in COMPASS_ORDER: E SE S SW W NW N NE
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    di = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

    # D8 power-of-2 code → compass index  (0..255 table, unused slots stay 0)
    idx = np.zeros(256, dtype=np.int64)
    idx[1] = 0; idx[2] = 1; idx[4] = 2; idx[8] = 3   # noqa: E702
    idx[16] = 4; idx[32] = 5; idx[64] = 6; idx[128] = 7

    nrow = dir_arr.shape[0]
    ncol = dir_arr.shape[1]

    tree_raw = np.zeros((nrow, ncol), dtype=np.uint32)

    buf_cap = np.int64(max(np.int64(1024), nvalid // np.int64(100) + np.int64(1)))
    seeds_geom = np.empty((buf_cap, 5), dtype=np.uint32)
    acc_buf    = np.empty(buf_cap, dtype=acc_arr.dtype)
    len_buf    = np.empty(buf_cap, dtype=np.float64)

    tree_id = np.uint32(0)
    nseed   = np.int64(0)

    for p in range(nvalid):
        istart = np.int64(cell_i[p])
        jstart = np.int64(cell_j[p])

        if tree_raw[istart, jstart] > np.uint32(0):
            continue  # already claimed by an earlier (higher-acc) segment

        tree_id += np.uint32(1)
        i = istart
        j = jstart
        tree_raw[i, j] = tree_id

        length  = np.float64(0.0)
        ncells  = np.int64(1)

        while True:
            if not _valid_acc(acc_arr[i, j], nodata_val):
                break

            # Find the upstream neighbour with the highest accumulation
            # that drains INTO the current cell (i, j).
            imax     = i
            jmax     = j
            nb_best  = np.int64(-1)
            # Type-generic zero: val - val == 0 for any numeric dtype.
            best_acc = acc_arr[i, j] - acc_arr[i, j]

            for nb in range(8):
                ik = i + di[nb]
                jl = j + dj[nb]
                if ik < 0 or ik >= nrow or jl < 0 or jl >= ncol:
                    continue
                dv = dir_arr[ik, jl]
                if dv <= np.uint8(0) or dv > np.uint8(128):
                    continue
                d_idx = idx[dv]
                # Confirm (ik, jl) points downstream to (i, j)
                if i != ik + di[d_idx] or j != jl + dj[d_idx]:
                    continue
                if tree_raw[ik, jl] > np.uint32(0):
                    continue  # already in a different segment
                acc_nb = acc_arr[ik, jl]
                if not _valid_acc(acc_nb, nodata_val):
                    continue
                if acc_nb >= best_acc:
                    imax     = ik
                    jmax     = jl
                    best_acc = acc_nb
                    nb_best  = np.int64(nb)

            if imax == i and jmax == j:
                break  # headwater — no valid upstream neighbour found

            length += row_dists[i, nb_best]
            ncells += np.int64(1)
            i = imax
            j = jmax
            tree_raw[i, j] = tree_id

            if dir_arr[i, j] == np.uint8(0):
                break  # sink cell

        # Discard degenerate single-cell "segments" — reset to background.
        if ncells <= np.int64(1):
            tree_raw[istart, jstart] = np.uint32(0)
            tree_id -= np.uint32(1)
            continue

        # Grow output buffers geometrically if needed.
        if nseed >= buf_cap:
            buf_cap   = buf_cap * np.int64(2)
            new_geom  = np.empty((buf_cap, 5), dtype=np.uint32)
            new_geom[:nseed] = seeds_geom[:nseed]
            seeds_geom = new_geom
            new_acc   = np.empty(buf_cap, dtype=acc_arr.dtype)
            new_acc[:nseed] = acc_buf[:nseed]
            acc_buf = new_acc
            new_len  = np.empty(buf_cap, dtype=np.float64)
            new_len[:nseed] = len_buf[:nseed]
            len_buf = new_len

        # Layout: mouth = high-acc start (istart, jstart); headwater = (i, j).
        seeds_geom[nseed, 0] = np.uint32(istart)   # mouth_row
        seeds_geom[nseed, 1] = np.uint32(jstart)   # mouth_col
        seeds_geom[nseed, 2] = np.uint32(i)         # headwater_row
        seeds_geom[nseed, 3] = np.uint32(j)         # headwater_col
        seeds_geom[nseed, 4] = np.uint32(ncells)
        acc_buf[nseed]       = acc_arr[istart, jstart]
        len_buf[nseed]       = length
        nseed += np.int64(1)

    return tree_raw, seeds_geom, acc_buf, len_buf, nseed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seeds_dtype(acc_dtype: np.dtype) -> np.dtype:
    """Structured dtype for the seeds array."""
    return np.dtype([
        ("mouth_row",     np.uint32),
        ("mouth_col",     np.uint32),
        ("headwater_row", np.uint32),
        ("headwater_col", np.uint32),
        ("ncells",        np.uint32),
        ("acc",           acc_dtype),
        ("length_m",      np.float64),
    ])


def _get_nodata_val(acc: np.ndarray, meta_nodata) -> object:
    """Return the nodata sentinel cast to the array's element dtype."""
    if meta_nodata is None:
        if np.issubdtype(acc.dtype, np.floating):
            return acc.dtype.type(float("nan"))
        return acc.dtype.type(np.iinfo(acc.dtype).max)
    return acc.dtype.type(meta_nodata)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def river_tree(
    flowdir: Grid,
    flowacc: Grid,
    mask: Grid | None = None,
) -> tuple[Grid, np.ndarray]:
    """Trace river segments and build a tree raster.

    Parameters
    ----------
    flowdir :
        ``GridType.FlowDir``, uint8.
    flowacc :
        ``GridType.FlowAcc``; dtype must be one of the supported FlowAcc dtypes.
    mask :
        Optional ``GridType.Mask`` (bool), same shape/transform as *flowdir*.
        When provided, only cells where the mask is ``True`` are eligible for
        tracing — the upstream walk stops at mask boundaries so no river
        segment can cross into masked-out areas.

    Returns
    -------
    tree_grid : Grid
        ``GridType.Tree``, uint32.  Cell value = 1-based segment ID; 0 = background.
    seeds : np.ndarray
        Structured array with fields
        ``(mouth_row, mouth_col, headwater_row, headwater_col, ncells, acc, length_m)``.
        ``acc`` dtype matches *flowacc*; ``length_m`` is always float64 metres.
    """
    # ---- validation -------------------------------------------------------
    check_type(flowdir, GridType.FlowDir)
    check_type(flowacc, GridType.FlowAcc)
    check_dtype(flowacc, _FLOWACC_DTYPES)
    check_shape_match(flowdir, flowacc)
    check_transform_match(flowdir, flowacc)
    check_crs_match(flowdir, flowacc)
    if mask is not None:
        check_type(mask, GridType.Mask)
        check_shape_match(flowdir, mask)
        check_transform_match(flowdir, mask)

    acc     = flowacc.array
    dir_arr = flowdir.array
    seeds_dtype = _make_seeds_dtype(acc.dtype)
    nodata_val  = _get_nodata_val(acc, flowacc.meta.nodata)

    # ---- apply mask to flow-direction before any tracing ------------------
    # Zeroing out-of-mask cells in dir_arr means the upstream walk in
    # _nb_trace stops at the mask boundary (dv == 0 is treated as nodata).
    if mask is not None:
        mask_arr = mask.array
        dir_arr  = np.where(mask_arr, dir_arr, np.uint8(0))

    # ---- collect valid cells (D8 direction + non-nodata acc) --------------
    valid_mask = (dir_arr > 0) & (dir_arr <= 128)
    if np.issubdtype(acc.dtype, np.floating):
        valid_mask &= ~np.isnan(acc)
    else:
        valid_mask &= acc != nodata_val

    rows, cols = np.nonzero(valid_mask)
    if rows.size == 0:
        empty_tree = np.zeros(flowdir.shape, dtype=np.uint32)
        return (
            Grid.create(
                array=empty_tree,
                type=GridType.Tree,
                transform=flowdir.meta.transform,
                crs=flowdir.meta.crs,
            ),
            np.empty(0, dtype=seeds_dtype),
        )

    # Sort cells by accumulation descending
    acc_at_cells = acc[rows, cols]
    order   = np.argsort(acc_at_cells)[::-1]
    cell_i  = rows[order].astype(np.uint32)
    cell_j  = cols[order].astype(np.uint32)
    nvalid  = np.int64(len(cell_i))

    # Per-row D8-neighbour distance table (metres, COMPASS_ORDER)
    row_dists = row_distance_table(
        flowdir.meta.transform,
        flowdir.shape[0],
        geographic=flowdir.meta.is_geographic,
    )

    # ---- numba tracing kernel --------------------------------------------
    tree_raw, seeds_geom, acc_buf, len_buf, nseed = _nb_trace(
        acc, nodata_val, dir_arr, row_dists, cell_i, cell_j, nvalid,
    )

    n = int(nseed)
    seeds = np.empty(n, dtype=seeds_dtype)
    if n > 0:
        seeds["mouth_row"]     = seeds_geom[:n, 0]
        seeds["mouth_col"]     = seeds_geom[:n, 1]
        seeds["headwater_row"] = seeds_geom[:n, 2]
        seeds["headwater_col"] = seeds_geom[:n, 3]
        seeds["ncells"]        = seeds_geom[:n, 4]
        seeds["acc"]           = acc_buf[:n]
        seeds["length_m"]      = len_buf[:n]

    return (
        Grid.create(
            array=tree_raw,
            type=GridType.Tree,
            transform=flowdir.meta.transform,
            crs=flowdir.meta.crs,
        ),
        seeds,
    )


# ---------------------------------------------------------------------------
# mask_seeds kernel
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nb_mask_seeds(
    seeds_geom,   # uint32 (nseed, 5): mouth_row, mouth_col, hw_row, hw_col, ncells
    nseed,        # int64
    acc_arr,      # 2-D numeric — full flowacc for acc lookup
    dir_arr,      # uint8 2-D
    mask_arr,     # bool 2-D
    row_dists,    # float64 (nrows, 8) — COMPASS_ORDER step distances
):
    """Trace each seed headwater→mouth and emit in-mask sub-seeds.

    For every seed, the kernel follows the D8 flowdir from headwater to mouth
    and records contiguous runs of in-mask cells.  Each run with ≥ 2 cells
    becomes one output sub-seed:

    * ``mouth_*``     — most-downstream in-mask cell of the run.
    * ``headwater_*`` — most-upstream in-mask cell of the run.
    * ``ncells``      — number of cells in the run.
    * acc             — ``acc_arr`` value at the run's mouth cell.
    * length_m        — cumulative step distances (metres) within the run.

    Single-cell runs are discarded.

    Returns
    -------
    out_geom : uint32 (n_out, 5)
    out_acc  : 1-D, same dtype as acc_arr
    out_len  : float64 1-D
    n_out    : int64
    """
    # D8 offsets (COMPASS_ORDER: E SE S SW W NW N NE)
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    di = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

    # D8 power-of-2 code → compass index (0..255 table)
    idx = np.zeros(256, dtype=np.int64)
    idx[1] = 0; idx[2] = 1; idx[4] = 2; idx[8] = 3    # noqa: E702
    idx[16] = 4; idx[32] = 5; idx[64] = 6; idx[128] = 7

    nrow = dir_arr.shape[0]
    ncol = dir_arr.shape[1]

    buf_cap = np.int64(max(np.int64(1024), nseed * np.int64(4)))
    out_geom = np.empty((buf_cap, 5), dtype=np.uint32)
    out_acc  = np.empty(buf_cap, dtype=acc_arr.dtype)
    out_len  = np.empty(buf_cap, dtype=np.float64)
    n_out    = np.int64(0)

    for s in range(nseed):
        hw_r      = np.int64(seeds_geom[s, 2])
        hw_c      = np.int64(seeds_geom[s, 3])
        m_r       = np.int64(seeds_geom[s, 0])
        m_c       = np.int64(seeds_geom[s, 1])
        max_steps = np.int64(seeds_geom[s, 4])

        row = hw_r
        col = hw_c

        in_run     = False
        run_ncells = np.int64(0)
        run_len    = np.float64(0.0)
        run_hw_r   = np.int64(0)
        run_hw_c   = np.int64(0)
        run_last_r = np.int64(0)
        run_last_c = np.int64(0)
        prev_dist  = np.float64(0.0)

        for _ in range(max_steps):
            in_mask = mask_arr[row, col]

            if in_mask:
                if not in_run:
                    in_run     = True
                    run_ncells = np.int64(1)
                    run_len    = np.float64(0.0)
                    run_hw_r   = row
                    run_hw_c   = col
                else:
                    run_ncells += np.int64(1)
                    run_len    += prev_dist
                run_last_r = row
                run_last_c = col
            else:
                if in_run and run_ncells >= np.int64(2):
                    if n_out >= buf_cap:
                        buf_cap  = buf_cap * np.int64(2)
                        ng = np.empty((buf_cap, 5), dtype=np.uint32)
                        ng[:n_out] = out_geom[:n_out]
                        out_geom = ng
                        na = np.empty(buf_cap, dtype=acc_arr.dtype)
                        na[:n_out] = out_acc[:n_out]
                        out_acc = na
                        nl = np.empty(buf_cap, dtype=np.float64)
                        nl[:n_out] = out_len[:n_out]
                        out_len = nl
                    out_geom[n_out, 0] = np.uint32(run_last_r)
                    out_geom[n_out, 1] = np.uint32(run_last_c)
                    out_geom[n_out, 2] = np.uint32(run_hw_r)
                    out_geom[n_out, 3] = np.uint32(run_hw_c)
                    out_geom[n_out, 4] = np.uint32(run_ncells)
                    out_acc[n_out]     = acc_arr[run_last_r, run_last_c]
                    out_len[n_out]     = run_len
                    n_out += np.int64(1)
                in_run     = False
                run_ncells = np.int64(0)

            if row == m_r and col == m_c:
                break

            dv = dir_arr[row, col]
            if dv == np.uint8(0) or dv > np.uint8(128):
                break
            d_idx = idx[dv]
            nr = row + di[d_idx]
            nc = col + dj[d_idx]
            if nr < np.int64(0) or nr >= nrow or nc < np.int64(0) or nc >= ncol:
                break

            prev_dist = row_dists[row, d_idx]
            row = nr
            col = nc

        # Emit any remaining active run after the trace ends
        if in_run and run_ncells >= np.int64(2):
            if n_out >= buf_cap:
                buf_cap  = buf_cap * np.int64(2)
                ng = np.empty((buf_cap, 5), dtype=np.uint32)
                ng[:n_out] = out_geom[:n_out]
                out_geom = ng
                na = np.empty(buf_cap, dtype=acc_arr.dtype)
                na[:n_out] = out_acc[:n_out]
                out_acc = na
                nl = np.empty(buf_cap, dtype=np.float64)
                nl[:n_out] = out_len[:n_out]
                out_len = nl
            out_geom[n_out, 0] = np.uint32(run_last_r)
            out_geom[n_out, 1] = np.uint32(run_last_c)
            out_geom[n_out, 2] = np.uint32(run_hw_r)
            out_geom[n_out, 3] = np.uint32(run_hw_c)
            out_geom[n_out, 4] = np.uint32(run_ncells)
            out_acc[n_out]     = acc_arr[run_last_r, run_last_c]
            out_len[n_out]     = run_len
            n_out += np.int64(1)

    return out_geom, out_acc, out_len, n_out


# ---------------------------------------------------------------------------
# mask_seeds public API
# ---------------------------------------------------------------------------

_REQUIRED_SEED_FIELDS = frozenset(
    {"mouth_row", "mouth_col", "headwater_row", "headwater_col", "ncells"}
)


def mask_seeds(
    seeds: np.ndarray,
    flowdir: Grid,
    flowacc: Grid,
    mask: Grid,
) -> np.ndarray:
    """Prune a seeds array to only in-mask contiguous sub-segments.

    Each seed (produced by :func:`river_tree`) is re-traced from headwater to
    mouth using *flowdir*.  Contiguous runs of cells where *mask* is ``True``
    are collected; each run with ≥ 2 cells becomes one output sub-seed.
    ``acc`` is read from *flowacc* at the run's most-downstream cell;
    ``length_m`` is computed via :func:`~fdup._core.geodesy.row_distance_table`.

    Parameters
    ----------
    seeds :
        Structured array as returned by :func:`river_tree`.  Must contain at
        least the fields ``mouth_row``, ``mouth_col``, ``headwater_row``,
        ``headwater_col``, and ``ncells``.
    flowdir :
        ``GridType.FlowDir``, uint8.
    flowacc :
        ``GridType.FlowAcc``; dtype must be one of the supported FlowAcc dtypes.
    mask :
        ``GridType.Mask`` (bool), same shape/transform/CRS as *flowdir*.

    Returns
    -------
    np.ndarray
        Structured array with the same fields as :func:`river_tree` seeds:
        ``(mouth_row, mouth_col, headwater_row, headwater_col, ncells,
        acc, length_m)``.  ``acc`` dtype matches *flowacc*; ``length_m`` is
        always float64 metres.  May be empty if no in-mask run of ≥ 2 cells
        is found.
    """
    # ---- validation -------------------------------------------------------
    check_type(flowdir, GridType.FlowDir)
    check_type(flowacc, GridType.FlowAcc)
    check_type(mask, GridType.Mask)
    check_dtype(flowacc, _FLOWACC_DTYPES)
    check_shape_match(flowdir, flowacc)
    check_transform_match(flowdir, flowacc)
    check_crs_match(flowdir, flowacc)
    check_shape_match(flowdir, mask)
    check_transform_match(flowdir, mask)
    check_crs_match(flowdir, mask)

    if seeds.dtype.names is None or not _REQUIRED_SEED_FIELDS.issubset(seeds.dtype.names):
        missing = _REQUIRED_SEED_FIELDS - set(seeds.dtype.names or [])
        raise ValueError(
            f"seeds array is missing required fields: {sorted(missing)}"
        )

    seeds_dtype = _make_seeds_dtype(flowacc.array.dtype)

    if len(seeds) == 0:
        return np.empty(0, dtype=seeds_dtype)

    # ---- build raw geometry array for the kernel --------------------------
    nseed = np.int64(len(seeds))
    seeds_geom = np.empty((int(nseed), 5), dtype=np.uint32)
    seeds_geom[:, 0] = seeds["mouth_row"].astype(np.uint32)
    seeds_geom[:, 1] = seeds["mouth_col"].astype(np.uint32)
    seeds_geom[:, 2] = seeds["headwater_row"].astype(np.uint32)
    seeds_geom[:, 3] = seeds["headwater_col"].astype(np.uint32)
    seeds_geom[:, 4] = seeds["ncells"].astype(np.uint32)

    # ---- per-row D8-neighbour distance table ------------------------------
    row_dists = row_distance_table(
        flowdir.meta.transform,
        flowdir.shape[0],
        geographic=flowdir.meta.is_geographic,
    )

    # ---- numba kernel -----------------------------------------------------
    out_geom, out_acc, out_len, n_out = _nb_mask_seeds(
        seeds_geom,
        nseed,
        flowacc.array,
        flowdir.array,
        mask.array,
        row_dists,
    )

    n = int(n_out)
    out = np.empty(n, dtype=seeds_dtype)
    if n > 0:
        out["mouth_row"]     = out_geom[:n, 0]
        out["mouth_col"]     = out_geom[:n, 1]
        out["headwater_row"] = out_geom[:n, 2]
        out["headwater_col"] = out_geom[:n, 3]
        out["ncells"]        = out_geom[:n, 4]
        out["acc"]           = out_acc[:n]
        out["length_m"]      = out_len[:n]
    return out


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup(dtype: np.dtype | None = None) -> None:  # noqa: ARG001
    """Pre-compile ``_nb_trace`` and ``_nb_mask_seeds`` for every supported FlowAcc dtype.

    The *dtype* argument is accepted for registry uniformity but ignored;
    all FlowAcc dtypes are compiled unconditionally.
    """
    # Minimal 4×4 grid: single stream (0,0)→SE→(1,1)→SE→(2,2), SE=code 2.
    # Accumulation increases downstream: (2,2)=3 > (1,1)=2 > (0,0)=1.
    _dir = np.zeros((4, 4), dtype=np.uint8)
    _dir[0, 0] = np.uint8(2)   # SE
    _dir[1, 1] = np.uint8(2)   # SE
    _dir[2, 2] = np.uint8(2)   # SE — drains out of grid

    _row_dists = np.ones((4, 8), dtype=np.float64) * 1000.0

    # Pre-sorted cells (descending acc): (2,2), (1,1), (0,0)
    _ci = np.array([2, 1, 0], dtype=np.uint32)
    _cj = np.array([2, 1, 0], dtype=np.uint32)
    _nv = np.int64(3)

    # Seed for mask_seeds warmup: one seed spanning (0,0)→(2,2), 3 cells
    _seeds_geom = np.array([[2, 2, 0, 0, 3]], dtype=np.uint32)
    _nseed = np.int64(1)
    # All-True mask
    _mask_arr = np.ones((4, 4), dtype=np.bool_)

    for _dt in _FLOWACC_DTYPES:
        if np.issubdtype(_dt, np.floating):
            _acc = np.array(
                [[1, 0, 0, 0],
                 [0, 2, 0, 0],
                 [0, 0, 3, 0],
                 [0, 0, 0, 0]],
                dtype=_dt,
            )
            _nd = _dt(float("nan"))
        else:
            _acc = np.array(
                [[1, 0, 0, 0],
                 [0, 2, 0, 0],
                 [0, 0, 3, 0],
                 [0, 0, 0, 0]],
                dtype=_dt,
            )
            _nd = _dt(np.iinfo(_dt).max)

        _nb_trace(_acc, _nd, _dir, _row_dists, _ci, _cj, _nv)
        _nb_mask_seeds(_seeds_geom, _nseed, _acc, _dir, _mask_arr, _row_dists)
