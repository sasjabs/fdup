"""River-tree tracing from D8 flow direction and accumulation.

Public API
----------
river_tree(flowdir, flowacc, mask=None) -> tuple[Grid, np.ndarray]
    Trace river segments as a tree raster plus a structured array of seeds.

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
        When provided, segments whose entire path lies outside the mask are
        pruned; surviving segments are re-densified so ``tree_grid == i+1``
        corresponds to ``seeds[i]``.

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

    # ---- optional mask pruning + re-densification -----------------------
    if mask is not None and n > 0:
        mask_arr = mask.array

        # For each cell owned by a segment, check whether it overlaps the mask.
        seed_has_masked = np.zeros(n, dtype=np.bool_)
        valid_tree  = tree_raw > 0
        tree_ids    = tree_raw[valid_tree].astype(np.intp) - 1   # 0-based
        mask_hits   = mask_arr[valid_tree]
        np.maximum.at(seed_has_masked, tree_ids, mask_hits)

        surviving = np.where(seed_has_masked)[0]   # 0-based indices of kept seeds
        n_surv    = len(surviving)

        # Build old-1-based → new-1-based remapping (0 = pruned)
        id_map = np.zeros(n + 1, dtype=np.uint32)
        for new_id, old_idx in enumerate(surviving, start=1):
            id_map[int(old_idx) + 1] = np.uint32(new_id)
        tree_raw = id_map[tree_raw]

        new_seeds = np.empty(n_surv, dtype=seeds_dtype)
        if n_surv > 0:
            new_seeds["mouth_row"]     = seeds["mouth_row"][surviving]
            new_seeds["mouth_col"]     = seeds["mouth_col"][surviving]
            new_seeds["headwater_row"] = seeds["headwater_row"][surviving]
            new_seeds["headwater_col"] = seeds["headwater_col"][surviving]
            new_seeds["ncells"]        = seeds["ncells"][surviving]
            new_seeds["acc"]           = seeds["acc"][surviving]
            new_seeds["length_m"]      = seeds["length_m"][surviving]
        seeds = new_seeds

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
# Warmup
# ---------------------------------------------------------------------------


def _warmup(dtype: np.dtype | None = None) -> None:  # noqa: ARG001
    """Pre-compile ``_nb_trace`` for every supported FlowAcc dtype.

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
