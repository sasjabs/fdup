"""
Implements COTAT / COTAT+ (Cell Outlet Tracing with an Area Threshold)
flow direction upscaling.

Base algorithm by Reed (2003): https://doi.org/10.1029/2003WR001989
COTAT+ outlet-selection refinement (Minimum Upstream Flow Path) by
Paz et al. (2006): https://doi.org/10.1029/2005WR004544

The plain COTAT outlet (largest-upstream-area pixel) is used when no MUFP
value is supplied; otherwise the COTAT+ outlet-selection scheme is applied.
"""

from __future__ import annotations

import math

import numpy as np
from affine import Affine
from numba import njit, prange

from fdup._core.d8 import (
    DECODE_DC, DECODE_DR, DECODE_VALID, ENCODE_DIR,
    DIR_DCOL, DIR_DROW,
)
from fdup._core.geodesy import row_distance_table
from fdup._core.types import Grid, GridType
from fdup._core.validation import (
    check_crs_match, check_dtype, check_shape_match,
    check_transform_match, check_type, normalize_k,
)


# =========================================================================
# JIT kernels
# =========================================================================

@njit(cache=True, inline="always")
def _fa(fa, r, c, ndv, has_ndv, ndv_is_nan):
    """Flow accumulation at ``(r, c)`` as float64. Nodata reads as 0.

    The sentinel test uses the raw array value: NaN by ``val != val``, any
    other sentinel by ``val == ndv`` in *fa*'s dtype. *ndv* is ignored when
    *has_ndv* is false; pass a zero of *fa*'s dtype in that case. Callers
    only invoke this for coordinates inside the original raster.
    """
    val = fa[r, c]
    if has_ndv:
        if ndv_is_nan:
            if val != val:
                return np.float64(0.0)
        elif val == ndv:
            return np.float64(0.0)
    return np.float64(val)


@njit(cache=True)
def _find_largest_in_cell(flowacc, i, j, kx, ky, orig_nrows, orig_ncols,
                          fa_ndv, fa_has_ndv, fa_ndv_is_nan):
    """Find pixel with largest flow accumulation in cell (i, j).

    Cell (i, j) covers *ky* fine rows by *kx* fine columns. The scan stops
    at the original raster edge. Nodata flow accumulation contributes 0.
    """
    r0 = i * ky
    c0 = j * kx
    r1 = min(r0 + ky, orig_nrows)
    c1 = min(c0 + kx, orig_ncols)
    best_r = r0
    best_c = c0
    best_val = _fa(flowacc, r0, c0, fa_ndv, fa_has_ndv, fa_ndv_is_nan)
    for r in range(r0, r1):
        for c in range(c0, c1):
            v = _fa(flowacc, r, c, fa_ndv, fa_has_ndv, fa_ndv_is_nan)
            if v > best_val:
                best_val = v
                best_r = r
                best_c = c
    return best_r, best_c


# -------------------------------------------------------------------------
# COTAT+ MUFP-based outlet selection
# -------------------------------------------------------------------------

@njit(cache=True)
def _pixel_valid(flowdir, r, c, orig_nrows, orig_ncols, flowdir_ndv, has_ndv):
    """True when pixel (r, c) lies inside the original extent and is not nodata."""
    if r >= orig_nrows or c >= orig_ncols:
        return False
    if has_ndv and flowdir[r, c] == flowdir_ndv:
        return False
    return True


@njit(cache=True)
def _largest_valid_in_cell(flowdir, flowacc, i, j, kx, ky,
                           orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                           fa_ndv, fa_has_ndv, fa_ndv_is_nan):
    """Largest-flowacc *valid* pixel of cell (i, j).

    Cell (i, j) covers *ky* fine rows by *kx* fine columns. Pixels outside
    the original extent and flow-direction nodata are ignored. The scan
    stops at the raster edge. Flow-accumulation nodata contributes 0.
    """
    r0 = i * ky
    c0 = j * kx
    r1 = min(r0 + ky, orig_nrows)
    c1 = min(c0 + kx, orig_ncols)
    best_r = r0
    best_c = c0
    best_val = -np.inf
    for r in range(r0, r1):
        for c in range(c0, c1):
            if not _pixel_valid(flowdir, r, c, orig_nrows, orig_ncols,
                                flowdir_ndv, has_ndv):
                continue
            v = _fa(flowacc, r, c, fa_ndv, fa_has_ndv, fa_ndv_is_nan)
            if v > best_val:
                best_val = v
                best_r = r
                best_c = c
    return best_r, best_c


@njit(cache=True)
def _trace_upstream_within_cell(flowdir, flowacc, sr, sc, i, j, kx, ky,
                                orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                                fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                                row_dists,
                                decode_dr, decode_dc, decode_valid,
                                dir_drow, dir_dcol):
    """Length of the upstream flow path of pixel (sr, sc) confined to cell (i, j).

    Cell (i, j) covers *ky* fine rows by *kx* fine columns; tracing stops
    after at most ``kx*ky`` steps.  At each junction the inflowing neighbour
    with the largest upstream area is followed.  Step lengths are taken from
    *row_dists* (metres, COMPASS_ORDER), so the returned path length is in
    **metres**.
    """
    r0 = i * ky
    c0 = j * kx
    cur_r = sr
    cur_c = sc
    total = 0.0
    max_steps = kx * ky
    steps = 0

    while steps < max_steps:
        steps += 1
        best_fa = -np.inf
        best_nr = -1
        best_nc = -1
        best_dist = 0.0
        for d in range(8):
            ddr = np.intp(dir_drow[d])
            ddc = np.intp(dir_dcol[d])
            nr = cur_r + ddr
            nc = cur_c + ddc
            if nr < r0 or nr >= r0 + ky or nc < c0 or nc >= c0 + kx:
                continue
            if not _pixel_valid(flowdir, nr, nc, orig_nrows, orig_ncols,
                                flowdir_ndv, has_ndv):
                continue
            nf = flowdir[nr, nc]
            if nf < 1 or nf > 128:
                continue
            nfi = np.intp(nf)
            if not decode_valid[nfi]:
                continue
            # Does this neighbour flow into the current pixel?
            if (nr + np.intp(decode_dr[nfi]) == cur_r
                    and nc + np.intp(decode_dc[nfi]) == cur_c):
                fa = _fa(flowacc, nr, nc, fa_ndv, fa_has_ndv, fa_ndv_is_nan)
                if fa > best_fa:
                    best_fa = fa
                    best_nr = nr
                    best_nc = nc
                    best_dist = row_dists[cur_r, d]
        if best_nr < 0:
            break
        total += best_dist
        cur_r = best_nr
        cur_c = best_nc

    return total


@njit(cache=True)
def _find_cell_outlet_mufp(flowdir, flowacc, i, j, kx, ky,
                           orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                           fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                           mufp, row_dists, decode_dr, decode_dc, decode_valid,
                           dir_drow, dir_dcol):
    """COTAT+ outlet pixel for cell (i, j) using the MUFP selection scheme.

    Cell (i, j) covers *ky* fine rows by *kx* fine columns.  Border capacity
    is ``2*(kx + ky)``; the exit map is indexed as ``(r - r0)*kx + (c - c0)``.
    """
    r0 = i * ky
    c0 = j * kx

    # Step 1: pixel with the largest upstream drainage area.
    lr1, lc1 = _largest_valid_in_cell(
        flowdir, flowacc, i, j, kx, ky,
        orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
        fa_ndv, fa_has_ndv, fa_ndv_is_nan)

    # Border pixels are those whose direction leaves the logical ky×kx cell.
    # The scan itself stops at the raster edge.
    cap = 2 * (kx + ky)
    bo_r = np.empty(cap, dtype=np.int64)
    bo_c = np.empty(cap, dtype=np.int64)
    bo_fa = np.empty(cap, dtype=np.float64)
    exit_map = np.full(kx * ky, -1, dtype=np.int64)
    nb = 0
    r1 = min(r0 + ky, orig_nrows)
    c1 = min(c0 + kx, orig_ncols)
    for r in range(r0, r1):
        for c in range(c0, c1):
            if not _pixel_valid(flowdir, r, c, orig_nrows, orig_ncols,
                                flowdir_ndv, has_ndv):
                continue
            f = flowdir[r, c]
            if f < 1 or f > 128:
                continue
            fi = np.intp(f)
            if not decode_valid[fi]:
                continue
            dr2 = r + np.intp(decode_dr[fi])
            dc2 = c + np.intp(decode_dc[fi])
            if dr2 < r0 or dr2 >= r0 + ky or dc2 < c0 or dc2 >= c0 + kx:
                bo_r[nb] = r
                bo_c[nb] = c
                bo_fa[nb] = _fa(flowacc, r, c, fa_ndv, fa_has_ndv, fa_ndv_is_nan)
                exit_map[(r - r0) * kx + (c - c0)] = nb
                nb += 1

    # Sink case, or the largest pixel is not itself a border out-facing pixel.
    if nb == 0:
        return lr1, lc1
    on_border = False
    for t in range(nb):
        if bo_r[t] == lr1 and bo_c[t] == lc1:
            on_border = True
            break
    if not on_border:
        return lr1, lc1

    # Step 2: border out-facing pixel draining the largest portion of the cell.
    counts = np.zeros(nb, dtype=np.int64)
    for r in range(r0, r1):
        for c in range(c0, c1):
            if not _pixel_valid(flowdir, r, c, orig_nrows, orig_ncols,
                                flowdir_ndv, has_ndv):
                continue
            cr = r
            cc = c
            steps = 0
            max_steps = kx * ky
            while steps < max_steps:
                steps += 1
                idx = exit_map[(cr - r0) * kx + (cc - c0)]
                if idx >= 0:
                    counts[idx] += 1
                    break
                # Inside the logical cell, past the raster edge: not a pixel.
                if not _pixel_valid(flowdir, cr, cc, orig_nrows, orig_ncols,
                                    flowdir_ndv, has_ndv):
                    break
                f = flowdir[cr, cc]
                if f < 1 or f > 128:
                    break
                fi = np.intp(f)
                if not decode_valid[fi]:
                    break
                ncr = cr + np.intp(decode_dr[fi])
                ncc = cc + np.intp(decode_dc[fi])
                if ncr < r0 or ncr >= r0 + ky or ncc < c0 or ncc >= c0 + kx:
                    break
                cr = ncr
                cc = ncc

    most_idx = 0
    for t in range(1, nb):
        if counts[t] > counts[most_idx]:
            most_idx = t

    # Steps 3-4: walk border pixels by descending upstream area; accept the
    # first whose upstream flow path exceeds MUFP, else fall back to the
    # most-draining border pixel once it is reached.
    used = np.zeros(nb, dtype=np.bool_)
    for _ in range(nb):
        sel = -1
        best = -np.inf
        for t in range(nb):
            if not used[t] and bo_fa[t] > best:
                best = bo_fa[t]
                sel = t
        if sel < 0:
            break
        used[sel] = True
        path = _trace_upstream_within_cell(
            flowdir, flowacc, np.intp(bo_r[sel]), np.intp(bo_c[sel]), i, j, kx, ky,
            orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
            fa_ndv, fa_has_ndv, fa_ndv_is_nan,
            row_dists,
            decode_dr, decode_dc, decode_valid, dir_drow, dir_dcol)
        if path > mufp:
            return bo_r[sel], bo_c[sel]
        if sel == most_idx:
            return bo_r[sel], bo_c[sel]

    return lr1, lc1


@njit(cache=True, parallel=True)
def _null_cells(fd, mrows, mcols, kx, ky, nrows, ncols):
    """True where a coarse cell has no in-extent, non-nodata flow-direction pixel.

    Cell ``(i, j)`` covers rows ``[i*ky, i*ky+ky)`` and columns
    ``[j*kx, j*kx+kx)``, clamped to ``nrows`` / ``ncols``. Pixels outside
    the original raster and pixels equal to 255 do not count. The scan
    stops at the first pixel that does.
    """
    out = np.empty((mrows, mcols), dtype=np.bool_)
    for i in prange(mrows):
        r1 = min(i * ky + ky, nrows)
        for j in range(mcols):
            c1 = min(j * kx + kx, ncols)
            null = True
            for r in range(i * ky, r1):
                for c in range(j * kx, c1):
                    if fd[r, c] != 255:
                        null = False
                        break
                if not null:
                    break
            out[i, j] = null
    return out


@njit(cache=True, parallel=True)
def _assign_all_outlets(flowdir, flowacc, null_cells, outlet_coords, mrows, mcols, kx, ky,
                        orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                        fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                        use_mufp, mufp, row_dists, decode_dr, decode_dc, decode_valid,
                        dir_drow, dir_dcol):
    """Assign outlet pixels for all non-null cells (parallel over rows).

    When *use_mufp* is False this reproduces the original COTAT scheme
    (largest-upstream-area pixel); otherwise the COTAT+ MUFP scheme is used.
    """
    for i in prange(mrows):
        for j in range(mcols):
            if not null_cells[i, j]:
                if use_mufp:
                    r, c = _find_cell_outlet_mufp(
                        flowdir, flowacc, i, j, kx, ky,
                        orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                        fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                        mufp, row_dists, decode_dr, decode_dc, decode_valid,
                        dir_drow, dir_dcol)
                else:
                    r, c = _find_largest_in_cell(
                        flowacc, i, j, kx, ky, orig_nrows, orig_ncols,
                        fa_ndv, fa_has_ndv, fa_ndv_is_nan)
                outlet_coords[i, j, 0] = r
                outlet_coords[i, j, 1] = c


@njit(cache=True)
def _trace_cell_direction(flowdir, flowacc, outlet_coords, ci, cj, kx, ky,
                          orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                          fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                          area_threshold, decode_dr, decode_dc, decode_valid,
                          encode_dir):
    """Trace flow downstream from a cell's outlet pixel; return its D8 direction.

    Implements the COTAT trace/stop/receiving-cell scheme: step downstream
    until a neighbouring cell's outlet pixel is reached whose upstream-area
    gain exceeds *area_threshold* (that cell becomes the receiving cell), or
    until the path hits a sink/shoreline or leaves the 3x3 neighbourhood (the
    receiving cell is then the cell of the last visited pixel).
    The coarse cell of a fine pixel is ``(r // ky, c // kx)``.
    """
    cur_r = np.intp(outlet_coords[ci, cj, 0])
    cur_c = np.intp(outlet_coords[ci, cj, 1])
    source_outlet_area = _fa(flowacc, cur_r, cur_c, fa_ndv, fa_has_ndv, fa_ndv_is_nan)

    recv_ci = np.intp(-1)
    recv_cj = np.intp(-1)
    prev_r = cur_r
    prev_c = cur_c

    while True:
        prev_r = cur_r
        prev_c = cur_c

        # hit_sink_or_shoreline(current_pixel)
        fdir = flowdir[cur_r, cur_c]
        if fdir < 1 or fdir > 128:
            break
        fdir_idx = np.intp(fdir)
        if not decode_valid[fdir_idx]:
            break
        next_r = cur_r + np.intp(decode_dr[fdir_idx])
        next_c = cur_c + np.intp(decode_dc[fdir_idx])
        if next_r < 0 or next_r >= orig_nrows or next_c < 0 or next_c >= orig_ncols:
            break
        if has_ndv and flowdir[next_r, next_c] == flowdir_ndv:
            break

        # step_downstream
        cur_r = next_r
        cur_c = next_c

        # out_of_neighborhood(source_cell, current_cell)
        cur_ci = cur_r // ky
        cur_cj = cur_c // kx
        if abs(cur_ci - ci) > 1 or abs(cur_cj - cj) > 1:
            break

        # is_outlet_pixel(current_pixel)
        out_r = np.intp(outlet_coords[cur_ci, cur_cj, 0])
        out_c = np.intp(outlet_coords[cur_ci, cur_cj, 1])
        if out_r == cur_r and out_c == cur_c:
            dA = _fa(flowacc, cur_r, cur_c, fa_ndv, fa_has_ndv, fa_ndv_is_nan) - source_outlet_area
            if dA > area_threshold:
                recv_ci = cur_ci
                recv_cj = cur_cj
                break

    if recv_ci < 0:
        recv_ci = prev_r // ky
        recv_cj = prev_c // kx

    diff_r = recv_ci - ci
    diff_c = recv_cj - cj
    if diff_r < -1 or diff_r > 1 or diff_c < -1 or diff_c > 1:
        return np.uint8(0)
    return encode_dir[np.intp(diff_r + 1), np.intp(diff_c + 1)]


@njit(cache=True, parallel=True)
def _assign_all_directions(flowdir, flowacc, outlet_coords, cells, null_cells,
                           kx, ky, orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                           fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                           area_threshold, decode_dr, decode_dc, decode_valid,
                           encode_dir, mrows, mcols):
    """Assign D8 directions for all non-null cells (parallel over rows)."""
    for i in prange(mrows):
        for j in range(mcols):
            if not null_cells[i, j]:
                cells[i, j] = _trace_cell_direction(
                    flowdir, flowacc, outlet_coords, i, j, kx, ky,
                    orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                    fa_ndv, fa_has_ndv, fa_ndv_is_nan,
                    area_threshold, decode_dr, decode_dc, decode_valid,
                    encode_dir)


@njit(cache=True)
def _fix_intersections_numba(cells, outlet_coords, flowacc, mrows, mcols,
                             fa_ndv, fa_has_ndv, fa_ndv_is_nan):
    """Fix diagonal flow intersections.

    Outlet areas go through ``_fa``, so nodata accumulation compares as 0.
    """
    for i in range(mrows - 1):
        for j in range(mcols - 1):
            if cells[i + 1, j] == 128 and cells[i + 1, j + 1] == 32:
                r0, c0 = outlet_coords[i, j, 0], outlet_coords[i, j, 1]
                r1, c1 = outlet_coords[i, j + 1, 0], outlet_coords[i, j + 1, 1]
                if _fa(flowacc, r0, c0, fa_ndv, fa_has_ndv, fa_ndv_is_nan) > _fa(
                        flowacc, r1, c1, fa_ndv, fa_has_ndv, fa_ndv_is_nan):
                    cells[i + 1, j] = 64
                else:
                    cells[i + 1, j + 1] = 64

            if cells[i, j] == 2 and cells[i, j + 1] == 8:
                r0, c0 = outlet_coords[i + 1, j + 1, 0], outlet_coords[i + 1, j + 1, 1]
                r1, c1 = outlet_coords[i + 1, j, 0], outlet_coords[i + 1, j, 1]
                if _fa(flowacc, r0, c0, fa_ndv, fa_has_ndv, fa_ndv_is_nan) > _fa(
                        flowacc, r1, c1, fa_ndv, fa_has_ndv, fa_ndv_is_nan):
                    cells[i, j + 1] = 4
                else:
                    cells[i, j] = 4

            if cells[i, j + 1] == 8 and cells[i + 1, j + 1] == 32:
                r0, c0 = outlet_coords[i, j, 0], outlet_coords[i, j, 1]
                r1, c1 = outlet_coords[i + 1, j, 0], outlet_coords[i + 1, j, 1]
                if _fa(flowacc, r0, c0, fa_ndv, fa_has_ndv, fa_ndv_is_nan) > _fa(
                        flowacc, r1, c1, fa_ndv, fa_has_ndv, fa_ndv_is_nan):
                    cells[i, j + 1] = 16
                else:
                    cells[i + 1, j + 1] = 16

            if cells[i, j] == 2 and cells[i + 1, j] == 128:
                r0, c0 = outlet_coords[i, j + 1, 0], outlet_coords[i, j + 1, 1]
                r1, c1 = outlet_coords[i + 1, j + 1, 0], outlet_coords[i + 1, j + 1, 1]
                if _fa(flowacc, r0, c0, fa_ndv, fa_has_ndv, fa_ndv_is_nan) > _fa(
                        flowacc, r1, c1, fa_ndv, fa_has_ndv, fa_ndv_is_nan):
                    cells[i, j] = 1
                else:
                    cells[i + 1, j] = 1


# =========================================================================
# JIT warm-up
# =========================================================================

def _warmup(dtype=np.float64):
    """Pre-compile all numba kernels for *dtype*.

    Compiles the sentinel path (a zero of *dtype* marks one pixel nodata).
    The NaN branch lives in the same ``_fa`` kernel, so this covers both.
    """
    flowdir_ndv = np.uint8(255)

    # Plain COTAT outlets + tracing + intersection fixing (2x2 cell grid).
    kx = ky = 1
    mrows = mcols = 2
    n_rows = mrows * ky
    n_cols = mcols * kx
    flowdir = np.zeros((n_rows, n_cols), dtype=np.uint8)
    flowacc = np.ones((n_rows, n_cols), dtype=dtype)
    fa_ndv = flowacc.dtype.type(0)
    flowacc[0, 0] = fa_ndv
    null_cells = _null_cells(flowdir, mrows, mcols, kx, ky, n_rows, n_cols)
    outlet_coords = np.full((mrows, mcols, 2), -1, dtype=np.int32)
    cells = np.full((mrows, mcols), 255, dtype=np.uint8)
    row_dists = np.ones((n_rows, 8), dtype=np.float64) * 1000.0

    _assign_all_outlets(flowdir, flowacc, null_cells, outlet_coords,
                        mrows, mcols, kx, ky, n_rows, n_cols, flowdir_ndv, True,
                        fa_ndv, True, False,
                        False, 0.0, row_dists, DECODE_DR, DECODE_DC, DECODE_VALID,
                        DIR_DROW, DIR_DCOL)
    _assign_all_directions(flowdir, flowacc, outlet_coords, cells,
                           null_cells, kx, ky, n_rows, n_cols, flowdir_ndv, True,
                           fa_ndv, True, False,
                           0.0, DECODE_DR, DECODE_DC, DECODE_VALID,
                           ENCODE_DIR, mrows, mcols)
    _fix_intersections_numba(cells, outlet_coords, flowacc, mrows, mcols,
                             fa_ndv, True, False)

    # COTAT+ MUFP outlet selection (single 1x1 coarse cell with kx=ky=2).
    kx = ky = 2
    mrows = mcols = 1
    n_rows = n_cols = 2
    flowdir = np.ones((n_rows, n_cols), dtype=np.uint8)
    flowacc = np.array([[1, 2], [1, 2]], dtype=dtype)
    fa_ndv = flowacc.dtype.type(0)
    null_cells = np.zeros((mrows, mcols), dtype=np.bool_)
    outlet_coords = np.full((mrows, mcols, 2), -1, dtype=np.int32)
    row_dists = np.ones((n_rows, 8), dtype=np.float64) * 1000.0

    _assign_all_outlets(flowdir, flowacc, null_cells, outlet_coords,
                        mrows, mcols, kx, ky, n_rows, n_cols, flowdir_ndv, True,
                        fa_ndv, True, False,
                        True, 500.0, row_dists, DECODE_DR, DECODE_DC, DECODE_VALID,
                        DIR_DROW, DIR_DCOL)


# =========================================================================
# COTAT public function
# =========================================================================

_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)


def COTAT(
    flowdir: Grid,
    flowacc: Grid,
    k: int | tuple[int, int],
    *,
    area_threshold: float = 0.0,
    mufp: float | None = None,
) -> Grid:
    """COTAT / COTAT+ flow direction upscaler.

    Parameters
    ----------
    flowdir:
        Fine-grid flow-direction raster.  Must be ``GridType.FlowDir``.
    flowacc:
        Fine-grid flow-accumulation raster.  Must be ``GridType.FlowAcc`` with
        a supported dtype (int32/uint32/int64/uint64/float32/float64).
        Must match *flowdir* in shape, transform, and CRS.
    k:
        Upscaling factor.  A positive integer (isotropic; equivalent to
        ``(k, k)``) or a length-2 ``(kx, ky)`` tuple of positive integers.
        *kx* is the X-axis / column scale (``transform.a``); *ky* is the
        Y-axis / row scale (``transform.e``).  A coarse cell covers a fine
        window of ``ky`` rows by ``kx`` columns.
    area_threshold:
        Minimum accumulated-area gain required at a downstream cell outlet
        for that cell to be selected as the receiving cell while tracing.
    mufp:
        Minimum Upstream Flow Path in **metres** for the COTAT+ outlet-selection
        scheme.  When ``None`` (default), the original COTAT outlet selection
        (largest-upstream-area pixel) is used.

    Returns
    -------
    Grid
        Coarse flow-direction grid (``GridType.FlowDir``, uint8).  Shape is
        ``(ceil(H/ky), ceil(W/kx))``.  The output transform is
        ``Affine(t.a * kx, t.b, t.c, t.d, t.e * ky, t.f)``.

    Notes
    -----
    Both inputs are read in place; nothing is copied or padded. Coarse
    shape is ``(ceil(H/ky), ceil(W/kx))``. Scans stop at the raster edge,
    so a partial boundary cell only visits pixels that exist. Cell-exit
    tests still use the full logical window ``[i*ky, i*ky+ky)`` by
    ``[j*kx, j*kx+kx)``: a fine step that would have landed on the old
    padded border stays inside the cell and is not an outlet.
    Flow-accumulation nodata is read as 0 inside the kernels (NaN by a
    self-inequality, every other sentinel by a comparison in the array
    dtype; the sentinel is cast with ``fa.dtype.type`` first, which keeps
    uint64/int64 values exact). A coarse cell is null when every in-extent
    pixel is flow-direction nodata (255).
    """
    check_type(flowdir, GridType.FlowDir)
    check_type(flowacc, GridType.FlowAcc)
    check_dtype(flowacc, allowed=_FLOWACC_DTYPES)
    check_shape_match(flowdir, flowacc)
    check_transform_match(flowdir, flowacc)
    check_crs_match(flowdir, flowacc)
    kx, ky = normalize_k(k)

    orig_nrows, orig_ncols = flowdir.shape
    mrows = math.ceil(orig_nrows / ky)
    mcols = math.ceil(orig_ncols / kx)

    fd = flowdir.array
    fa = flowacc.array
    ndv = flowacc.meta.nodata
    # Same-dtype sentinel. Unused when fa_has_ndv is false; a NaN nodata is
    # detected with ``val != val``, so the scalar itself is a dummy zero.
    if ndv is None:
        fa_ndv = fa.dtype.type(0)
        fa_has_ndv = False
        fa_ndv_is_nan = False
    elif isinstance(ndv, float) and np.isnan(ndv):
        fa_ndv = fa.dtype.type(0)
        fa_has_ndv = True
        fa_ndv_is_nan = True
    else:
        fa_ndv = fa.dtype.type(ndv)
        fa_has_ndv = True
        fa_ndv_is_nan = False

    # A cell is null when every in-extent fine pixel is flowdir nodata.
    null_cells = _null_cells(fd, mrows, mcols, kx, ky, orig_nrows, orig_ncols)

    flowdir_ndv = np.uint8(255)
    has_ndv     = True

    use_mufp = mufp is not None
    mufp_val = float(mufp) if use_mufp else 0.0

    row_dists = row_distance_table(
        flowdir.meta.transform,
        orig_nrows,
        geographic=flowdir.meta.is_geographic,
    )

    outlet_coords = np.full((mrows, mcols, 2), -1, dtype=np.int32)
    _assign_all_outlets(
        fd, fa, null_cells, outlet_coords,
        mrows, mcols, kx, ky, orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
        fa_ndv, fa_has_ndv, fa_ndv_is_nan,
        use_mufp, mufp_val, row_dists, DECODE_DR, DECODE_DC, DECODE_VALID,
        DIR_DROW, DIR_DCOL,
    )

    cells = np.full((mrows, mcols), np.uint8(255), dtype=np.uint8)
    _assign_all_directions(
        fd, fa, outlet_coords,
        cells, null_cells, kx, ky, orig_nrows, orig_ncols,
        flowdir_ndv, has_ndv, fa_ndv, fa_has_ndv, fa_ndv_is_nan,
        area_threshold,
        DECODE_DR, DECODE_DC, DECODE_VALID, ENCODE_DIR,
        mrows, mcols,
    )

    del null_cells

    _fix_intersections_numba(cells, outlet_coords, fa, mrows, mcols,
                             fa_ndv, fa_has_ndv, fa_ndv_is_nan)

    t = flowdir.meta.transform
    out_transform = Affine(t.a * kx, t.b, t.c, t.d, t.e * ky, t.f)

    return Grid.create(
        array=cells,
        type=GridType.FlowDir,
        transform=out_transform,
        crs=flowdir.meta.crs,
    )
