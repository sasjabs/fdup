"""
Implements DMM (Double Maximum Method)
for flow direction upscaling by Olivera et al. (2002)
https://doi.org/10.1029/2001WR000726
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from numba import njit, prange

from fdup._core.d8 import (
    DECODE_DC, DECODE_DR, DIR_DCOL, DIR_DROW, ENCODE_DIR,
)
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_dtype, check_type, normalize_k


# =========================================================================
# JIT helper functions (module-level so Numba can inline / cache them)
# =========================================================================

@njit(cache=True)
def _nb_check_null(valid_mask, ci, cj, kx, ky, shift_c, shift_r):
    """True when A-grid cell (ci, cj) contains no valid fine-grid pixels.

    The cell window is *ky* rows × *kx* cols of logical coordinates
    starting at ``(ci*ky + shift_r, cj*kx + shift_c)``. Logical
    ``(r, c)`` maps to raw ``(r - shift_r, c - shift_c)``; a coordinate
    outside ``valid_mask`` is invalid (the former padded border).
    """
    r0 = ci * ky + shift_r
    c0 = cj * kx + shift_c
    H = valid_mask.shape[0]
    W = valid_mask.shape[1]
    for r in range(r0, r0 + ky):
        for c in range(c0, c0 + kx):
            if r < shift_r or c < shift_c:
                continue
            rr = r - shift_r
            cc = c - shift_c
            if rr >= H or cc >= W:
                continue
            if valid_mask[rr, cc]:
                return False
    return True


@njit(cache=True)
def _nb_cell_max_flowacc(flowacc, valid_mask, ci, cj, kx, ky, shift_c, shift_r):
    """Maximum flowacc value inside A-grid cell (ci, cj).

    The cell window is *ky* rows × *kx* cols of logical coordinates
    starting at ``(ci*ky + shift_r, cj*kx + shift_c)``. Logical
    ``(r, c)`` maps to raw ``(r - shift_r, c - shift_c)``; a coordinate
    outside the fine grid is invalid. Returns -inf when the cell
    contains no valid pixel.
    """
    r0 = ci * ky + shift_r
    c0 = cj * kx + shift_c
    H = flowacc.shape[0]
    W = flowacc.shape[1]
    found = False
    best = np.float64(0.0)
    for r in range(r0, r0 + ky):
        for c in range(c0, c0 + kx):
            if r < shift_r or c < shift_c:
                continue
            rr = r - shift_r
            cc = c - shift_c
            if rr >= H or cc >= W:
                continue
            if valid_mask[rr, cc]:
                v = np.float64(flowacc[rr, cc])
                if not found or v > best:
                    best = v
                    found = True
    return best if found else -np.inf


@njit(cache=True)
def _nb_find_max_pixel(flowacc, valid_mask, ci, cj, kx, ky, dr, dc, nrows, ncols,
                       shift_r, shift_c):
    """Return the logical (r, c) of the highest-flowacc pixel inside cell
    (ci, cj) accessed with row/column offsets *dr* / *dc*.

    *nrows* / *ncols* are the logical dimensions (fine shape plus one
    shift of border on each side). Logical ``(r, c)`` maps to raw
    ``(r - shift_r, c - shift_c)``; out-of-range raw coordinates are
    invalid. Returned coordinates stay in that logical frame.

    Tie-break: smallest squared Euclidean distance to cell centre, then
    min local row, then min local col.
    Returns (-1, -1) when no valid pixel exists.
    """
    r0 = ci * ky + dr
    c0 = cj * kx + dc
    rb = r0 + ky if r0 + ky <= nrows else nrows
    cb = c0 + kx if c0 + kx <= ncols else ncols
    if r0 < 0 or c0 < 0 or r0 >= rb or c0 >= cb:
        return -1, -1

    H = flowacc.shape[0]
    W = flowacc.shape[1]
    center_r = (ky - 1) * 0.5
    center_c = (kx - 1) * 0.5

    found      = False
    best_val   = np.float64(0.0)
    best_dist2 = np.inf
    best_lr    = ky + 1
    best_lc    = kx + 1
    best_r     = -1
    best_c     = -1

    for r in range(r0, rb):
        for c in range(c0, cb):
            if r < shift_r or c < shift_c:
                continue
            rr = r - shift_r
            cc = c - shift_c
            if rr >= H or cc >= W or not valid_mask[rr, cc]:
                continue
            v  = np.float64(flowacc[rr, cc])
            lr = r - r0
            lc = c - c0
            dist2 = (lr - center_r) ** 2 + (lc - center_c) ** 2

            better = False
            if not found:
                better = True
            elif v > best_val:
                better = True
            elif v == best_val:
                if dist2 < best_dist2:
                    better = True
                elif dist2 == best_dist2:
                    if lr < best_lr or (lr == best_lr and lc < best_lc):
                        better = True

            if better:
                found      = True
                best_val   = v
                best_dist2 = dist2
                best_lr    = lr
                best_lc    = lc
                best_r     = r
                best_c     = c

    return best_r, best_c


@njit(cache=True)
def _nb_receiver(cells, ci, cj, mrows, mcols, valid_mask, kx, ky, shift_c, shift_r,
                 decode_dr, decode_dc):
    """Follow direction stored in cells[ci, cj] to the receiver cell.

    Returns (-1, -1) on dead end (sink, nodata, out-of-bounds, or null cell).
    """
    d = np.int32(cells[ci, cj])
    if d < 1 or d > 128:
        return -1, -1
    dr = decode_dr[d]
    dc = decode_dc[d]
    if dr == 0 and dc == 0:
        return -1, -1
    ni = ci + dr
    nj = cj + dc
    if ni < 0 or ni >= mrows or nj < 0 or nj >= mcols:
        return -1, -1
    if _nb_check_null(valid_mask, ni, nj, kx, ky, shift_c, shift_r):
        return -1, -1
    return ni, nj


# =========================================================================
# Main algorithm kernels
# =========================================================================

@njit(cache=True, parallel=True)
def _nb_assign_cell_directions(flowacc, valid_mask, cells, kx, ky, shift_c, shift_r,
                                mrows, mcols, dir_drow, dir_dcol, encode_dir):
    """Assign a D8 flow direction to every coarse A-grid cell.

    Rows are processed in parallel (prange); each row writes only to its own
    slice of cells[i, :] and reads flowacc / valid_mask read-only.
    Fine pixels map to B-grid cells as ``(pi // ky, pj // kx)``; the
    discharge cell is ``((dpi - shift_r) // ky, (dpj - shift_c) // kx)``.
    ``pi`` and ``dpi`` are logical coordinates (raw index plus the shift).
    """
    # Logical frame: the fine grid plus the A/B-grid shift on every side.
    # Accessors subtract the shift; the arrays themselves are unpadded.
    nrows = flowacc.shape[0] + 2 * shift_r
    ncols = flowacc.shape[1] + 2 * shift_c

    for i in prange(mrows):
        for j in range(mcols):

            if _nb_check_null(valid_mask, i, j, kx, ky, shift_c, shift_r):
                cells[i, j] = np.uint8(255)
                continue

            discharge_i = i
            discharge_j = j

            pi, pj = _nb_find_max_pixel(
                flowacc, valid_mask, i, j, kx, ky, shift_r, shift_c, nrows, ncols,
                shift_r, shift_c,
            )
            if pi >= 0:
                bi = pi // ky
                bj = pj // kx

                dpi, dpj = _nb_find_max_pixel(
                    flowacc, valid_mask, bi, bj, kx, ky, 0, 0, nrows, ncols,
                    shift_r, shift_c,
                )
                if dpi >= 0:
                    discharge_i = (dpi - shift_r) // ky
                    discharge_j = (dpj - shift_c) // kx

            current_max = _nb_cell_max_flowacc(
                flowacc, valid_mask, i, j, kx, ky, shift_c, shift_r
            )

            if discharge_i == i and discharge_j == j:
                best_max = current_max
                best_i   = i
                best_j   = j
                for d_idx in range(8):
                    ni = i + dir_drow[d_idx]
                    nj = j + dir_dcol[d_idx]
                    if ni < 0 or ni >= mrows or nj < 0 or nj >= mcols:
                        continue
                    if _nb_check_null(valid_mask, ni, nj, kx, ky, shift_c, shift_r):
                        continue
                    nbr = _nb_cell_max_flowacc(
                        flowacc, valid_mask, ni, nj, kx, ky, shift_c, shift_r
                    )
                    if nbr > best_max:
                        best_max = nbr
                        best_i   = ni
                        best_j   = nj
                discharge_i = best_i
                discharge_j = best_j
            else:
                di = discharge_i - i
                dj = discharge_j - j
                if abs(di) > 1 or abs(dj) > 1:
                    discharge_i = i
                    discharge_j = j
                else:
                    recv_max = _nb_cell_max_flowacc(
                        flowacc, valid_mask, discharge_i, discharge_j,
                        kx, ky, shift_c, shift_r
                    )
                    if recv_max <= current_max:
                        discharge_i = i
                        discharge_j = j

            cells[i, j] = encode_dir[np.intp(discharge_i - i + 1),
                                     np.intp(discharge_j - j + 1)]


@njit(cache=True)
def _nb_fix_counter_flows(cells, flowacc, valid_mask, kx, ky, shift_c, shift_r,
                           mrows, mcols):
    """Resolve mutual head-to-head flow pairs."""
    SINK = np.uint8(0)
    for i in range(mrows - 1):
        for j in range(mcols - 1):

            if cells[i, j] == np.uint8(1) and cells[i, j + 1] == np.uint8(16):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i, j,
                                          kx, ky, shift_c, shift_r) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i, j + 1,
                                              kx, ky, shift_c, shift_r)):
                    cells[i, j]     = SINK
                else:
                    cells[i, j + 1] = SINK

            if cells[i, j] == np.uint8(4) and cells[i + 1, j] == np.uint8(64):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i, j,
                                          kx, ky, shift_c, shift_r) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i + 1, j,
                                              kx, ky, shift_c, shift_r)):
                    cells[i, j]     = SINK
                else:
                    cells[i + 1, j] = SINK

            if cells[i, j] == np.uint8(2) and cells[i + 1, j + 1] == np.uint8(32):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i, j,
                                          kx, ky, shift_c, shift_r) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i + 1, j + 1,
                                              kx, ky, shift_c, shift_r)):
                    cells[i, j]         = SINK
                else:
                    cells[i + 1, j + 1] = SINK

            if (i >= 1
                    and cells[i, j]         == np.uint8(128)
                    and cells[i - 1, j + 1] == np.uint8(8)):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i, j,
                                          kx, ky, shift_c, shift_r) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i - 1, j + 1,
                                              kx, ky, shift_c, shift_r)):
                    cells[i, j]         = SINK
                else:
                    cells[i - 1, j + 1] = SINK


@njit(cache=True)
def _nb_fix_intersections(cells, flowacc, valid_mask, kx, ky, shift_c, shift_r,
                           mrows, mcols, max_passes):
    """Iteratively resolve crossing diagonal flows and local 2x2 conflicts."""
    for _ in range(max_passes):
        changed = False
        for i in range(mrows - 1):
            for j in range(mcols - 1):
                tl = _nb_cell_max_flowacc(
                    flowacc, valid_mask, i, j, kx, ky, shift_c, shift_r
                )
                tr = _nb_cell_max_flowacc(
                    flowacc, valid_mask, i, j + 1, kx, ky, shift_c, shift_r
                )
                bl = _nb_cell_max_flowacc(
                    flowacc, valid_mask, i + 1, j, kx, ky, shift_c, shift_r
                )

                if cells[i, j] == np.uint8(2) and cells[i, j + 1] == np.uint8(8):
                    if tl > tr:
                        cells[i, j + 1] = np.uint8(4)
                    else:
                        cells[i, j]     = np.uint8(4)
                    changed = True

                if (cells[i + 1, j]     == np.uint8(128)
                        and cells[i + 1, j + 1] == np.uint8(32)):
                    if tl > tr:
                        cells[i + 1, j]     = np.uint8(64)
                    else:
                        cells[i + 1, j + 1] = np.uint8(64)
                    changed = True

                if (cells[i,     j + 1] == np.uint8(8)
                        and cells[i + 1, j + 1] == np.uint8(32)):
                    if tl > bl:
                        cells[i,     j + 1] = np.uint8(16)
                    else:
                        cells[i + 1, j + 1] = np.uint8(16)
                    changed = True

                if (cells[i,     j] == np.uint8(2)
                        and cells[i + 1, j] == np.uint8(128)):
                    br = _nb_cell_max_flowacc(
                        flowacc, valid_mask, i + 1, j + 1, kx, ky, shift_c, shift_r
                    )
                    if tr > br:
                        cells[i,     j] = np.uint8(1)
                    else:
                        cells[i + 1, j] = np.uint8(1)
                    changed = True

        if not changed:
            break


@njit(cache=True)
def _nb_fix_small_cycles(cells, flowacc, valid_mask, kx, ky, shift_c, shift_r,
                          mrows, mcols, max_passes, decode_dr, decode_dc):
    """Break small cycles confined within a 2x2 coarse neighbourhood.

    The cell with the lowest max flowacc in the cycle is converted to a sink.
    """
    SINK    = np.uint8(0)
    NODATA  = np.uint8(255)
    MAX_LEN = 9

    for _ in range(max_passes):
        changed = False
        for i in range(mrows):
            for j in range(mcols):
                if cells[i, j] == NODATA or cells[i, j] == SINK:
                    continue

                path_r   = np.empty(MAX_LEN, dtype=np.int32)
                path_c   = np.empty(MAX_LEN, dtype=np.int32)
                path_len = 0
                cur_i    = i
                cur_j    = j

                while True:
                    cyc_start = -1
                    for p in range(path_len):
                        if path_r[p] == cur_i and path_c[p] == cur_j:
                            cyc_start = p
                            break

                    if cyc_start >= 0:
                        cyc_len = path_len - cyc_start
                        if 2 <= cyc_len <= 4:
                            min_r = cur_i; max_r = cur_i
                            min_c = cur_j; max_c = cur_j
                            for p in range(cyc_start, path_len):
                                if path_r[p] < min_r: min_r = path_r[p]
                                if path_r[p] > max_r: max_r = path_r[p]
                                if path_c[p] < min_c: min_c = path_c[p]
                                if path_c[p] > max_c: max_c = path_c[p]
                            if max_r - min_r <= 1 and max_c - min_c <= 1:
                                w_i = -1; w_j = -1; w_acc = np.inf
                                for p in range(cyc_start, path_len):
                                    acc = _nb_cell_max_flowacc(
                                        flowacc, valid_mask,
                                        path_r[p], path_c[p],
                                        kx, ky, shift_c, shift_r
                                    )
                                    if acc < w_acc:
                                        w_acc = acc
                                        w_i   = path_r[p]
                                        w_j   = path_c[p]
                                if w_i >= 0:
                                    cells[w_i, w_j] = SINK
                                    changed = True
                        break

                    if path_len >= MAX_LEN:
                        break

                    path_r[path_len] = cur_i
                    path_c[path_len] = cur_j
                    path_len += 1

                    ni, nj = _nb_receiver(
                        cells, cur_i, cur_j, mrows, mcols, valid_mask,
                        kx, ky, shift_c, shift_r,
                        decode_dr, decode_dc
                    )
                    if ni < 0:
                        break
                    cur_i = ni
                    cur_j = nj

        if not changed:
            break


@njit(cache=True)
def _nb_enforce_nodata(cells, valid_mask, kx, ky, shift_c, shift_r, mrows, mcols):
    """Set every cell that contains no valid fine-grid pixel to nodata (255)."""
    for i in range(mrows):
        for j in range(mcols):
            if _nb_check_null(valid_mask, i, j, kx, ky, shift_c, shift_r):
                cells[i, j] = np.uint8(255)


# =========================================================================
# JIT warm-up
# =========================================================================

def _warmup(dtype=np.float64):
    """Force ahead-of-time compilation of every JIT kernel for *dtype*.

    Arrays are the unpadded fine grid. Logical dimensions are that shape
    plus the A/B-grid shift on each side, matching ``DMM``.
    """
    kx = ky = 2
    shift_c = shift_r = 1
    fine_rows = ky * 2
    fine_cols = kx * 2
    nrows = fine_rows + 2 * shift_r
    ncols = fine_cols + 2 * shift_c
    mrows = nrows // ky - 1
    mcols = ncols // kx - 1

    flowacc    = np.arange(fine_rows * fine_cols, dtype=dtype).reshape(fine_rows, fine_cols)
    valid_mask = np.ones((fine_rows, fine_cols), dtype=np.bool_)

    cells = np.full((mrows, mcols), np.uint8(255), dtype=np.uint8)

    _nb_assign_cell_directions(flowacc, valid_mask, cells, kx, ky, shift_c, shift_r,
                               mrows, mcols, DIR_DROW, DIR_DCOL, ENCODE_DIR)
    _nb_fix_counter_flows     (cells, flowacc, valid_mask, kx, ky, shift_c, shift_r,
                               mrows, mcols)
    _nb_fix_intersections     (cells, flowacc, valid_mask, kx, ky, shift_c, shift_r,
                               mrows, mcols, 1)
    _nb_fix_small_cycles      (cells, flowacc, valid_mask, kx, ky, shift_c, shift_r,
                               mrows, mcols, 1, DECODE_DR, DECODE_DC)
    _nb_enforce_nodata        (cells, valid_mask, kx, ky, shift_c, shift_r,
                               mrows, mcols)


# =========================================================================
# DMM public function
# =========================================================================

_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)


def DMM(flowacc: Grid, k: int | tuple[int, int]) -> Grid:
    """Double Maximum Method flow direction upscaler (Olivera et al. 2002).

    Parameters
    ----------
    flowacc:
        Fine-grid flow-accumulation raster.  Must be ``GridType.FlowAcc`` with
        a supported dtype (int32/uint32/int64/uint64/float32/float64).
    k:
        Upscaling factor.  A positive **even** integer (isotropic; equivalent
        to ``(k, k)``) or a length-2 ``(kx, ky)`` tuple of positive **even**
        integers.  *kx* is the X-axis / column scale (``transform.a``); *ky*
        is the Y-axis / row scale (``transform.e``).  Both axes must be even
        because the A/B-grid half-cell shift is ``kx // 2`` / ``ky // 2``.
        A coarse cell covers a fine window of ``ky`` rows by ``kx`` columns.

    Returns
    -------
    Grid
        Coarse flow-direction grid (``GridType.FlowDir``, uint8).  Shape is
        ``(H // ky, W // kx)`` (floor division; leftover fine rows/columns
        that are not a multiple of *ky*/*kx* are dropped).  The output
        transform is ``Affine(t.a * kx, t.b, t.c, t.d, t.e * ky, t.f)``.
    """
    check_type(flowacc, GridType.FlowAcc)
    check_dtype(flowacc, allowed=_FLOWACC_DTYPES)
    kx, ky = normalize_k(k, even=True)

    ndv = flowacc.meta.nodata
    fa = flowacc.array

    # A/B-grid half-cell shift. Logical coordinates are the fine grid plus
    # this many rows/cols of invalid border on each side. Accessors map
    # logical (r, c) to raw (r - shift_r, c - shift_c) and treat
    # out-of-range as invalid, so the border is not stored.
    shift_r = ky // 2
    shift_c = kx // 2

    # One unpadded bool per fine pixel. Fix passes call
    # _nb_cell_max_flowacc heavily; testing nodata inline there would
    # cost more than this mask.
    valid_mask = np.empty(fa.shape, dtype=np.bool_)
    if ndv is None:
        valid_mask.fill(True)
    elif isinstance(ndv, float) and np.isnan(ndv):
        np.isnan(fa, out=valid_mask)
        np.logical_not(valid_mask, out=valid_mask)
    else:
        np.not_equal(fa, ndv, out=valid_mask)

    H, W = fa.shape
    nrows = H + 2 * shift_r
    ncols = W + 2 * shift_c
    mrows = nrows // ky - 1   # == H // ky
    mcols = ncols // kx - 1   # == W // kx
    cells = np.full((mrows, mcols), np.uint8(255), dtype=np.uint8)

    _nb_assign_cell_directions(fa, valid_mask, cells, kx, ky,
                               shift_c, shift_r, mrows, mcols,
                               DIR_DROW, DIR_DCOL, ENCODE_DIR)
    _nb_fix_counter_flows(cells, fa, valid_mask, kx, ky,
                          shift_c, shift_r, mrows, mcols)
    _nb_fix_intersections(cells, fa, valid_mask, kx, ky,
                          shift_c, shift_r, mrows, mcols, 4)
    _nb_fix_small_cycles(cells, fa, valid_mask, kx, ky,
                         shift_c, shift_r, mrows, mcols, 4,
                         DECODE_DR, DECODE_DC)
    _nb_enforce_nodata(cells, valid_mask, kx, ky, shift_c, shift_r,
                       mrows, mcols)

    t = flowacc.meta.transform
    out_transform = Affine(t.a * kx, t.b, t.c, t.d, t.e * ky, t.f)

    return Grid.create(
        array=cells,
        type=GridType.FlowDir,
        transform=out_transform,
        crs=flowacc.meta.crs,
    )
