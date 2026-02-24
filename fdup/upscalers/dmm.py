"""
Implements DMM (Double Maximum Method)
for flow direction upscaling by Olivera et al. (2002)
https://doi.org/10.1029/2001WR000726
"""

import numpy as np
from numba import njit, prange

from fdup.base import BaseUpscaler, DIR_DROW, DIR_DCOL, DECODE_DR, DECODE_DC, ENCODE_DIR


# =========================================================================
# JIT helper functions (module-level so Numba can inline / cache them)
# =========================================================================

@njit(cache=True)
def _nb_check_null(valid_mask, ci, cj, k, shift):
    """True when A-grid cell (ci, cj) contains no valid fine-grid pixels."""
    r0 = ci * k + shift
    c0 = cj * k + shift
    for r in range(r0, r0 + k):
        for c in range(c0, c0 + k):
            if valid_mask[r, c]:
                return False
    return True


@njit(cache=True)
def _nb_cell_max_flowacc(flowacc, valid_mask, ci, cj, k, shift):
    """Maximum flowacc value inside A-grid cell (ci, cj).

    Returns -inf when the cell contains no valid pixel.
    """
    r0 = ci * k + shift
    c0 = cj * k + shift
    found = False
    best = np.float64(0.0)
    for r in range(r0, r0 + k):
        for c in range(c0, c0 + k):
            if valid_mask[r, c]:
                v = np.float64(flowacc[r, c])
                if not found or v > best:
                    best = v
                    found = True
    return best if found else -np.inf


@njit(cache=True)
def _nb_find_max_pixel(flowacc, valid_mask, ci, cj, k, d, nrows, ncols):
    """Return the global (r, c) of the highest-flowacc pixel inside cell
    (ci, cj) accessed with offset *d*.

    Tie-break: smallest squared Euclidean distance to cell centre, then
    min local row, then min local col.
    Returns (-1, -1) when no valid pixel exists.
    """
    r0 = ci * k + d
    c0 = cj * k + d
    rb = r0 + k if r0 + k <= nrows else nrows
    cb = c0 + k if c0 + k <= ncols else ncols
    if r0 < 0 or c0 < 0 or r0 >= rb or c0 >= cb:
        return -1, -1

    center_r = (k - 1) * 0.5
    center_c = (k - 1) * 0.5

    found      = False
    best_val   = np.float64(0.0)
    best_dist2 = np.inf
    best_lr    = k + 1
    best_lc    = k + 1
    best_r     = -1
    best_c     = -1

    for r in range(r0, rb):
        for c in range(c0, cb):
            if not valid_mask[r, c]:
                continue
            v  = np.float64(flowacc[r, c])
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
def _nb_receiver(cells, ci, cj, mrows, mcols, valid_mask, k, shift,
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
    if _nb_check_null(valid_mask, ni, nj, k, shift):
        return -1, -1
    return ni, nj


# =========================================================================
# Main algorithm kernels
# =========================================================================

@njit(cache=True, parallel=True)
def _nb_assign_cell_directions(flowacc, valid_mask, cells, k, shift, mrows, mcols,
                                dir_drow, dir_dcol, encode_dir):
    """Assign a D8 flow direction to every coarse A-grid cell.

    Rows are processed in parallel (prange); each row writes only to its own
    slice of cells[i, :] and reads flowacc / valid_mask read-only.
    """
    nrows = flowacc.shape[0]
    ncols = flowacc.shape[1]

    for i in prange(mrows):
        for j in range(mcols):

            if _nb_check_null(valid_mask, i, j, k, shift):
                cells[i, j] = np.uint8(255)
                continue

            discharge_i = i
            discharge_j = j

            pi, pj = _nb_find_max_pixel(
                flowacc, valid_mask, i, j, k, shift, nrows, ncols
            )
            if pi >= 0:
                bi = pi // k
                bj = pj // k

                dpi, dpj = _nb_find_max_pixel(
                    flowacc, valid_mask, bi, bj, k, 0, nrows, ncols
                )
                if dpi >= 0:
                    discharge_i = (dpi - shift) // k
                    discharge_j = (dpj - shift) // k

            current_max = _nb_cell_max_flowacc(flowacc, valid_mask, i, j, k, shift)

            if discharge_i == i and discharge_j == j:
                best_max = current_max
                best_i   = i
                best_j   = j
                for d_idx in range(8):
                    ni = i + dir_drow[d_idx]
                    nj = j + dir_dcol[d_idx]
                    if ni < 0 or ni >= mrows or nj < 0 or nj >= mcols:
                        continue
                    if _nb_check_null(valid_mask, ni, nj, k, shift):
                        continue
                    nbr = _nb_cell_max_flowacc(flowacc, valid_mask, ni, nj, k, shift)
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
                        flowacc, valid_mask, discharge_i, discharge_j, k, shift
                    )
                    if recv_max <= current_max:
                        discharge_i = i
                        discharge_j = j

            cells[i, j] = encode_dir[np.intp(discharge_i - i + 1),
                                     np.intp(discharge_j - j + 1)]


@njit(cache=True)
def _nb_fix_counter_flows(cells, flowacc, valid_mask, k, shift, mrows, mcols):
    """Resolve mutual head-to-head flow pairs."""
    SINK = np.uint8(0)
    for i in range(mrows - 1):
        for j in range(mcols - 1):

            if cells[i, j] == np.uint8(1) and cells[i, j + 1] == np.uint8(16):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i, j,     k, shift) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i, j + 1, k, shift)):
                    cells[i, j]     = SINK
                else:
                    cells[i, j + 1] = SINK

            if cells[i, j] == np.uint8(4) and cells[i + 1, j] == np.uint8(64):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i,     j, k, shift) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i + 1, j, k, shift)):
                    cells[i, j]     = SINK
                else:
                    cells[i + 1, j] = SINK

            if cells[i, j] == np.uint8(2) and cells[i + 1, j + 1] == np.uint8(32):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i,     j,     k, shift) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i + 1, j + 1, k, shift)):
                    cells[i, j]         = SINK
                else:
                    cells[i + 1, j + 1] = SINK

            if (i >= 1
                    and cells[i, j]         == np.uint8(128)
                    and cells[i - 1, j + 1] == np.uint8(8)):
                if (_nb_cell_max_flowacc(flowacc, valid_mask, i,     j,     k, shift) >
                        _nb_cell_max_flowacc(flowacc, valid_mask, i - 1, j + 1, k, shift)):
                    cells[i, j]         = SINK
                else:
                    cells[i - 1, j + 1] = SINK


@njit(cache=True)
def _nb_fix_intersections(cells, flowacc, valid_mask, k, shift, mrows, mcols,
                           max_passes):
    """Iteratively resolve crossing diagonal flows and local 2x2 conflicts."""
    for _ in range(max_passes):
        changed = False
        for i in range(mrows - 1):
            for j in range(mcols - 1):
                tl = _nb_cell_max_flowacc(flowacc, valid_mask, i,     j,     k, shift)
                tr = _nb_cell_max_flowacc(flowacc, valid_mask, i,     j + 1, k, shift)
                bl = _nb_cell_max_flowacc(flowacc, valid_mask, i + 1, j,     k, shift)

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
                    br = _nb_cell_max_flowacc(flowacc, valid_mask, i + 1, j + 1, k, shift)
                    if tr > br:
                        cells[i,     j] = np.uint8(1)
                    else:
                        cells[i + 1, j] = np.uint8(1)
                    changed = True

        if not changed:
            break


@njit(cache=True)
def _nb_fix_small_cycles(cells, flowacc, valid_mask, k, shift, mrows, mcols,
                          max_passes, decode_dr, decode_dc):
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
                                        path_r[p], path_c[p], k, shift
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
                        cells, cur_i, cur_j, mrows, mcols, valid_mask, k, shift,
                        decode_dr, decode_dc
                    )
                    if ni < 0:
                        break
                    cur_i = ni
                    cur_j = nj

        if not changed:
            break


@njit(cache=True)
def _nb_enforce_nodata(cells, valid_mask, k, shift, mrows, mcols):
    """Set every cell that contains no valid fine-grid pixel to nodata (255)."""
    for i in range(mrows):
        for j in range(mcols):
            if _nb_check_null(valid_mask, i, j, k, shift):
                cells[i, j] = np.uint8(255)


# =========================================================================
# JIT warm-up
# =========================================================================

def _warmup_jit(dtype=np.float64):
    """Force ahead-of-time compilation of every JIT kernel."""
    k     = 2
    shift = 1
    n     = k * 2 + 2 * shift
    mrows = mcols = n // k - 1

    flowacc = np.arange(n * n, dtype=dtype).reshape(n, n)

    valid_mask = np.ones((n, n), dtype=np.bool_)
    valid_mask[0, :]  = False
    valid_mask[-1, :] = False
    valid_mask[:, 0]  = False
    valid_mask[:, -1] = False

    cells = np.full((mrows, mcols), np.uint8(255), dtype=np.uint8)

    _nb_assign_cell_directions(flowacc, valid_mask, cells, k, shift, mrows, mcols,
                               DIR_DROW, DIR_DCOL, ENCODE_DIR)
    _nb_fix_counter_flows     (cells, flowacc, valid_mask, k, shift, mrows, mcols)
    _nb_fix_intersections     (cells, flowacc, valid_mask, k, shift, mrows, mcols, 1)
    _nb_fix_small_cycles      (cells, flowacc, valid_mask, k, shift, mrows, mcols, 1,
                               DECODE_DR, DECODE_DC)
    _nb_enforce_nodata        (cells, valid_mask, k, shift, mrows, mcols)


for _dtype in (np.uint32, np.float32, np.float64):
    _warmup_jit(_dtype)


# =========================================================================
# DMM upscaler class
# =========================================================================

class DMM(BaseUpscaler):
    """Double Maximum Method flow direction upscaler.

    Uses an A-grid / B-grid displacement mechanism.  Requires *k* to be a
    positive even integer.
    """

    def _pad_with_shift(self, array, k, shift, pad_value):
        """Pad to a multiple of *k*, then add a symmetric half-cell frame."""
        array = self._pad_to_multiple(array, k, pad_value)
        if shift:
            array = np.pad(
                array,
                pad_width=((shift, shift), (shift, shift)),
                mode="constant",
                constant_values=pad_value,
            )
        return array

    def upscale(self, k):
        """Run DMM upscaling with scaling factor *k* (positive even int).

        Returns a copy of the resulting uint8 flow-direction array.
        """
        if not isinstance(k, int) or k <= 0 or k % 2 != 0:
            raise ValueError("Scaling factor k must be a positive even integer")
        if self._flowacc_raw is None:
            raise RuntimeError("No data loaded. Call load_flowacc() first.")

        shift = k // 2
        ndv = self._flowacc_nodata

        nodata_mask = self._build_nodata_mask(self._flowacc_raw, ndv)
        valid_mask = ~nodata_mask

        flowacc = self._pad_with_shift(
            self._flowacc_raw, k, shift,
            ndv if ndv is not None else 0,
        )
        valid_mask = self._pad_with_shift(
            valid_mask.astype(np.uint8), k, shift, 0,
        ).astype(bool)

        mrows = flowacc.shape[0] // k - 1
        mcols = flowacc.shape[1] // k - 1
        cells = np.full((mrows, mcols), self.DIR_NODATA, dtype=np.uint8)

        flowacc    = np.ascontiguousarray(flowacc)
        valid_mask = np.ascontiguousarray(valid_mask)

        _nb_assign_cell_directions(flowacc, valid_mask, cells, k, shift, mrows, mcols,
                                   DIR_DROW, DIR_DCOL, ENCODE_DIR)
        _nb_fix_counter_flows(cells, flowacc, valid_mask, k, shift, mrows, mcols)
        _nb_fix_intersections(cells, flowacc, valid_mask, k, shift, mrows, mcols, 4)
        _nb_fix_small_cycles(cells, flowacc, valid_mask, k, shift, mrows, mcols, 4,
                             DECODE_DR, DECODE_DC)
        _nb_enforce_nodata(cells, valid_mask, k, shift, mrows, mcols)

        self.cells_ = cells
        self.k_ = k
        return self.cells_.copy()
