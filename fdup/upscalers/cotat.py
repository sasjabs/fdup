"""
Implements COTAT (Cell Outlet Tracing with an Area Threshold)
flow direction upscaling algorithm by Reed (2003)
https://doi.org/10.1029/2003WR001989
"""

import numpy as np
from numba import njit, prange

from fdup.base import BaseUpscaler, DECODE_DR, DECODE_DC, DECODE_VALID, ENCODE_DIR


# =========================================================================
# JIT kernels
# =========================================================================

@njit(cache=True)
def _find_largest_in_cell(flowacc, i, j, k):
    """Find pixel with largest flow accumulation in cell (i, j)."""
    r0 = i * k
    c0 = j * k
    best_r = r0
    best_c = c0
    best_val = np.float64(flowacc[r0, c0])
    for r in range(r0, r0 + k):
        for c in range(c0, c0 + k):
            v = np.float64(flowacc[r, c])
            if v > best_val:
                best_val = v
                best_r = r
                best_c = c
    return best_r, best_c


@njit(cache=True, parallel=True)
def _assign_all_outlets(flowacc, null_cells, outlet_coords, mrows, mcols, k):
    """Assign outlet pixels for all non-null cells (parallel over rows)."""
    for i in prange(mrows):
        for j in range(mcols):
            if not null_cells[i, j]:
                r, c = _find_largest_in_cell(flowacc, i, j, k)
                outlet_coords[i, j, 0] = r
                outlet_coords[i, j, 1] = c


@njit(cache=True)
def _trace_cell_direction(flowdir, flowacc, outlet_coords, ci, cj, k,
                          orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                          area_threshold, decode_dr, decode_dc, decode_valid,
                          encode_dir):
    """Trace flow from a cell's outlet pixel and return its D8 direction."""
    cr = np.intp(outlet_coords[ci, cj, 0])
    cc = np.intp(outlet_coords[ci, cj, 1])
    latest_r = cr
    latest_c = cc
    init_area = np.float64(flowacc[cr, cc])
    area_diff = 0.0

    while area_diff <= area_threshold:
        fdir = flowdir[cr, cc]

        if fdir < 1 or fdir > 128:
            break
        fdir_idx = np.intp(fdir)
        if not decode_valid[fdir_idx]:
            break

        prev_r = cr
        prev_c = cc
        cr = cr + np.intp(decode_dr[fdir_idx])
        cc = cc + np.intp(decode_dc[fdir_idx])

        if cr < 0 or cr >= orig_nrows or cc < 0 or cc >= orig_ncols:
            latest_r = prev_r
            latest_c = prev_c
            break

        if has_ndv and flowdir[cr, cc] == flowdir_ndv:
            latest_r = prev_r
            latest_c = prev_c
            break

        cur_ci = cr // k
        cur_cj = cc // k
        if abs(cur_ci - ci) > 1 or abs(cur_cj - cj) > 1:
            latest_r = prev_r
            latest_c = prev_c
            break

        out_r = np.intp(outlet_coords[cur_ci, cur_cj, 0])
        out_c = np.intp(outlet_coords[cur_ci, cur_cj, 1])
        if out_r == cr and out_c == cc:
            latest_r = cr
            latest_c = cc
            current_area = np.float64(flowacc[cr, cc])
            area_diff = current_area - init_area
        elif cur_ci != prev_r // k or cur_cj != prev_c // k:
            latest_r = prev_r
            latest_c = prev_c

    discharge_ci = latest_r // k
    discharge_cj = latest_c // k
    diff_r = discharge_ci - ci
    diff_c = discharge_cj - cj
    if diff_r < -1 or diff_r > 1 or diff_c < -1 or diff_c > 1:
        return np.uint8(0)
    return encode_dir[diff_r + 1, diff_c + 1]


@njit(cache=True, parallel=True)
def _assign_all_directions(flowdir, flowacc, outlet_coords, cells, null_cells,
                           k, orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                           area_threshold, decode_dr, decode_dc, decode_valid,
                           encode_dir, mrows, mcols):
    """Assign D8 directions for all non-null cells (parallel over rows)."""
    for i in prange(mrows):
        for j in range(mcols):
            if not null_cells[i, j]:
                cells[i, j] = _trace_cell_direction(
                    flowdir, flowacc, outlet_coords, i, j, k,
                    orig_nrows, orig_ncols, flowdir_ndv, has_ndv,
                    area_threshold, decode_dr, decode_dc, decode_valid,
                    encode_dir)


@njit(cache=True)
def _fix_intersections_numba(cells, outlet_coords, flowacc, mrows, mcols):
    """Fix diagonal flow intersections."""
    for i in range(mrows - 1):
        for j in range(mcols - 1):
            if cells[i + 1, j] == 128 and cells[i + 1, j + 1] == 32:
                r0, c0 = outlet_coords[i, j, 0], outlet_coords[i, j, 1]
                r1, c1 = outlet_coords[i, j + 1, 0], outlet_coords[i, j + 1, 1]
                if flowacc[r0, c0] > flowacc[r1, c1]:
                    cells[i + 1, j] = 64
                else:
                    cells[i + 1, j + 1] = 64

            if cells[i, j] == 2 and cells[i, j + 1] == 8:
                r0, c0 = outlet_coords[i + 1, j + 1, 0], outlet_coords[i + 1, j + 1, 1]
                r1, c1 = outlet_coords[i + 1, j, 0], outlet_coords[i + 1, j, 1]
                if flowacc[r0, c0] > flowacc[r1, c1]:
                    cells[i, j + 1] = 4
                else:
                    cells[i, j] = 4

            if cells[i, j + 1] == 8 and cells[i + 1, j + 1] == 32:
                r0, c0 = outlet_coords[i, j, 0], outlet_coords[i, j, 1]
                r1, c1 = outlet_coords[i + 1, j, 0], outlet_coords[i + 1, j, 1]
                if flowacc[r0, c0] > flowacc[r1, c1]:
                    cells[i, j + 1] = 16
                else:
                    cells[i + 1, j + 1] = 16

            if cells[i, j] == 2 and cells[i + 1, j] == 128:
                r0, c0 = outlet_coords[i, j + 1, 0], outlet_coords[i, j + 1, 1]
                r1, c1 = outlet_coords[i + 1, j + 1, 0], outlet_coords[i + 1, j + 1, 1]
                if flowacc[r0, c0] > flowacc[r1, c1]:
                    cells[i, j] = 1
                else:
                    cells[i + 1, j] = 1


# =========================================================================
# JIT warm-up
# =========================================================================

def _warmup():
    """Pre-compile all numba kernels for uint32, float32 and float64."""
    k = 1
    mrows = mcols = 2
    n = mrows * k

    flowdir = np.zeros((n, n), dtype=np.uint8)
    null_cells = np.zeros((mrows, mcols), dtype=np.bool_)
    outlet_coords = np.full((mrows, mcols, 2), -1, dtype=np.int32)
    cells = np.full((mrows, mcols), 255, dtype=np.uint8)
    flowdir_ndv = np.uint8(255)

    for fa_dtype in (np.uint32, np.float32, np.float64):
        flowacc = np.ones((n, n), dtype=fa_dtype)
        outlet_coords[:] = -1
        cells[:] = 255

        _assign_all_outlets(flowacc, null_cells, outlet_coords, mrows, mcols, k)
        _assign_all_directions(flowdir, flowacc, outlet_coords, cells,
                               null_cells, k, n, n, flowdir_ndv, True,
                               0.0, DECODE_DR, DECODE_DC, DECODE_VALID,
                               ENCODE_DIR, mrows, mcols)
        _fix_intersections_numba(cells, outlet_coords, flowacc, mrows, mcols)


_warmup()


# =========================================================================
# COTAT upscaler class
# =========================================================================

class COTAT(BaseUpscaler):
    """COTAT flow direction upscaler.

    Requires both a fine-grid flow-direction raster and a flow-accumulation
    raster.
    """

    def __init__(self):
        super().__init__()
        self._flowdir_raw = None
        self._flowdir_nodata = None

    def load_flowdir(self, path):
        """Load a fine-grid flow-direction raster (single band).

        The rasterio profile from this raster is used as the reference
        profile for ``save()``.
        """
        array, profile, nodata = self._read_raster(path)
        self._flowdir_raw = array
        self._flowdir_nodata = nodata
        self._profile = profile

    def load_flowacc(self, path):
        """Load a flow-accumulation raster without overwriting the profile.

        The spatial reference profile is taken from the flowdir raster
        (set via ``load_flowdir``), matching the original COTAT behaviour.
        """
        array, profile, nodata = self._read_raster(path)
        self._flowacc_raw = array
        self._flowacc_nodata = nodata
        if self._profile is None:
            self._profile = profile

    def upscale(self, k, area_threshold=0.0):
        """Run COTAT upscaling.

        Parameters
        ----------
        k : int
            Scaling factor (positive integer).
        area_threshold : float
            Maximum accumulated-area difference allowed while tracing
            downstream along the fine-grid flow path.

        Returns a copy of the resulting uint8 flow-direction array.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Scaling factor k must be a positive integer")
        if self._flowdir_raw is None:
            raise RuntimeError("No flow direction data. Call load_flowdir() first.")
        if self._flowacc_raw is None:
            raise RuntimeError("No flow accumulation data. Call load_flowacc() first.")

        flowdir_array = self._flowdir_raw
        flowacc_array = self._flowacc_raw
        orig_nrows, orig_ncols = flowdir_array.shape

        flowdir_padded = self._pad_to_multiple(flowdir_array, k, pad_value=0)
        flowacc_padded = self._pad_to_multiple(flowacc_array, k, pad_value=0)

        ndv = self._flowacc_nodata
        if ndv is not None:
            flowacc_padded = np.where(flowacc_padded == ndv, 0, flowacc_padded)

        nrows, ncols = flowdir_padded.shape
        mrows = nrows // k
        mcols = ncols // k

        # Build cell-level null mask from flowdir validity
        valid = np.zeros((nrows, ncols), dtype=bool)
        valid[:orig_nrows, :orig_ncols] = True
        if self._flowdir_nodata is not None:
            valid[:orig_nrows, :orig_ncols] &= (
                flowdir_padded[:orig_nrows, :orig_ncols] != self._flowdir_nodata
            )
        valid_reshaped = valid.reshape(mrows, k, mcols, k)
        null_cells = ~np.any(valid_reshaped, axis=(1, 3))

        # Assign outlet pixels
        outlet_coords = np.full((mrows, mcols, 2), -1, dtype=np.int32)
        _assign_all_outlets(flowacc_padded, null_cells, outlet_coords, mrows, mcols, k)

        # Assign cell directions
        cells = np.full((mrows, mcols), 255, dtype=np.uint8)

        if self._flowdir_nodata is not None:
            has_ndv = True
            flowdir_ndv = flowdir_padded.dtype.type(self._flowdir_nodata)
        else:
            has_ndv = False
            flowdir_ndv = flowdir_padded.dtype.type(0)

        _assign_all_directions(
            flowdir_padded, flowacc_padded, outlet_coords,
            cells, null_cells, k, orig_nrows, orig_ncols,
            flowdir_ndv, has_ndv, area_threshold,
            DECODE_DR, DECODE_DC, DECODE_VALID, ENCODE_DIR,
            mrows, mcols,
        )

        # Fix intersections
        _fix_intersections_numba(cells, outlet_coords, flowacc_padded, mrows, mcols)

        self.cells_ = cells
        self.k_ = k
        return self.cells_.copy()
