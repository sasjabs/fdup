"""
Implements NSA (Network Scaling Algorithm)
for flow direction upscaling by Fekete et al. (2001)
https://doi.org/10.1029/2001WR900024
"""
import numpy as np
from numba import njit

from fdup.base import BaseUpscaler, DIR_CODES, DIR_DROW, DIR_DCOL, DIR_DIST


# =========================================================================
# JIT kernels
# =========================================================================

@njit(cache=True)
def _aggregate_jit(flowacc_array, nodata_mask, k):
    """MAX-aggregate *flowacc_array* into coarse cells of side *k*.

    Works for uint32, float32 and float64 inputs.
    """
    krows = flowacc_array.shape[0] // k
    kcols = flowacc_array.shape[1] // k
    flowacc_aggr = np.zeros((krows, kcols), dtype=flowacc_array.dtype)
    nodata_aggr  = np.zeros((krows, kcols), dtype=np.bool_)

    for i in range(krows):
        for j in range(kcols):
            found_valid = False
            max_val = flowacc_array[i * k, j * k]
            for di in range(k):
                for dj in range(k):
                    if not nodata_mask[i * k + di, j * k + dj]:
                        val = flowacc_array[i * k + di, j * k + dj]
                        if (not found_valid) or (val > max_val):
                            max_val = val
                            found_valid = True
            if found_valid:
                flowacc_aggr[i, j] = max_val
            else:
                nodata_aggr[i, j] = True

    return flowacc_aggr, nodata_aggr


@njit(cache=True)
def _flowdir_jit(flowacc_aggr, nodata_aggr,
                 dir_drow, dir_dcol, dir_dist, dir_codes):
    """Assign D8 flow directions from coarse flow-accumulation gradients.

    The steepest upward gradient among valid neighbours determines the
    direction.  Gradient is normalised by inter-cell distance so cardinal
    and diagonal neighbours are treated consistently.
    """
    krows   = flowacc_aggr.shape[0]
    kcols   = flowacc_aggr.shape[1]
    n_dirs  = dir_codes.shape[0]
    cells   = np.full((krows, kcols), np.uint8(255), dtype=np.uint8)

    for i in range(krows):
        for j in range(kcols):
            if nodata_aggr[i, j]:
                continue

            cell_val = np.float64(flowacc_aggr[i, j])
            max_grad = np.float64(0.0)
            best_dir = np.uint8(0)

            for d in range(n_dirs):
                ni = i + dir_drow[d]
                nj = j + dir_dcol[d]
                if ni < 0 or nj < 0 or ni >= krows or nj >= kcols:
                    continue
                if nodata_aggr[ni, nj]:
                    continue
                grad = (np.float64(flowacc_aggr[ni, nj]) - cell_val) / dir_dist[d]
                if grad > max_grad:
                    max_grad = grad
                    best_dir = dir_codes[d]

            cells[i, j] = best_dir

    return cells


# =========================================================================
# JIT warm-up
# =========================================================================

def _warmup_jit():
    """Trigger JIT compilation for uint32, float32 and float64 inputs."""
    k       = 2
    mock_fa = np.array([[1,  2,  3,  4],
                         [5,  6,  7,  8],
                         [9,  10, 11, 12],
                         [13, 14, 15, 16]], dtype=np.int32)
    mock_nd = np.zeros((4, 4), dtype=np.bool_)
    mock_nd[0, 0] = True

    for dtype in (np.uint32, np.float32, np.float64):
        fa       = mock_fa.astype(dtype)
        aggr, nd = _aggregate_jit(fa, mock_nd, k)
        _flowdir_jit(aggr, nd, DIR_DROW, DIR_DCOL, DIR_DIST, DIR_CODES)


_warmup_jit()


# =========================================================================
# NSA upscaler class
# =========================================================================

class NSA(BaseUpscaler):
    """Network Scaling Algorithm flow direction upscaler."""

    def __init__(self):
        super().__init__()
        self._nodata_mask_padded = None

    def upscale(self, k):
        """Run NSA upscaling with scaling factor *k* (positive int).

        Returns a copy of the resulting uint8 flow-direction array.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Scaling factor k must be a positive integer")
        if self._flowacc_raw is None and self._flowacc_padded is None:
            raise RuntimeError("No data loaded. Call load_flowacc() first.")

        if self._flowacc_raw is not None:
            # First call: build padded state from the raw array, then free it.
            ndv = self._flowacc_nodata
            nodata_mask = self._build_nodata_mask(self._flowacc_raw, ndv)
            fa = self._flowacc_raw.copy()
            fa[nodata_mask] = 0
            self._orig_shape = self._flowacc_raw.shape
            self._flowacc_padded = self._pad_to_multiple(fa, k, pad_value=0)
            self._nodata_mask_padded = self._pad_to_multiple(
                nodata_mask.astype(np.uint8), k, pad_value=1)
            self._flowacc_raw = None
            self._padded_k = k
        elif k != self._padded_k:
            # k changed: crop to original extent and re-pad.
            self._flowacc_padded = self._repad(self._flowacc_padded, self._orig_shape, k)
            self._nodata_mask_padded = self._repad(
                self._nodata_mask_padded, self._orig_shape, k, pad_value=1)
            self._padded_k = k

        flowacc_aggr, nodata_aggr = _aggregate_jit(
            self._flowacc_padded, self._nodata_mask_padded.astype(bool), k)
        cells = _flowdir_jit(
            flowacc_aggr, nodata_aggr,
            DIR_DROW, DIR_DCOL, DIR_DIST, DIR_CODES,
        )

        self.cells_ = cells
        self.k_ = k
        return self.cells_.copy()
