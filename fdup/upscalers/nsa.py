"""
Implements NSA (Network Scaling Algorithm)
for flow direction upscaling by Fekete et al. (2001)
https://doi.org/10.1029/2001WR900024
"""

from __future__ import annotations

import math

import numpy as np
from affine import Affine
from numba import njit

from fdup._core.d8 import DIR_CODES, DIR_DCOL, DIR_DIST, DIR_DROW
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_dtype, check_type, normalize_k


# =========================================================================
# Coarse-cell geometry
# =========================================================================

def _dir_distances(kx: int, ky: int) -> np.ndarray:
    """Per-direction inter-cell distances for a coarse cell of *ky* rows by *kx* cols.

    Only the ratios between entries matter, since ``_flowdir_jit`` compares
    normalised gradients within a single call.

    Square cells reuse the unit-cell ``DIR_DIST`` table verbatim rather than a
    scaled copy: scaling reduces the diagonal entries to a different rounding
    (``hypot(k, k) != k * sqrt(2)`` for e.g. k = 3, 6, 11, 12), which could
    flip a near-tie between a cardinal and a diagonal neighbour.  Reusing the
    table keeps direction selection bit-identical to the isotropic path.
    """
    if kx == ky:
        return DIR_DIST
    return np.hypot(
        DIR_DROW.astype(np.float64) * ky,
        DIR_DCOL.astype(np.float64) * kx,
    )


# =========================================================================
# JIT kernels
# =========================================================================

@njit(cache=True)
def _aggregate_jit(flowacc_array, nodata_mask, kx, ky):
    """MAX-aggregate *flowacc_array* into coarse cells of size (*ky*, *kx*).

    A coarse cell covers *ky* fine rows by *kx* fine columns.
    Works for uint32, float32 and float64 inputs.
    """
    krows = flowacc_array.shape[0] // ky
    kcols = flowacc_array.shape[1] // kx
    flowacc_aggr = np.zeros((krows, kcols), dtype=flowacc_array.dtype)
    nodata_aggr  = np.zeros((krows, kcols), dtype=np.bool_)

    for i in range(krows):
        for j in range(kcols):
            found_valid = False
            max_val = flowacc_array[i * ky, j * kx]
            for di in range(ky):
                for dj in range(kx):
                    if not nodata_mask[i * ky + di, j * kx + dj]:
                        val = flowacc_array[i * ky + di, j * kx + dj]
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

def _warmup(dtype=np.float64):
    """Trigger JIT compilation for the given *dtype*."""
    kx = ky = 2
    mock_fa = np.array([[1,  2,  3,  4],
                         [5,  6,  7,  8],
                         [9,  10, 11, 12],
                         [13, 14, 15, 16]], dtype=dtype)
    mock_nd = np.zeros((4, 4), dtype=np.bool_)
    mock_nd[0, 0] = True
    dir_dist = _dir_distances(kx, ky)
    aggr, nd = _aggregate_jit(mock_fa, mock_nd, kx, ky)
    _flowdir_jit(aggr, nd, DIR_DROW, DIR_DCOL, dir_dist, DIR_CODES)


# =========================================================================
# NSA public function
# =========================================================================

_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)


def NSA(flowacc: Grid, k: int | tuple[int, int]) -> Grid:
    """Network Scaling Algorithm flow direction upscaler (Fekete et al. 2001).

    Parameters
    ----------
    flowacc:
        Fine-grid flow-accumulation raster.  Must be ``GridType.FlowAcc`` with
        a supported dtype (int32/uint32/int64/uint64/float32/float64).
    k:
        Upscaling factor.  A positive integer (isotropic; equivalent to
        ``(k, k)``) or a length-2 ``(kx, ky)`` tuple of positive integers.
        *kx* is the X-axis / column scale (``transform.a``); *ky* is the
        Y-axis / row scale (``transform.e``).  A coarse cell covers a fine
        window of ``ky`` rows by ``kx`` columns.

    Returns
    -------
    Grid
        Coarse flow-direction grid (``GridType.FlowDir``, uint8).  Shape is
        ``(ceil(H/ky), ceil(W/kx))``.  The output transform is
        ``Affine(t.a * kx, t.b, t.c, t.d, t.e * ky, t.f)``.

    Notes
    -----
    The fine-grid array is pre-copied into a
    ``(ceil(H/ky)*ky, ceil(W/kx)*kx)`` buffer filled with zeros (nodata
    sentinel = 0).  Partial boundary cells beyond the original extent are
    marked nodata via the nodata-mask buffer (filled with ``True``).

    Rectangular coarse cells weight cardinal vs diagonal neighbours using
    ``hypot(drow*ky, dcol*kx)`` inter-cell distances; square cells reuse the
    unit-cell table, so direction selection is bit-identical to the isotropic
    path (see ``_dir_distances``).
    """
    check_type(flowacc, GridType.FlowAcc)
    check_dtype(flowacc, allowed=_FLOWACC_DTYPES)
    kx, ky = normalize_k(k)

    H, W   = flowacc.array.shape
    ndv    = flowacc.meta.nodata

    # Build nodata_mask for the original array.
    if ndv is None:
        nodata_mask = np.zeros((H, W), dtype=bool)
    elif isinstance(ndv, float) and np.isnan(ndv):
        nodata_mask = np.isnan(flowacc.array)
    else:
        nodata_mask = flowacc.array == ndv

    # Pre-allocate ceil-padded buffers.  Extra cells beyond the original
    # extent are treated as nodata (nd_buf=True, fa_buf=0).
    ceil_rows = math.ceil(H / ky)
    ceil_cols = math.ceil(W / kx)

    fa_zeroed = flowacc.array.copy()
    fa_zeroed[nodata_mask] = 0

    fa_buf = np.zeros((ceil_rows * ky, ceil_cols * kx), dtype=flowacc.array.dtype)
    nd_buf = np.ones((ceil_rows * ky, ceil_cols * kx), dtype=np.bool_)

    fa_buf[:H, :W] = fa_zeroed
    nd_buf[:H, :W] = nodata_mask

    flowacc_aggr, nodata_aggr = _aggregate_jit(fa_buf, nd_buf, kx, ky)

    dir_dist = _dir_distances(kx, ky)
    cells = _flowdir_jit(
        flowacc_aggr, nodata_aggr,
        DIR_DROW, DIR_DCOL, dir_dist, DIR_CODES,
    )

    t = flowacc.meta.transform
    out_transform = Affine(t.a * kx, t.b, t.c, t.d, t.e * ky, t.f)

    return Grid.create(
        array=cells,
        type=GridType.FlowDir,
        transform=out_transform,
        crs=flowacc.meta.crs,
    )
