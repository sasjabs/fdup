"""
Implements NSA (Network Scaling Algorithm)
for flow direction upscaling by Fekete et al. (2001)
https://doi.org/10.1029/2001WR900024
"""

from __future__ import annotations

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
def _aggregate_jit(fa, ndv, has_ndv, ndv_is_nan, kx, ky):
    """MAX-aggregate *fa* into coarse cells of size (*ky*, *kx*).

    Output shape is ``(ceil(H/ky), ceil(W/kx))``. Each window is clamped with
    ``min(i*ky+ky, H)`` / ``min(j*kx+kx, W)``, so partial edge cells are the
    max of the pixels that exist. Nodata is tested inline on the raw value:
    NaN by ``val != val``, any other sentinel by ``val == ndv`` in *fa*'s
    dtype. *ndv* is ignored when *has_ndv* is false; pass a zero of *fa*'s
    dtype in that case. A window with no valid pixel is nodata.
    """
    H = fa.shape[0]
    W = fa.shape[1]
    krows = (H + ky - 1) // ky
    kcols = (W + kx - 1) // kx
    flowacc_aggr = np.zeros((krows, kcols), dtype=fa.dtype)
    nodata_aggr = np.zeros((krows, kcols), dtype=np.bool_)

    for i in range(krows):
        r1 = min(i * ky + ky, H)
        for j in range(kcols):
            c1 = min(j * kx + kx, W)
            found_valid = False
            max_val = ndv
            for r in range(i * ky, r1):
                for c in range(j * kx, c1):
                    val = fa[r, c]
                    if has_ndv:
                        if ndv_is_nan:
                            if val != val:
                                continue
                        elif val == ndv:
                            continue
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
    """Trigger JIT compilation for the given *dtype*.

    Compiles the sentinel path (a zero of *dtype* marks one pixel nodata).
    The NaN branch is part of the same kernel, so this covers both flags.
    """
    kx = ky = 2
    mock_fa = np.array([[1,  2,  3,  4],
                         [5,  6,  7,  8],
                         [9,  10, 11, 12],
                         [13, 14, 15, 16]], dtype=dtype)
    ndv = mock_fa.dtype.type(0)
    mock_fa[0, 0] = ndv
    dir_dist = _dir_distances(kx, ky)
    aggr, nd = _aggregate_jit(mock_fa, ndv, True, False, kx, ky)
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
    The fine array is read in place. Coarse shape is
    ``(ceil(H/ky), ceil(W/kx))``; each window stops at the raster edge, so
    partial boundary cells contribute only the pixels that exist. Nodata is
    skipped inside the aggregate kernel: NaN by a self-inequality, every
    other sentinel by a comparison in the array dtype (the sentinel is cast
    with ``fa.dtype.type`` first, which keeps uint64/int64 values exact).
    A window with no valid pixel is nodata. New storage is the coarse
    accumulation, its nodata mask, and the flow-direction result.

    Rectangular coarse cells weight cardinal vs diagonal neighbours using
    ``hypot(drow*ky, dcol*kx)`` inter-cell distances; square cells reuse the
    unit-cell table, so direction selection is bit-identical to the isotropic
    path (see ``_dir_distances``).
    """
    check_type(flowacc, GridType.FlowAcc)
    check_dtype(flowacc, allowed=_FLOWACC_DTYPES)
    kx, ky = normalize_k(k)

    fa = flowacc.array
    ndv = flowacc.meta.nodata
    # Same-dtype sentinel. Unused when has_ndv is false; a NaN nodata is
    # detected with ``val != val``, so the scalar itself is a dummy zero.
    if ndv is None:
        ndv_scalar = fa.dtype.type(0)
        has_ndv = False
        ndv_is_nan = False
    elif isinstance(ndv, float) and np.isnan(ndv):
        ndv_scalar = fa.dtype.type(0)
        has_ndv = True
        ndv_is_nan = True
    else:
        ndv_scalar = fa.dtype.type(ndv)
        has_ndv = True
        ndv_is_nan = False

    flowacc_aggr, nodata_aggr = _aggregate_jit(
        fa, ndv_scalar, has_ndv, ndv_is_nan, kx, ky,
    )

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
