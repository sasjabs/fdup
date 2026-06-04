"""D8 flow direction from a DEM Grid.

Public API
----------
d8(dem, spherical=True) -> Grid
    Compute ESRI-encoded D8 flow directions.  Returns a FlowDir Grid.

_warmup(dtype) -> None
    Pre-compile numba kernels for all supported DEM dtypes.  The *dtype*
    argument is accepted but ignored; compilation covers the full DEM-dtype
    matrix internally.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fdup._core.geodesy import spherical_distance
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_dtype, check_type

_DEM_DTYPES = (np.int16, np.int32, np.int64, np.float32, np.float64)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _d8_cartesian(dem: np.ndarray, nodata: float,
                  dx: float, dy: float) -> np.ndarray:
    """D8 kernel using Euclidean distances between cell centres."""
    nrows, ncols = dem.shape
    out = np.full((nrows, ncols), np.uint8(255), dtype=np.uint8)

    dr    = ( 0,  1,  1,  1,  0, -1, -1, -1)
    dc    = ( 1,  1,  0, -1, -1, -1,  0,  1)
    codes = ( 1,  2,  4,  8, 16, 32, 64, 128)

    dd = math.sqrt(dx * dx + dy * dy)
    dists = (dx, dd, dy, dd, dx, dd, dy, dd)

    for r in prange(nrows):
        for c in range(ncols):
            z = dem[r, c]
            if z == nodata or math.isnan(z):
                continue

            best_grad = 0.0
            best_code = 0

            for d in range(8):
                nr = r + dr[d]
                nc = c + dc[d]
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    nz = dem[nr, nc]
                    if nz != nodata and not math.isnan(nz):
                        g = (z - nz) / dists[d]
                        if g > best_grad:
                            best_grad = g
                            best_code = codes[d]

            out[r, c] = np.uint8(best_code)

    return out


@njit(parallel=True, cache=True)
def _d8_spherical(dem: np.ndarray, nodata: float,
                  xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """D8 kernel using great-circle distances between cell centres.

    Uses ``fdup._core.geodesy.spherical_distance`` inlined by numba.
    ``xs`` are longitudes and ``ys`` are latitudes (decimal degrees).
    """
    nrows, ncols = dem.shape
    out = np.full((nrows, ncols), np.uint8(255), dtype=np.uint8)

    dr    = ( 0,  1,  1,  1,  0, -1, -1, -1)
    dc    = ( 1,  1,  0, -1, -1, -1,  0,  1)
    codes = ( 1,  2,  4,  8, 16, 32, 64, 128)

    for r in prange(nrows):
        lat1 = ys[r]
        for c in range(ncols):
            z = dem[r, c]
            if z == nodata or math.isnan(z):
                continue

            best_grad = 0.0
            best_code = 0
            lon1 = xs[c]

            for d in range(8):
                nr = r + dr[d]
                nc = c + dc[d]
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    nz = dem[nr, nc]
                    if nz != nodata and not math.isnan(nz):
                        dist = spherical_distance(lon1, lat1, xs[nc], ys[nr])
                        if dist > 0.0:
                            g = (z - nz) / dist
                            if g > best_grad:
                                best_grad = g
                                best_code = codes[d]

            out[r, c] = np.uint8(best_code)

    return out


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _cell_centres(transform, nrows: int, ncols: int):
    """Return ``(xs, ys)`` 1-D float64 arrays of cell-centre coordinates."""
    col_idx = np.arange(ncols, dtype=np.float64)
    row_idx = np.arange(nrows, dtype=np.float64)
    xs = transform.c + (col_idx + 0.5) * transform.a
    ys = transform.f + (row_idx + 0.5) * transform.e
    return xs, ys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def d8(dem: Grid, spherical: bool = True) -> Grid:
    """Compute ESRI-encoded D8 flow directions from a DEM Grid.

    Parameters
    ----------
    dem :
        Input DEM.  Must be ``GridType.DEM`` with dtype in
        ``{int16, int32, int64, float32, float64}``.
    spherical :
        When ``True`` **and** ``dem.meta.is_geographic`` is ``True``,
        inter-cell distances are computed with the great-circle (Vincenty)
        formula.  Otherwise Euclidean (flat-Earth) distances are used.

    Returns
    -------
    Grid
        ``GridType.FlowDir``, uint8, same transform and CRS as *dem*.
        ESRI encoding: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE,
        0=sink/mouth, 255=nodata.
    """
    check_type(dem, GridType.DEM)
    check_dtype(dem, _DEM_DTYPES)

    dem_f64 = dem.array.astype(np.float64)
    nodata_f = float(dem.meta.nodata) if dem.meta.nodata is not None else float("nan")
    nrows, ncols = dem.shape

    if spherical and dem.meta.is_geographic:
        xs, ys = _cell_centres(dem.meta.transform, nrows, ncols)
        flowdir = _d8_spherical(dem_f64, nodata_f, xs, ys)
    else:
        dx = abs(float(dem.meta.transform.a))
        dy = abs(float(dem.meta.transform.e))
        flowdir = _d8_cartesian(dem_f64, nodata_f, dx, dy)

    return Grid.create(
        array=flowdir,
        type=GridType.FlowDir,
        transform=dem.meta.transform,
        crs=dem.meta.crs,
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(dtype: np.dtype | None = None) -> None:  # noqa: ARG001
    """Pre-compile numba kernels for all DEM dtypes.

    The *dtype* argument is accepted for registry uniformity but ignored;
    compilation covers the full ``_DEM_DTYPES`` matrix.
    """
    tiny_base = np.array([
        [10.0, 9.0, 8.0, 7.0],
        [ 9.0, 5.0, 4.0, 6.0],
        [ 8.0, 4.0, 3.0, 5.0],
        [ 7.0, 6.0, 5.0, 4.0],
    ], dtype=np.float64)
    nd = -9999.0
    xs = np.array([10.0, 10.01, 10.02, 10.03], dtype=np.float64)
    ys = np.array([50.0, 49.99, 49.98, 49.97], dtype=np.float64)

    for dt in _DEM_DTYPES:
        arr = tiny_base.astype(np.float64)
        _d8_cartesian(arr, nd, 30.0, 30.0)
        _d8_spherical(arr, nd, xs, ys)
