"""Geodesy helpers: cell areas, great-circle and Euclidean distances.

All distance functions return metres.  ``get_cell_areas`` returns:

* **km²** for geographic CRS (``geographic=True``), where cell area varies
  by latitude using the spherical-trapezoid formula.
* **CRS units²** for projected CRS (``geographic=False``), i.e.
  ``abs(transform.a * transform.e)`` without any unit conversion.  For a
  metre-based CRS this is m²; for a foot-based CRS this is ft²; callers are
  responsible for interpreting the unit correctly.

The ``geographic`` flag controls whether calculations account for latitude
distortion (True) or assume a flat projected CRS (False).
"""

from __future__ import annotations

import math

import numpy as np
from affine import Affine
from numba import njit

# Mean Earth radius (metres), matching the value used throughout the raw scripts.
_EARTH_RADIUS_M: float = 6_371_000.0

# D8 COMPASS_ORDER offsets: (dr, dc) for E SE S SW W NW N NE
_COMPASS_DR = (0,  1,  1,  1,  0, -1, -1, -1)
_COMPASS_DC = (1,  1,  0, -1, -1, -1,  0,  1)


# ---------------------------------------------------------------------------
# njit kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def spherical_distance(
    lon1_deg: float,
    lat1_deg: float,
    lon2_deg: float,
    lat2_deg: float,
) -> float:
    """Vincenty great-circle distance in metres.

    Parameters
    ----------
    lon1_deg, lat1_deg, lon2_deg, lat2_deg :
        Coordinates in decimal degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    lon1 = math.radians(lon1_deg)
    lat1 = math.radians(lat1_deg)
    lon2 = math.radians(lon2_deg)
    lat2 = math.radians(lat2_deg)
    dlon = lon2 - lon1
    y1 = (math.cos(lat2) * math.sin(dlon)) ** 2
    y2 = (math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)) ** 2
    x = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
    return _EARTH_RADIUS_M * math.atan2(math.sqrt(y1 + y2), x)


@njit(cache=True)
def euclidean_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Euclidean (flat-Earth) distance between two 2-D points.

    Parameters
    ----------
    x1, y1, x2, y2 :
        Coordinates in any consistent unit (metres for projected CRS).

    Returns
    -------
    float
        Distance in the same unit as the inputs.
    """
    return math.hypot(x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


def get_cell_areas(
    transform: Affine,
    nrows: int,
    *,
    geographic: bool,
) -> np.ndarray:
    """Per-row cell area for a raster.

    Parameters
    ----------
    transform :
        Rasterio/affine ``Affine`` transform.
    nrows :
        Number of rows in the raster.
    geographic :
        ``True`` for geographic CRS (lat/lon degrees); cell area then varies
        by latitude and is returned in **km²**.
        ``False`` for projected CRS; the area is constant across rows and is
        returned in **CRS units²** (e.g. m² for a metre-based CRS).

    Returns
    -------
    np.ndarray, shape (nrows,), dtype float64
        Per-row cell area.  Unit is **km²** when ``geographic=True`` and
        **CRS units²** when ``geographic=False``.
    """
    if geographic:
        # Spherical trapezoid area (m²) per row, then convert to km².
        dy_m = abs(transform.e) * (math.pi / 180.0) * _EARTH_RADIUS_M
        dx_deg = abs(transform.a)
        rows = np.arange(nrows, dtype=np.float64)
        lat_centers = transform.f + (rows + 0.5) * transform.e
        dx_m = dx_deg * (math.pi / 180.0) * _EARTH_RADIUS_M * np.cos(np.deg2rad(lat_centers))
        return (dx_m * dy_m) / 1e6
    else:
        # Projected: constant cell area in CRS units² (no unit conversion).
        # Callers are responsible for interpreting the unit based on their CRS.
        area_crs2 = abs(transform.a * transform.e)
        return np.full(nrows, area_crs2, dtype=np.float64)


def row_distance_table(
    transform: Affine,
    nrows: int,
    *,
    geographic: bool,
) -> np.ndarray:
    """D8 neighbour distances (metres) for every row, in ``COMPASS_ORDER``.

    The eight columns correspond to the directions in
    ``fdup._core.d8.COMPASS_ORDER``: E, SE, S, SW, W, NW, N, NE.

    Parameters
    ----------
    transform :
        Rasterio/affine ``Affine`` transform.
    nrows :
        Number of rows in the raster.
    geographic :
        ``True`` for geographic CRS (distance varies by latitude via
        ``spherical_distance``).  ``False`` for projected CRS where pixel
        spacing is in metres (constant rows).

    Returns
    -------
    np.ndarray, shape (nrows, 8), dtype float64
        Distance in metres from a cell centre to each D8 neighbour,
        per row.
    """
    if not geographic:
        dx = abs(transform.a)
        dy = abs(transform.e)
        dd = math.hypot(dx, dy)
        # E  SE  S  SW  W  NW  N  NE
        row = np.array([dx, dd, dy, dd, dx, dd, dy, dd], dtype=np.float64)
        return np.tile(row, (nrows, 1))

    table = np.empty((nrows, 8), dtype=np.float64)
    for i in range(nrows):
        # Cell centre of (row=i, col=0) in decimal degrees
        lat1 = transform.f + (i + 0.5) * transform.e
        lon1 = transform.c + 0.5 * transform.a
        for d in range(8):
            dr = _COMPASS_DR[d]
            dc = _COMPASS_DC[d]
            lat2 = transform.f + (i + dr + 0.5) * transform.e
            lon2 = transform.c + (dc + 0.5) * transform.a
            table[i, d] = spherical_distance(lon1, lat1, lon2, lat2)
    return table
