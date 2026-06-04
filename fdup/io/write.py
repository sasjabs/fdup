"""Write a :class:`~fdup._core.types.Grid` to a GeoTIFF file."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio.transform import Affine  # re-exported for convenience

from fdup._core.types import Grid, GridType, _ALLOWED_DTYPES
from fdup._core.validation import check_dtype


def write(
    grid: Grid,
    path: Union[str, Path],
    *,
    overwrite: bool = False,
    compress: str = "deflate",
) -> None:
    """Write *grid* to a GeoTIFF at *path*.

    Parameters
    ----------
    grid:
        Grid to serialise.
    path:
        Destination file path.  The parent directory must already exist.
    overwrite:
        When ``False`` (default) and *path* already exists, raise
        :exc:`FileExistsError`.
    compress:
        GDAL compression codec passed to rasterio (default ``"deflate"``).

    Raises
    ------
    FileExistsError
        If *path* exists and *overwrite* is ``False``.
    ValueError
        If ``grid.array.dtype`` is not canonical for ``grid.meta.type``.
    """
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}. "
            "Pass overwrite=True to replace it."
        )

    grid_type = grid.meta.type

    # ------------------------------------------------------------------
    # Dtype validation (skip Mask — we handle it below)
    # ------------------------------------------------------------------
    if grid_type != GridType.Mask:
        check_dtype(grid, allowed=_ALLOWED_DTYPES[grid_type])

    # ------------------------------------------------------------------
    # Mask special-case: convert bool → uint8 for GTiff storage
    # ------------------------------------------------------------------
    if grid_type == GridType.Mask:
        write_array = grid.array.astype(np.uint8)
        write_dtype = "uint8"
        write_nodata = 255
    else:
        write_array = grid.array
        write_dtype = grid.array.dtype.name
        write_nodata = grid.meta.nodata

    # ------------------------------------------------------------------
    # Compression predictor: 3 for floating-point, 2 for integer
    # ------------------------------------------------------------------
    if np.issubdtype(grid.array.dtype, np.floating):
        predictor = 3
    else:
        predictor = 2

    rows, cols = grid.shape
    profile: dict = {
        "driver": "GTiff",
        "count": 1,
        "height": rows,
        "width": cols,
        "dtype": write_dtype,
        "transform": grid.meta.transform,
        "crs": grid.meta.crs,
        "nodata": write_nodata,
        "compress": compress,
        "predictor": predictor,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(write_array, 1)
