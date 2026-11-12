"""Read a single-band raster file into a :class:`~fdup._core.types.Grid`."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from affine import Affine

from fdup._core.types import Grid, GridType


def _to_uint8_remap_nodata(
    array: np.ndarray,
    file_nodata: Union[int, float, None],
    sentinel: int,
) -> np.ndarray:
    """Cast *array* to uint8 and replace *file_nodata* with *sentinel*.

    The buffer from ``src.read`` is owned. When it is already uint8, nodata
    is rewritten in place. A wider dtype is masked first: the cast can wrap
    the file nodata onto a valid uint8 value.
    """
    sentinel_u8 = np.uint8(sentinel)
    if array.dtype == np.uint8:
        if file_nodata is not None and file_nodata != sentinel:
            array[array == file_nodata] = sentinel_u8
        return array

    nodata_mask = (array == file_nodata) if file_nodata is not None else None
    array = array.astype(np.uint8, copy=False)
    if nodata_mask is not None:
        array[nodata_mask] = sentinel_u8
    return array


def read(
    path: Union[str, Path],
    grid_type: GridType = GridType.FlowDir,
) -> Grid:
    """Read a single-band GeoTIFF (or any rasterio-supported format) into a Grid.

    Parameters
    ----------
    path:
        Path to the raster file.
    grid_type:
        Semantic type to assign to the returned Grid.  Controls per-type
        dtype normalisation and nodata conventions.

    Returns
    -------
    Grid
        Grid whose array dtype and nodata sentinel follow the canonical
        contract defined in ``fdup._core.types``.

    Raises
    ------
    ValueError
        If the file has more than one band.
    TypeError
        If the on-disk dtype is incompatible with *grid_type*.
    """
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                f"Expected a single-band raster, got {src.count} bands: {path}"
            )
        array: np.ndarray = src.read(1)
        transform: Affine = src.transform
        crs = src.crs  # may be None
        file_nodata = src.nodata  # may be None

    # ------------------------------------------------------------------
    # Per-type normalisation
    # ------------------------------------------------------------------

    if grid_type == GridType.FlowDir:
        array = _to_uint8_remap_nodata(array, file_nodata, sentinel=255)
        nodata: Union[int, float, None] = 255

    elif grid_type == GridType.Mask:
        # (array != 0) is already bool; a further astype(bool) would copy it.
        nodata_mask = (array == file_nodata) if file_nodata is not None else None
        array = array != 0
        if nodata_mask is not None:
            array[nodata_mask] = False
        nodata = None

    elif grid_type == GridType.FlowAcc:
        _FLOWACC_DTYPES = (
            np.dtype(np.int32),
            np.dtype(np.uint32),
            np.dtype(np.int64),
            np.dtype(np.uint64),
            np.dtype(np.float32),
            np.dtype(np.float64),
        )
        if not any(array.dtype == d for d in _FLOWACC_DTYPES):
            raise TypeError(
                f"FlowAcc raster has unsupported dtype {array.dtype!r}. "
                f"Allowed: {[d.str for d in _FLOWACC_DTYPES]}"
            )
        if file_nodata is None:
            if np.issubdtype(array.dtype, np.floating):
                nodata = float("nan")
            else:
                nodata = int(np.iinfo(array.dtype).max)
        else:
            nodata = file_nodata

    elif grid_type == GridType.DEM:
        _DEM_DTYPES = (
            np.dtype(np.int16),
            np.dtype(np.int32),
            np.dtype(np.int64),
            np.dtype(np.float32),
            np.dtype(np.float64),
        )
        if not any(array.dtype == d for d in _DEM_DTYPES):
            raise TypeError(
                f"DEM raster has unsupported dtype {array.dtype!r}. "
                f"Allowed: {[d.str for d in _DEM_DTYPES]}"
            )
        if file_nodata is None:
            if np.issubdtype(array.dtype, np.floating):
                nodata = float("nan")
            else:
                nodata = None
        else:
            nodata = file_nodata

    elif grid_type == GridType.Tree:
        array = array.astype(np.uint32, copy=False)
        nodata = 0

    elif grid_type == GridType.Strahler:
        array = _to_uint8_remap_nodata(array, file_nodata, sentinel=0)
        nodata = 0

    else:
        raise ValueError(f"Unsupported grid_type: {grid_type!r}")

    return Grid.create(
        array=array,
        type=grid_type,
        transform=transform,
        crs=crs,
        nodata=nodata,
    )
