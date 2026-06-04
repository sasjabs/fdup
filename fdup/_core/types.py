"""Core value objects: GridType, GridMeta, Grid."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Union

import numpy as np
from affine import Affine

try:
    import rasterio.crs as _rcrs
    CRS = _rcrs.CRS
except ImportError:  # pragma: no cover
    CRS = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# GridType
# ---------------------------------------------------------------------------

class GridType(enum.Enum):
    DEM = "DEM"
    FlowDir = "FlowDir"
    FlowAcc = "FlowAcc"
    Mask = "Mask"
    Tree = "Tree"


# ---------------------------------------------------------------------------
# Canonical dtype sets per GridType
# ---------------------------------------------------------------------------

_DEM_DTYPES = (np.int16, np.int32, np.int64, np.float32, np.float64)
_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)
_FLOWDIR_DTYPES = (np.uint8,)
_MASK_DTYPES = (np.bool_,)
_TREE_DTYPES = (np.uint32,)

_ALLOWED_DTYPES: dict[GridType, tuple[type, ...]] = {
    GridType.DEM:     _DEM_DTYPES,
    GridType.FlowDir: _FLOWDIR_DTYPES,
    GridType.FlowAcc: _FLOWACC_DTYPES,
    GridType.Mask:    _MASK_DTYPES,
    GridType.Tree:    _TREE_DTYPES,
}


def _default_nodata(grid_type: GridType, dtype: np.dtype) -> Union[int, float, None]:
    """Return the canonical nodata sentinel for a given type/dtype combination."""
    if grid_type == GridType.FlowDir:
        return 255
    if grid_type == GridType.Mask:
        return None
    if grid_type == GridType.Tree:
        return 0
    # DEM and FlowAcc
    if np.issubdtype(dtype, np.floating):
        return float("nan")
    return int(np.iinfo(dtype).max)


# ---------------------------------------------------------------------------
# GridMeta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridMeta:
    type: GridType
    transform: Affine
    crs: Union[CRS, None]
    nodata: Union[int, float, None]

    @property
    def is_geographic(self) -> bool:
        """True when the CRS is geographic (or unknown/None)."""
        if self.crs is None:
            return True
        return bool(self.crs.is_geographic)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    array: np.ndarray
    meta: GridMeta

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.array.shape
        return (rows, cols)

    @staticmethod
    def create(
        array: np.ndarray,
        type: GridType,  # noqa: A002
        transform: Affine,
        crs: Union[CRS, None] = None,
        nodata: Union[int, float, None] = ...,  # type: ignore[assignment]
    ) -> "Grid":
        """Validated factory.

        Coerces *array* to the canonical dtype for *type* where possible,
        remaps nodata for FlowDir, and fills in a sensible nodata default
        when the caller omits the argument.
        """
        array = np.asarray(array)
        input_dtype = array.dtype

        # --- dtype coercion / validation ---
        if type == GridType.FlowDir:
            array = array.astype(np.uint8)
            # Remap any caller-supplied nodata to the canonical sentinel 255
            if nodata is not ... and nodata is not None and nodata != 255:
                array = np.where(array == nodata, np.uint8(255), array)

        elif type == GridType.Mask:
            array = array.astype(np.bool_)

        elif type == GridType.Tree:
            array = array.astype(np.uint32)

        else:
            # DEM / FlowAcc: keep input dtype if it is in the allowed set
            allowed = _ALLOWED_DTYPES[type]
            if not any(input_dtype == np.dtype(d) for d in allowed):
                raise TypeError(
                    f"GridType.{type.name} does not allow dtype {input_dtype!r}. "
                    f"Allowed: {[np.dtype(d).str for d in allowed]}"
                )

        # --- nodata default ---
        if nodata is ...:
            nodata = _default_nodata(type, array.dtype)

        # FlowDir always has nodata 255 regardless of what the caller passed
        if type == GridType.FlowDir:
            nodata = 255

        # Tree always has nodata 0
        if type == GridType.Tree:
            nodata = 0

        meta = GridMeta(type=type, transform=transform, crs=crs, nodata=nodata)
        return Grid(array=array, meta=meta)
