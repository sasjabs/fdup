"""Spatially align two same-resolution Grids to a common extent.

Public API
----------
match_grids(reference, other) -> Grid
    Crop/pad *other* so it covers exactly the same extent as *reference*.

Note: Implemented in Phase 3, Step 3.6.
"""

from __future__ import annotations

import numpy as np

from fdup._core.types import Grid, GridType
from fdup._core.validation import check_aligned, check_crs_match


def _nodata_fill(grid_type: GridType, dtype: np.dtype, meta_nodata):
    """Return the appropriate nodata fill value for padding."""
    if grid_type == GridType.FlowDir:
        return np.uint8(255)
    if grid_type == GridType.Mask:
        return False
    if grid_type == GridType.Tree:
        return np.uint32(0)
    # DEM or FlowAcc
    if np.issubdtype(dtype, np.floating):
        return dtype.type("nan")
    # integer DEM: prefer stored nodata, fall back to iinfo.max
    if grid_type == GridType.DEM and meta_nodata is not None:
        return dtype.type(meta_nodata)
    return dtype.type(np.iinfo(dtype).max)


def match_grids(reference: Grid, other: Grid) -> Grid:
    """Align *other* to *reference*'s extent.

    Both grids must share the same CRS, identical pixel size, and their
    origins must be on the same pixel grid (i.e. ``check_aligned`` with
    ``kx == ky == 1``).  Cells in the output that fall outside *other*'s
    extent are filled with the type-appropriate nodata sentinel.

    Parameters
    ----------
    reference :
        Target extent.
    other :
        Grid to crop / pad.

    Returns
    -------
    Grid
        Same type as *other*; shape and transform identical to *reference*.
    """
    check_crs_match(reference, other)

    ref_t = reference.meta.transform
    oth_t = other.meta.transform

    if abs(ref_t.a - oth_t.a) >= 1e-9 or abs(ref_t.e - oth_t.e) >= 1e-9:
        raise ValueError(
            f"Pixel size mismatch: reference ({ref_t.a}, {ref_t.e}) vs "
            f"other ({oth_t.a}, {oth_t.e})."
        )

    kx, ky, off_r, off_c = check_aligned(other, reference)
    if kx != 1 or ky != 1:
        raise ValueError(
            f"match_grids requires equal-resolution grids (kx={kx}, ky={ky}). "
            "Use disaggregate_mask first if resolutions differ."
        )

    ref_nrows, ref_ncols = reference.shape
    oth_nrows, oth_ncols = other.shape

    fill = _nodata_fill(other.meta.type, other.array.dtype, other.meta.nodata)
    out = np.full((ref_nrows, ref_ncols), fill, dtype=other.array.dtype)

    # Row/col range of 'other' that overlaps 'reference'
    # 'other' starts at row off_r, col off_c within 'reference's coordinate frame.
    # But check_aligned(other, reference) gives the offset of 'other's origin
    # within 'reference's pixel grid — meaning off_r/off_c are where 'other'
    # starts relative to 'reference'.
    # We need to figure out the overlap region.

    # In reference coords: other occupies rows [off_r, off_r+oth_nrows)
    #                                   cols [off_c, off_c+oth_ncols)
    ref_r0 = off_r
    ref_r1 = off_r + oth_nrows
    ref_c0 = off_c
    ref_c1 = off_c + oth_ncols

    # Clip to reference extent
    clip_r0 = max(0, ref_r0)
    clip_r1 = min(ref_nrows, ref_r1)
    clip_c0 = max(0, ref_c0)
    clip_c1 = min(ref_ncols, ref_c1)

    if clip_r0 < clip_r1 and clip_c0 < clip_c1:
        # Corresponding region in 'other'
        oth_r0 = clip_r0 - ref_r0
        oth_r1 = clip_r1 - ref_r0
        oth_c0 = clip_c0 - ref_c0
        oth_c1 = clip_c1 - ref_c0
        out[clip_r0:clip_r1, clip_c0:clip_c1] = other.array[oth_r0:oth_r1, oth_c0:oth_c1]

    return Grid.create(
        array=out,
        type=other.meta.type,
        transform=reference.meta.transform,
        crs=reference.meta.crs,
        nodata=other.meta.nodata,
    )
