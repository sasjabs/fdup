"""Watershed comparison via the squared Ochiai overlap index.

Public API
----------
compare_watersheds(mask1, mask2) -> (float, Grid)
    Returns the **squared** Ochiai coefficient and the intersection mask.

    The squared form is::

        ochiai² = area_inter² / (area1 × area2)

    which equals the plain Ochiai coefficient squared.  It is equivalent to
    the Szymkiewicz–Simpson overlap index when one watershed fully contains
    the other, and is bounded to [0, 1].  The raw reference script uses this
    formulation.

_warmup(dtype)
    No-op; exposed for warmup-registry uniformity (no numba kernel here).
"""

from __future__ import annotations

import math

import numpy as np

from fdup._core.geodesy import get_cell_areas
from fdup._core.types import Grid, GridType
from fdup._core.validation import (
    check_crs_match,
    check_shape_match,
    check_transform_match,
    check_type,
)

_PREP_HINT = (
    "Hint: call utils.disaggregate_mask() on the coarse mask, then "
    "utils.match_grids() to align it with the fine mask before comparing."
)


def compare_watersheds(mask1: Grid, mask2: Grid) -> tuple[float, Grid]:
    """Compare two watershed masks and return the squared Ochiai overlap index.

    The two masks must be already aligned (same shape, same transform, same
    CRS).  The most common reason they are *not* aligned is that one mask
    comes from a coarse-resolution flow direction grid; in that case, upscale
    it first with :func:`fdup.utils.disaggregate_mask` and then align with
    :func:`fdup.utils.match_grids`.

    Parameters
    ----------
    mask1, mask2 :
        Watershed masks (``GridType.Mask``).  Both must share the same CRS,
        shape, and affine transform.

    Returns
    -------
    ochiai : float
        Squared Ochiai coefficient in [0, 1].  A value of 1 indicates perfect
        spatial overlap; 0 indicates no overlap.  If either mask has zero area
        the returned score is 0.0.
    intersection : Grid
        Boolean ``GridType.Mask`` grid marking pixels that belong to both
        watersheds, sharing the transform and CRS of *mask1*.

    Raises
    ------
    ValueError
        When the grids have incompatible types, CRS, shapes, or transforms.
    """
    check_type(mask1, GridType.Mask)
    check_type(mask2, GridType.Mask)
    check_crs_match(mask1, mask2)

    try:
        check_shape_match(mask1, mask2)
    except ValueError as exc:
        raise ValueError(f"{exc}\n{_PREP_HINT}") from exc

    try:
        check_transform_match(mask1, mask2)
    except ValueError as exc:
        raise ValueError(f"{exc}\n{_PREP_HINT}") from exc

    inter_arr: np.ndarray = mask1.array & mask2.array

    cell_areas = get_cell_areas(
        mask1.meta.transform,
        mask1.shape[0],
        geographic=mask1.meta.is_geographic,
    )

    area1 = float(np.sum(mask1.array * cell_areas[:, np.newaxis]))
    area2 = float(np.sum(mask2.array * cell_areas[:, np.newaxis]))
    area_inter = float(np.sum(inter_arr * cell_areas[:, np.newaxis]))

    denom = area1 * area2
    ochiai = 0.0 if denom == 0.0 or not math.isfinite(denom) else (area_inter ** 2) / denom

    intersection = Grid.create(
        array=inter_arr,
        type=GridType.Mask,
        transform=mask1.meta.transform,
        crs=mask1.meta.crs,
    )
    return float(ochiai), intersection


def _warmup(dtype: object) -> None:  # noqa: ARG001
    """No-op warmup stub (no numba kernels in this module)."""
