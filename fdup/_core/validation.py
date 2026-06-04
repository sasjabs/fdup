"""Grid validation helpers shared across all fdup submodules."""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from affine import Affine

from fdup._core.types import Grid, GridType


def check_type(
    grid: Grid,
    expected: Union[GridType, tuple[GridType, ...]],
) -> None:
    """Raise ``ValueError`` when *grid* does not have one of the *expected* types."""
    types = expected if isinstance(expected, tuple) else (expected,)
    if grid.meta.type not in types:
        names = ", ".join(t.name for t in types)
        raise ValueError(
            f"Expected grid type {names}, got {grid.meta.type.name}."
        )


def check_dtype(
    grid: Grid,
    allowed: tuple[np.dtype, ...],
) -> None:
    """Raise ``ValueError`` when *grid.array.dtype* is not in *allowed*."""
    if not any(grid.array.dtype == np.dtype(d) for d in allowed):
        allowed_strs = [np.dtype(d).str for d in allowed]
        raise ValueError(
            f"Grid dtype {grid.array.dtype!r} is not in the allowed set: {allowed_strs}."
        )


def check_crs_match(*grids: Grid) -> None:
    """Raise ``ValueError`` when the grids do not share a common CRS.

    Two ``None`` CRSes are treated as a match.  A mixed ``None``/non-``None``
    pair is an error.  Non-``None`` CRSes are compared with rasterio equality.
    """
    if len(grids) < 2:
        return
    ref = grids[0].meta.crs
    for g in grids[1:]:
        crs = g.meta.crs
        if ref is None and crs is None:
            continue
        if ref is None or crs is None:
            raise ValueError(
                f"CRS mismatch: one grid has CRS={ref!r} and another has CRS={crs!r}."
            )
        if ref != crs:
            raise ValueError(
                f"CRS mismatch between grids: {ref!r} vs {crs!r}."
            )


def check_shape_match(*grids: Grid) -> None:
    """Raise ``ValueError`` when the grids do not share the same (rows, cols) shape."""
    if len(grids) < 2:
        return
    ref_shape = grids[0].shape
    for g in grids[1:]:
        if g.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: {ref_shape} vs {g.shape}."
            )


def check_transform_match(*grids: Grid, atol: float = 1e-9) -> None:
    """Raise ``ValueError`` when the grids do not share the same ``Affine`` transform.

    Comparison is component-wise using ``math.isclose(..., abs_tol=atol)``.
    The six components compared are ``a, b, c, d, e, f``.
    """
    if len(grids) < 2:
        return
    ref: Affine = grids[0].meta.transform
    for g in grids[1:]:
        t: Affine = g.meta.transform
        for name, v1, v2 in (
            ("a", ref.a, t.a),
            ("b", ref.b, t.b),
            ("c", ref.c, t.c),
            ("d", ref.d, t.d),
            ("e", ref.e, t.e),
            ("f", ref.f, t.f),
        ):
            if not math.isclose(v1, v2, abs_tol=atol):
                raise ValueError(
                    f"Transform mismatch at component '{name}': {v1} vs {v2} "
                    f"(atol={atol})."
                )


def check_aligned(
    coarse: Grid,
    fine: Grid,
    *,
    atol: float = 1e-9,
    align_atol: float = 1e-6,
) -> tuple[int, int, int, int]:
    """Verify *coarse* and *fine* are on a shared pixel grid and return scale factors.

    The coarse pixel size must be an integer multiple of the fine pixel size in
    both X and Y.  The coarse grid origin must land exactly on a fine-grid pixel
    boundary.

    Parameters
    ----------
    coarse, fine:
        Grids to compare.  Both must be axis-aligned (``b ≈ 0`` and ``d ≈ 0``).
    atol:
        Absolute tolerance used when testing whether the resolution ratio is
        integral.
    align_atol:
        Absolute tolerance used when testing whether the coarse origin lands on
        a fine-pixel boundary.

    Returns
    -------
    (kx, ky, off_r, off_c) : tuple[int, int, int, int]
        *kx* / *ky* — number of fine pixels per coarse pixel along X (columns)
        and Y (rows).
        *off_r* / *off_c* — fine-pixel row/column index of the top-left corner
        of the coarse grid within the fine grid (non-negative integers).

    Raises
    ------
    ValueError
        On any alignment violation, including negative offsets (coarse grid
        starts outside the fine grid).
    """
    ct = coarse.meta.transform
    ft = fine.meta.transform

    if ft.a == 0.0:
        raise ValueError("Fine transform has zero X pixel spacing.")
    if ft.e == 0.0:
        raise ValueError("Fine transform has zero Y pixel spacing.")

    # --- resolution ratio ---
    kx_raw = ct.a / ft.a
    ky_raw = ct.e / ft.e
    kx = int(round(kx_raw))
    ky = int(round(ky_raw))

    if kx <= 0 or ky <= 0:
        raise ValueError(
            f"Resolution ratio is not a positive integer: kx={kx_raw:.12g}, ky={ky_raw:.12g}."
        )
    if not math.isclose(kx_raw, kx, abs_tol=atol):
        raise ValueError(
            f"Coarse X resolution is not an integer multiple of fine X resolution "
            f"(ratio={kx_raw:.12g}, nearest integer={kx})."
        )
    if not math.isclose(ky_raw, ky, abs_tol=atol):
        raise ValueError(
            f"Coarse Y resolution is not an integer multiple of fine Y resolution "
            f"(ratio={ky_raw:.12g}, nearest integer={ky})."
        )

    # --- origin alignment ---
    # Fine-pixel (col, row) coordinates of the coarse origin.
    inv_f = ~ft
    fc0, fr0 = inv_f * (ct.c, ct.f)

    if not math.isclose(fc0, round(fc0), abs_tol=align_atol):
        raise ValueError(
            f"Coarse origin X does not align to the fine pixel grid "
            f"(fractional fine-column index={fc0:.12g})."
        )
    if not math.isclose(fr0, round(fr0), abs_tol=align_atol):
        raise ValueError(
            f"Coarse origin Y does not align to the fine pixel grid "
            f"(fractional fine-row index={fr0:.12g})."
        )

    off_c = int(round(fc0))
    off_r = int(round(fr0))

    if off_r < 0:
        raise ValueError(
            f"Coarse grid starts above the fine grid (off_r={off_r} < 0)."
        )
    if off_c < 0:
        raise ValueError(
            f"Coarse grid starts left of the fine grid (off_c={off_c} < 0)."
        )

    return kx, ky, off_r, off_c
