"""Boundary-safe array window extraction for upscaling kernels."""

from __future__ import annotations

import numpy as np


def cell_slice(
    array: np.ndarray,
    i: int,
    j: int,
    k: int,
    fill,
) -> np.ndarray:
    """Extract a ``(k, k)`` window from *array* centred at ``[i:i+k, j:j+k]``.

    When the requested window lies entirely inside the array the returned view
    shares memory with *array* (no copy).  When any edge of the window falls
    outside the array boundary (including negative *i* / *j*) a fresh
    ``(k, k)`` buffer is allocated, filled with *fill*, and the in-bounds
    portion is copied into it.

    Parameters
    ----------
    array :
        Source 2-D array.
    i, j :
        Top-left (row, column) index of the window.  May be negative.
    k :
        Side length of the square window.
    fill :
        Scalar value written to out-of-bounds cells; cast to ``array.dtype``.

    Returns
    -------
    np.ndarray, shape (k, k), dtype == array.dtype
        The extracted window.
    """
    nrows, ncols = array.shape

    # Fast path: window is fully inside the array.
    if i >= 0 and i + k <= nrows and j >= 0 and j + k <= ncols:
        return array[i : i + k, j : j + k]

    # Slow path: allocate a fill buffer and copy the in-bounds portion.
    out = np.full((k, k), fill, dtype=array.dtype)

    # Clamp to valid source bounds.
    r0 = max(0, i)
    r1 = min(nrows, i + k)
    c0 = max(0, j)
    c1 = min(ncols, j + k)

    if r0 < r1 and c0 < c1:
        # Destination slice inside the (k, k) buffer.
        dr0 = r0 - i
        dr1 = dr0 + (r1 - r0)
        dc0 = c0 - j
        dc1 = dc0 + (c1 - c0)
        out[dr0:dr1, dc0:dc1] = array[r0:r1, c0:c1]

    return out
