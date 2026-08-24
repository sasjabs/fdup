"""Boundary-safe array window extraction for upscaling kernels."""

from __future__ import annotations

import numpy as np

from fdup._core.validation import normalize_k


def cell_slice(
    array: np.ndarray,
    i: int,
    j: int,
    k: int | tuple[int, int],
    fill,
) -> np.ndarray:
    """Extract a ``(ky, kx)`` window from *array* at ``[i:i+ky, j:j+kx]``.

    When the requested window lies entirely inside the array the returned view
    shares memory with *array* (no copy).  When any edge of the window falls
    outside the array boundary (including negative *i* / *j*) a fresh
    ``(ky, kx)`` buffer is allocated, filled with *fill*, and the in-bounds
    portion is copied into it.

    Parameters
    ----------
    array :
        Source 2-D array.
    i, j :
        Top-left (row, column) index of the window.  May be negative.
    k :
        Window size.  A positive integer (square; equivalent to ``(k, k)``)
        or a length-2 ``(kx, ky)`` tuple of positive integers.  *kx* is the
        column extent; *ky* is the row extent.
    fill :
        Scalar value written to out-of-bounds cells; cast to ``array.dtype``.

    Returns
    -------
    np.ndarray, shape (ky, kx), dtype == array.dtype
        The extracted window.
    """
    kx, ky = normalize_k(k)
    nrows, ncols = array.shape

    # Fast path: window is fully inside the array.
    if i >= 0 and i + ky <= nrows and j >= 0 and j + kx <= ncols:
        return array[i : i + ky, j : j + kx]

    # Slow path: allocate a fill buffer and copy the in-bounds portion.
    out = np.full((ky, kx), fill, dtype=array.dtype)

    # Clamp to valid source bounds.
    r0 = max(0, i)
    r1 = min(nrows, i + ky)
    c0 = max(0, j)
    c1 = min(ncols, j + kx)

    if r0 < r1 and c0 < c1:
        # Destination slice inside the (ky, kx) buffer.
        dr0 = r0 - i
        dr1 = dr0 + (r1 - r0)
        dc0 = c0 - j
        dc1 = dc0 + (c1 - c0)
        out[dr0:dr1, dc0:dc1] = array[r0:r1, c0:c1]

    return out
