"""Flow-direction windrose and Earth Mover's Distance.

Public API
----------
flowdir_windrose(flowdir, *, dist=True) -> tuple[np.ndarray, np.ndarray]
    Compute an 8-bin windrose from a D8 flow-direction grid.
    Returns ``(absolute_sums, distribution)`` in ``COMPASS_ORDER`` bin order
    (E, SE, S, SW, W, NW, N, NE).

    When ``dist=True`` (default) each bin accumulates the sum of distances
    (metres) to the receiving cell centre.  When ``dist=False`` each bin
    accumulates the raw flow-direction cell count.

    Geographic CRS: distances via Vincenty formula (row-varying).
    Projected CRS: distances via Euclidean formula (constant across rows).

windrose_emd(windrose1, windrose2) -> float
    Circular Wasserstein-1 distance between two normalised 8-bin windrose
    distributions.  Both inputs must already sum to 1 (±1e-6).

_warmup(dtype)
    Pre-compile ``_nb_windrose`` for the warmup registry.  The *dtype*
    argument is accepted for uniformity; the kernel always operates on
    uint8 / float64 and does not specialise per FlowAcc dtype.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from fdup._core.d8 import COMPASS_ORDER
from fdup._core.geodesy import row_distance_table
from fdup._core.types import Grid, GridType
from fdup._core.validation import check_type


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _nb_windrose(
    dir_arr: np.ndarray,
    dist_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate per-direction distance sums and counts in parallel.

    Parameters
    ----------
    dir_arr :
        uint8 2-D ESRI D8 flow-direction array.  Nodata (255) and sink (0)
        cells are skipped.
    dist_table :
        float64 ``(nrows, 8)`` array from ``row_distance_table``; entry
        ``[i, d]`` is the distance in metres from a row-*i* cell centre to
        its D8 neighbour in compass direction *d*.

    Returns
    -------
    row_dists : float64 (nrows, 8)
        Accumulated distance per row and direction.
    row_counts : float64 (nrows, 8)
        Accumulated cell count per row and direction.
    """
    nrow = dir_arr.shape[0]
    ncol = dir_arr.shape[1]

    d8_to_idx = np.zeros(256, dtype=np.uint8)
    d8_to_idx[1]   = np.uint8(0)   # E
    d8_to_idx[2]   = np.uint8(1)   # SE
    d8_to_idx[4]   = np.uint8(2)   # S
    d8_to_idx[8]   = np.uint8(3)   # SW
    d8_to_idx[16]  = np.uint8(4)   # W
    d8_to_idx[32]  = np.uint8(5)   # NW
    d8_to_idx[64]  = np.uint8(6)   # N
    d8_to_idx[128] = np.uint8(7)   # NE

    row_dists  = np.zeros((nrow, 8), dtype=np.float64)
    row_counts = np.zeros((nrow, 8), dtype=np.float64)

    for i in prange(nrow):
        for j in range(ncol):
            fd = np.uint8(dir_arr[i, j])
            if fd == np.uint8(0) or fd == np.uint8(255):
                continue
            d = np.int64(d8_to_idx[fd])
            row_dists[i, d]  += dist_table[i, d]
            row_counts[i, d] += 1.0

    return row_dists, row_counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flowdir_windrose(
    flowdir: Grid,
    *,
    dist: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an 8-bin flow-direction windrose.

    Bins are ordered per ``fdup._core.d8.COMPASS_ORDER``:
    E, SE, S, SW, W, NW, N, NE.

    Parameters
    ----------
    flowdir :
        ``GridType.FlowDir`` grid.
    dist :
        If ``True`` (default) each bin value is the sum of distances
        (metres) from all cells pointing in that direction to their
        receiving cell centre.  If ``False`` each bin value is the raw
        count of cells pointing in that direction.

    Returns
    -------
    absolute_sums : np.ndarray, shape (8,), dtype float64
        Per-direction totals (metres when ``dist=True``, counts when
        ``dist=False``).
    distribution : np.ndarray, shape (8,), dtype float64
        Normalised windrose (``absolute_sums / absolute_sums.sum()``).
        All-zeros array when the grid contains no valid flow-direction
        cells.

    Raises
    ------
    TypeError
        When *flowdir* is not a ``GridType.FlowDir`` grid.
    """
    check_type(flowdir, GridType.FlowDir)

    nrows = flowdir.shape[0]
    transform = flowdir.meta.transform
    geographic = flowdir.meta.is_geographic

    dist_table = row_distance_table(transform, nrows, geographic=geographic)
    row_dists, row_counts = _nb_windrose(flowdir.array, dist_table)

    if dist:
        absolute_sums = np.sum(row_dists, axis=0)
    else:
        absolute_sums = np.sum(row_counts, axis=0)

    total = float(absolute_sums.sum())
    if total == 0.0:
        distribution = np.zeros(8, dtype=np.float64)
    else:
        distribution = absolute_sums / total

    return absolute_sums, distribution


def windrose_emd(
    windrose1: np.ndarray,
    windrose2: np.ndarray,
) -> float:
    """Circular Wasserstein-1 (Earth Mover's) distance between two windroses.

    Uses the closed-form analytical solution for EMD on an 8-node cycle
    graph: if ``a_i = p_i - q_i`` and ``S_i`` are the cumulative sums of
    ``a_i``, then ``EMD = min_c sum_i |S_i - c|`` where *c* is any median
    of the ``S_i`` values.  The result is normalised to ``[0, 1]`` by
    dividing by the maximum possible EMD for 8 equal-spacing bins (= 4).

    Both inputs must be normalised distributions that sum to 1 (±1e-6).
    The bin order must match ``fdup._core.d8.COMPASS_ORDER``
    (E, SE, S, SW, W, NW, N, NE).

    Parameters
    ----------
    windrose1, windrose2 :
        Float arrays of exactly 8 elements summing to 1.  Typically the
        *distribution* array returned by :func:`flowdir_windrose`.

    Returns
    -------
    float
        Normalised circular EMD in ``[0, 1]``.

    Raises
    ------
    ValueError
        When either array does not have shape ``(8,)`` or does not sum to
        1 (±1e-6).
    """
    w1 = np.asarray(windrose1, dtype=np.float64)
    w2 = np.asarray(windrose2, dtype=np.float64)

    if w1.shape != (8,) or w2.shape != (8,):
        raise ValueError(
            f"Both windrose arrays must have shape (8,) matching "
            f"fdup._core.d8.COMPASS_ORDER = {COMPASS_ORDER}. "
            f"Got shapes {w1.shape} and {w2.shape}."
        )

    for name, w in (("windrose1", w1), ("windrose2", w2)):
        total = float(w.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{name} must be a normalised distribution summing to 1.0 "
                f"(±1e-6), but got sum={total:.8f}.  "
                f"Pass the *distribution* output of flowdir_windrose, or "
                f"normalise manually via arr / arr.sum()."
            )

    diff = w1 - w2
    cumulative = np.cumsum(diff)
    med = np.median(cumulative)
    emd = float(np.sum(np.abs(cumulative - med)))

    # Maximum EMD for 8 equally-spaced circular bins is n/2 = 4.
    normalized = emd / 4.0
    return float(np.clip(normalized, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup(dtype: object) -> None:  # noqa: ARG001
    """Pre-compile ``_nb_windrose``.

    The *dtype* argument is accepted for warmup-registry uniformity.
    FlowDir is always uint8, so the kernel does not specialise per FlowAcc
    dtype.
    """
    dummy_dir = np.zeros((4, 4), dtype=np.uint8)
    dummy_dir[1, 1] = np.uint8(4)   # S: (1,1) → (2,1)
    dummy_dir[2, 2] = np.uint8(1)   # E: (2,2) → (2,3)
    dummy_table = np.ones((4, 8), dtype=np.float64)
    _nb_windrose(dummy_dir, dummy_table)
