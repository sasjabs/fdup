"""Binned, flow-accumulation-weighted aggregation of per-seed accuracy scores.

Public API
----------
accur_bins(accurs, seeds, cumulative=True, bins=None)
        -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]
    Aggregate per-seed flow direction accuracy scores (as returned by
    :func:`fdup.evals.compare_flowdir`) into log-spaced bins, weighting each
    seed by its mouth flow accumulation.  Also returns the bin cutoffs as
    absolute element counts and as fractions of the array length.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_bins(n: int) -> np.ndarray:
    """Log-spaced bin cutoffs (element counts) for an array of length *n*.

    Cutoffs are ``floor(n * [1e(k), 2e(k), ..., 9e(k), 1e(k+1), ..., 9e-1, 1])``
    where *k* is the integer satisfying ``1 <= floor(n * 1e(k)) <= 9``.
    Integer arithmetic is used throughout, so there is no float rounding at
    decade boundaries.  The result is strictly increasing and ends at *n*.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    depth = len(str(n)) - 1   # == -k
    cutoffs = [
        (n * m) // 10 ** e
        for e in range(depth, 0, -1)
        for m in range(1, 10)
    ]
    cutoffs.append(n)
    return np.asarray(cutoffs, dtype=np.int64)


def _check_bins(bins, n: int) -> np.ndarray:
    """Validate user-supplied bin cutoffs and return them as int64."""
    arr = np.asarray(bins)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"bins must be a non-empty 1-D array of cutoffs, got shape {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"bins must be numeric, got dtype {arr.dtype!r}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise ValueError("bins must contain integer element counts.")
    out = arr.astype(np.int64)
    if out[0] < 1 or out[-1] > n:
        raise ValueError(
            f"bins must lie within [1, {n}] (number of seeds), "
            f"got [{out[0]}, {out[-1]}]."
        )
    if np.any(np.diff(out) <= 0):
        raise ValueError("bins must be strictly increasing.")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def accur_bins(
    accurs: np.ndarray,
    seeds: np.ndarray,
    cumulative: bool = True,
    bins: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Aggregate per-seed accuracy scores into flow-accumulation-weighted bins.

    Seeds are ranked by ``seeds["acc"]`` in descending order (a stable sort,
    so the order of :func:`fdup.utils.river_tree` output is preserved).  Bin
    cutoffs are element counts along that ranking; by default they are
    log-spaced::

        floor(N * [1e(k), 2e(k), ..., 9e(k), 1e(k+1), ..., 9e(k+1), 1e-1, ..., 9e-1, 1])

    where ``N = len(accurs)`` and *k* is the integer satisfying
    ``1 <= floor(N * 1e(k)) <= 9`` (e.g. ``N = 1234567`` → ``k = -6``).

    Each bin value is the weighted average ``sum(acc * accur) / sum(acc)``
    over the seeds it covers.

    Parameters
    ----------
    accurs :
        1-D array of per-seed accuracy scores, same length as *seeds*, as
        returned by :func:`fdup.evals.compare_flowdir`.  Negative values
        (the ``-1`` degenerate-seed marker) and NaN are excluded from
        aggregation.
    seeds :
        Structured array as returned by :func:`fdup.utils.river_tree` or
        :func:`fdup.utils.mask_seeds`.  The ``acc`` field is used both for
        ranking and as weights.
    cumulative :
        When ``True`` (default), bin *i* aggregates seeds ``[0, bins[i])``.
        When ``False``, bin *i* aggregates seeds ``[bins[i-1], bins[i])``
        (``[0, bins[0])`` for the first bin).
    bins :
        Optional custom cutoffs: a strictly increasing 1-D array of integer
        element counts in ``[1, N]``.  When ``None`` (default) the log-spaced
        cutoffs above are used.

    Returns
    -------
    aggregated : np.ndarray float64, shape (nbins,)
        Aggregated accuracy per bin.  NaN for bins containing no valid seed
        (or only zero-weight seeds).
    (bins_absolute, bins_relative) : tuple[np.ndarray, np.ndarray]
        Bin cutoffs as int64 element counts and as float64 fractions of
        ``N`` (``bins_absolute / N``; the default bins end at 1.0).

    All arrays are empty when *seeds* is empty and *bins* is ``None``.

    Raises
    ------
    ValueError
        When *accurs* is not 1-D, its length differs from *seeds*, *seeds*
        has no ``acc`` field, or *bins* is malformed.
    """
    # ---- validation -------------------------------------------------------
    accurs = np.asarray(accurs)
    if accurs.ndim != 1:
        raise ValueError(f"accurs must be a 1-D array, got shape {accurs.shape}.")
    if seeds.dtype.names is None or "acc" not in seeds.dtype.names:
        raise ValueError(
            "seeds must be a structured array with an 'acc' field "
            "(as returned by fdup.utils.river_tree())."
        )
    n = len(accurs)
    if len(seeds) != n:
        raise ValueError(
            f"accurs must have the same length as seeds ({len(seeds)}), got {n}."
        )

    cutoffs = _log_bins(n) if bins is None else _check_bins(bins, n)
    if cutoffs.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, (cutoffs, empty.copy())

    # ---- rank by accumulation and mask invalid scores ---------------------
    weights = seeds["acc"].astype(np.float64)
    order   = np.argsort(-weights, kind="stable")
    weights = weights[order]
    values  = accurs[order].astype(np.float64)

    valid = np.isfinite(values) & (values >= 0.0) & np.isfinite(weights)
    weights = np.where(valid, weights, 0.0)
    values  = np.where(valid, values, 0.0)

    # ---- prefix sums: index c → sum over seeds [0, c) ---------------------
    cum_w  = np.concatenate(([0.0], np.cumsum(weights)))
    cum_wv = np.concatenate(([0.0], np.cumsum(weights * values)))

    if cumulative:
        num = cum_wv[cutoffs]
        den = cum_w[cutoffs]
    else:
        lo  = np.concatenate(([0], cutoffs[:-1]))
        num = cum_wv[cutoffs] - cum_wv[lo]
        den = cum_w[cutoffs] - cum_w[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0.0] = np.nan
    return out, (cutoffs, cutoffs / np.float64(n))
