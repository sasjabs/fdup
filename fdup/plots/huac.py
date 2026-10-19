"""Log-log scatter of a hierarchical upstream-area comparison (HUAC).

Public API
----------
huac_scatterplot(huac_table, *, color, point_size, stats, size, ...)
        -> tuple[Figure, Axes]
    Scatter coarse upstream area against fine max upstream area on shared
    base-10 log axes.  Points may be coloured by relative error.  An
    optional OLS fit is drawn in log10 space.  Returns ``(fig, ax)`` and
    does not call ``plt.show()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure

from fdup.plots._style import RC, _get_cmap, add_titles, save

__all__ = ["huac_scatterplot"]

_REQUIRED_COLUMNS = (
    "coarse_upstream_area",
    "fine_max_upstream_area",
    "relative_upstream_area_error",
)

_X_COLUMN = "fine_max_upstream_area"
_Y_COLUMN = "coarse_upstream_area"
_ERROR_COLUMN = "relative_upstream_area_error"

_DEFAULT_FIGSIZE = (5.0, 5.0)
_NORM_EPS = 1e-12
_LOG_LIMIT_EPS = 1e-12
_COLORBAR_ASPECT = 20.0
_COLORBAR_PAD_FRAC = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_columns(huac_table: pd.DataFrame) -> None:
    try:
        columns = huac_table.columns
    except AttributeError as exc:
        raise TypeError(
            "huac_table must be a pandas DataFrame, "
            f"got {type(huac_table).__name__}."
        ) from exc
    missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            "huac_table is missing required columns: " + ", ".join(missing) + "."
        )


def _numeric_column(huac_table: pd.DataFrame, name: str) -> np.ndarray:
    values = huac_table[name]
    if isinstance(values, pd.DataFrame):
        raise ValueError(f"Column {name!r} is duplicated in huac_table.")
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=np.float64)


def _filter_points(
    huac_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep rows with positive finite areas and a finite relative error."""
    x = _numeric_column(huac_table, _X_COLUMN)
    y = _numeric_column(huac_table, _Y_COLUMN)
    err = _numeric_column(huac_table, _ERROR_COLUMN)
    keep = (
        np.isfinite(x)
        & (x > 0.0)
        & np.isfinite(y)
        & (y > 0.0)
        & np.isfinite(err)
    )
    if not np.any(keep):
        raise ValueError(
            "No rows left to plot after dropping non-positive or non-finite "
            "upstream areas and non-finite relative errors."
        )
    return x[keep], y[keep], err[keep]


def _figsize(size: float | Sequence[float] | None) -> tuple[float, float]:
    if size is None:
        return _DEFAULT_FIGSIZE
    if isinstance(size, bool):
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        )
    if isinstance(size, Real):
        side = float(size)
        if side <= 0.0:
            raise ValueError(f"size must be positive, got {size!r}.")
        return (side, side)
    if isinstance(size, (str, bytes)):
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        )
    try:
        seq = tuple(size)
    except TypeError as exc:
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        ) from exc
    if len(seq) != 2:
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        )
    try:
        width, height = float(seq[0]), float(seq[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        ) from exc
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"size dimensions must be positive, got {size!r}.")
    return (width, height)


def _shared_log_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Decade bounds covering every positive finite value of *x* and *y*."""
    both = np.concatenate([x, y])
    positive = both[np.isfinite(both) & (both > 0.0)]
    if positive.size == 0:
        raise ValueError("No positive finite upstream areas to plot.")
    log_vals = np.log10(positive)
    lo = 10.0 ** np.floor(np.min(log_vals) + _LOG_LIMIT_EPS)
    hi = 10.0 ** np.ceil(np.max(log_vals) - _LOG_LIMIT_EPS)
    if not hi > lo:
        lo = lo / 10.0
        hi = hi * 10.0
    return float(lo), float(hi)


def _error_norm_limits(err: np.ndarray) -> tuple[float, float]:
    """Limits with ``vmin < 0 < vmax`` for a diverging norm centered at 0."""
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return -_NORM_EPS, _NORM_EPS
    err_min = float(np.min(finite))
    err_max = float(np.max(finite))
    span = max(float(np.max(np.abs(finite))), _NORM_EPS)
    if err_min >= 0.0:
        vmin = -span
        vmax = max(err_max, _NORM_EPS)
    elif err_max <= 0.0:
        vmin = min(err_min, -_NORM_EPS)
        vmax = span
    else:
        vmin = err_min
        vmax = err_max
    if not vmin < 0.0 < vmax:
        vmin = min(vmin, -_NORM_EPS)
        vmax = max(vmax, _NORM_EPS)
    return vmin, vmax


def _resolve_point_color(color: str) -> tuple[Colormap | None, tuple | None]:
    """Return ``(cmap, None)`` or ``(None, rgba)``.

    A matplotlib colormap name wins over a colour of the same spelling.
    """
    if not isinstance(color, str):
        raise TypeError(f"color must be a string, got {type(color).__name__}.")
    try:
        return _get_cmap(color), None
    except ValueError as exc:
        if "ambiguous" in str(exc):
            raise
        try:
            rgba = to_rgba(color)
        except ValueError:
            raise ValueError(
                f"{color!r} is neither a matplotlib colormap name nor a valid color."
            ) from exc
        return None, rgba


def _log10_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """OLS of ``log10(y)`` on ``log10(x)``.  Returns slope, intercept, R², RMSE."""
    if x.size < 2:
        raise ValueError(
            f"stats=True requires at least 2 points, got {x.size} after filtering."
        )
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept = (float(v) for v in np.polyfit(log_x, log_y, 1))
    yhat = slope * log_x + intercept
    resid = log_y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return slope, intercept, r2, rmse


def _add_fit_line(
    ax: Axes,
    slope: float,
    intercept: float,
    lo: float,
    hi: float,
) -> None:
    x_line = np.array([lo, hi], dtype=np.float64)
    y_line = 10.0 ** (slope * np.log10(x_line) + intercept)
    ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=0.6, zorder=3)


def _add_error_colorbar(
    fig: Figure,
    ax: Axes,
    mappable,
    label: str | None,
) -> None:
    """Place a vertical colorbar to the right of the square axes box.

    ``fig.colorbar(..., ax=ax)`` reads the axes position from before
    ``set_box_aspect``, so the bar is attached to the adjusted position
    and matches the square box.
    """
    ax.apply_aspect()
    pos = ax.get_position()
    fig_w, fig_h = (float(v) for v in fig.get_size_inches())
    width = float(pos.height) * fig_h / (_COLORBAR_ASPECT * fig_w)
    pad = _COLORBAR_PAD_FRAC * float(pos.width)
    x0 = float(pos.x1) + pad
    if x0 + width > 0.99:
        x0 = max(float(pos.x1) + 0.01, 0.99 - width)
    cax = fig.add_axes([x0, float(pos.y0), width, float(pos.height)])
    cbar = fig.colorbar(mappable, cax=cax)
    if label is not None:
        cbar.set_label(label)


def _add_stats_box(ax: Axes, r2: float, rmse: float) -> None:
    ax.text(
        0.04,
        0.96,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=4,
        bbox={
            "boxstyle": "square,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.6,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def huac_scatterplot(
    huac_table: pd.DataFrame,
    *,
    color: str = "jet",
    point_size: float = 0.25,
    stats: bool = True,
    size: float | Sequence[float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output: str | Path | None = "huac_scatterplot.png",
    dpi: float = 300,
) -> tuple[Figure, Axes]:
    """Scatter upscaled upstream area against fine reference accumulation.

    X is ``fine_max_upstream_area`` and Y is ``coarse_upstream_area``.
    Both axes are base-10 logarithmic and share one range, the smallest
    decade bounds that cover every positive finite value of either column.
    A range that collapses to a single power of ten is widened by one
    decade on each side.  The axes box is square in display space
    (``set_box_aspect(1)``).

    Parameters
    ----------
    huac_table :
        HUAC cell table.  Required columns are ``coarse_upstream_area``,
        ``fine_max_upstream_area``, and ``relative_upstream_area_error``.
        Rows with a non-positive or non-finite area, or a non-finite
        relative error, are dropped.
    color :
        Matplotlib colormap name (matched case-insensitively), used to
        colour points by ``relative_upstream_area_error`` with a
        diverging norm centered at 0, or one colour specification
        (HEXA strings work).  Default ``'jet'``.
    point_size :
        Scatter marker area passed as ``s``.  Default ``0.25``.
    stats :
        When ``True`` (default), draw an OLS line fitted to
        ``log10(x)`` and ``log10(y)`` and annotate R² and RMSE of that
        fit, both in log10 units.  At least two points are required.
    size :
        Figure size in inches.  ``None`` uses ``(5, 5)``.  A number is
        both width and height.  A ``(width, height)`` pair is ``figsize``.
    title, subtitle :
        Optional texts centered above the axes.
    legend :
        When ``True`` and *color* is a colormap, draw a vertical colorbar
        to the right of the axes.  A single colour never draws a colorbar.
    legend_title :
        Colorbar label.  ``None`` omits the label.
    xlabel, ylabel :
        Axis labels.  ``None`` uses ``'Reference upstream area, km$^2$'``
        and ``'Upscaled upstream area, km$^2$'``.
    output :
        Path to write a PNG.  Default ``'huac_scatterplot.png'``.
        ``None`` skips saving.
    dpi :
        Resolution used when *output* is not ``None``.  Default ``300``.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes.  ``plt.show()`` is not called and the
        figure is not closed.

    Raises
    ------
    ValueError
        When a required column is missing, no rows remain after filtering,
        *color* is neither a colormap nor a colour, *size* is not a
        supported figure size, or *stats* is ``True`` with fewer than
        two points.
    TypeError
        When *huac_table* is not a DataFrame or *color* is not a string.
    """
    _require_columns(huac_table)
    x, y, err = _filter_points(huac_table)
    figsize = _figsize(size)
    cmap, rgba = _resolve_point_color(color)
    lo, hi = _shared_log_limits(x, y)
    x_label = xlabel if xlabel is not None else r"Reference upstream area, km$^2$"
    y_label = ylabel if ylabel is not None else r"Upscaled upstream area, km$^2$"

    fit: tuple[float, float, float, float] | None = None
    if stats:
        fit = _log10_ols(x, y)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale("log")
        ax.set_yscale("log")

        if cmap is not None:
            vmin, vmax = _error_norm_limits(err)
            collection = ax.scatter(
                x,
                y,
                c=err,
                s=point_size,
                cmap=cmap,
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
                edgecolors="none",
                zorder=2,
            )
        else:
            collection = None
            ax.scatter(
                x,
                y,
                color=rgba,
                s=point_size,
                edgecolors="none",
                zorder=2,
            )

        if fit is not None:
            slope, intercept, r2, rmse = fit
            _add_fit_line(ax, slope, intercept, lo, hi)
            _add_stats_box(ax, r2, rmse)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if collection is not None and legend:
            _add_error_colorbar(fig, ax, collection, legend_title)

        add_titles(ax, title, subtitle)
        save(fig, output, dpi)
        return fig, ax
