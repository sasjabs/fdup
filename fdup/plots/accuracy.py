"""Weighted-accuracy curves versus binned river counts or fractions.

Public API
----------
accurs_plot(accurs, bins, *, ...) -> tuple[Figure, Axes]
    Plot one or more accuracy curves on a logarithmic x axis.  Returns
    ``(fig, ax)`` and does not call ``plt.show()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fdup.plots._style import RC, add_titles, legend_right, resolve_colors, save

__all__ = ["accurs_plot"]

_DEFAULT_WIDTH = 6.0
_DEFAULT_HEIGHT = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_scalar(item: Any) -> bool:
    """True for a numeric (or otherwise non-sequence) element."""
    if isinstance(item, (str, bytes)):
        return True
    if isinstance(item, np.ndarray):
        return item.ndim == 0
    return bool(np.isscalar(item))


def _as_float_1d(item: Any, name: str, index: int | None) -> np.ndarray:
    """Convert one curve to a 1-D float array."""
    where = name if index is None else f"{name}[{index}]"
    try:
        arr = np.asarray(item, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be numeric.") from exc
    if arr.ndim != 1:
        raise ValueError(f"{where} must be 1-D, got shape {arr.shape}.")
    return arr


def _curves_from_array(arr: np.ndarray, name: str) -> list[np.ndarray]:
    """One curve for a 1-D array, or one curve per row of a 2-D array."""
    if arr.ndim == 1:
        return [_as_float_1d(arr, name, None)]
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one curve.")
        return [_as_float_1d(arr[i], name, i) for i in range(arr.shape[0])]
    raise ValueError(
        f"{name} must be a 1-D array, a sequence of 1-D arrays, or a 2-D "
        f"array with one curve per row, got shape {arr.shape}."
    )


def _curves_from_sequence(data: Sequence[Any], name: str) -> list[np.ndarray]:
    """One curve from a sequence of scalars, or one curve per array element."""
    if len(data) == 0:
        raise ValueError(f"{name} must contain at least one curve.")
    scalar_flags = [_is_scalar(item) for item in data]
    if all(scalar_flags):
        return [_as_float_1d(data, name, None)]
    if any(scalar_flags):
        raise ValueError(
            f"{name} mixes scalars and arrays. Pass one 1-D array, a "
            "sequence of 1-D arrays, or a 2-D array with one curve per row."
        )
    return [_as_float_1d(item, name, i) for i, item in enumerate(data)]


def _normalize_curves(data: Any, name: str) -> list[np.ndarray]:
    """Return one 1-D float array per curve in *data*."""
    if isinstance(data, np.ndarray) and data.dtype != object:
        return _curves_from_array(data, name)
    if isinstance(data, (list, tuple)):
        return _curves_from_sequence(data, name)
    if isinstance(data, np.ndarray) and data.dtype == object:
        return _curves_from_sequence(list(data), name)
    try:
        arr = np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a 1-D array, a sequence of 1-D arrays, or a 2-D "
            "array with one curve per row."
        ) from exc
    if arr.dtype == object:
        raise ValueError(
            f"{name} must be a 1-D array, a sequence of 1-D arrays, or a 2-D "
            "array with one curve per row."
        )
    return _curves_from_array(arr, name)


def _positive_finite_xlim(x_curves: Sequence[np.ndarray]) -> tuple[float, float]:
    """Min/max of positive finite bin values across all curves."""
    positive = np.concatenate(
        [x[np.isfinite(x) & (x > 0.0)] for x in x_curves]
    )
    if positive.size == 0:
        raise ValueError(
            "bins must contain at least one positive finite value for a "
            "logarithmic x axis."
        )
    return float(np.min(positive)), float(np.max(positive))


def _ylim(yrange: Sequence[float] | None) -> tuple[float, float]:
    """Resolve *yrange* to ``(ymin, ymax)``."""
    if yrange is None:
        return (0.0, 1.0)
    if isinstance(yrange, (str, bytes)):
        raise ValueError(
            f"yrange must be None or a (ymin, ymax) pair, got {yrange!r}."
        )
    try:
        seq = tuple(yrange)
    except TypeError as exc:
        raise ValueError(
            f"yrange must be None or a (ymin, ymax) pair, got {yrange!r}."
        ) from exc
    if len(seq) != 2:
        raise ValueError(
            f"yrange must be None or a (ymin, ymax) pair, got {yrange!r}."
        )
    try:
        ymin, ymax = float(seq[0]), float(seq[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"yrange must be None or a (ymin, ymax) pair, got {yrange!r}."
        ) from exc
    if ymin == ymax:
        raise ValueError(f"yrange bounds must differ, got {yrange!r}.")
    return (ymin, ymax)


def _is_scalar_number(size: Any) -> bool:
    if isinstance(size, (bool, np.bool_)):
        return False
    if isinstance(size, (int, float, np.number)):
        return True
    return (
        isinstance(size, np.ndarray)
        and size.shape == ()
        and np.issubdtype(size.dtype, np.number)
    )


def _figsize(size: float | Sequence[float] | None) -> tuple[float, float]:
    """Resolve *size* to ``(width, height)`` inches, keeping a 6:4 aspect."""
    if size is None:
        return (_DEFAULT_WIDTH, _DEFAULT_HEIGHT)
    if _is_scalar_number(size):
        height = float(size)
        if not height > 0.0:
            raise ValueError(f"size must be positive, got {size!r}.")
        return (height * _DEFAULT_WIDTH / _DEFAULT_HEIGHT, height)
    if isinstance(size, (str, bytes)):
        raise ValueError(
            f"size must be None, a number, or a (width, height) pair, got {size!r}."
        )
    try:
        seq = tuple(size)
    except TypeError as exc:
        raise ValueError(
            f"size must be None, a number, or a (width, height) pair, got {size!r}."
        ) from exc
    if len(seq) != 2:
        raise ValueError(
            f"size must be None, a number, or a (width, height) pair, got {size!r}."
        )
    try:
        width, height = float(seq[0]), float(seq[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"size must be None, a number, or a (width, height) pair, got {size!r}."
        ) from exc
    if not width > 0.0 or not height > 0.0:
        raise ValueError(f"size dimensions must be positive, got {size!r}.")
    return (width, height)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def accurs_plot(
    accurs,
    bins,
    *,
    colors: str | Sequence = "Set1",
    linewidth: float = 1.0,
    size: float | Sequence[float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    legend_labels: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    yrange: Sequence[float] | None = None,
    output: str | Path | None = "accurs_plot.png",
    dpi: float = 300,
) -> tuple[Figure, Axes]:
    """Plot weighted accuracy against river bins on a logarithmic x axis.

    Each curve is one ``(bins, accurs)`` pair.  A 1-D array is a single
    curve, a list or tuple of arrays is one curve per element, and a 2-D
    array is one curve per row.  Empty-bin NaNs are left in place so the
    line breaks there.

    Parameters
    ----------
    accurs :
        Accuracy values, one 1-D array per curve (see above).
    bins :
        Bin positions on the x axis, in the same layout and lengths as
        *accurs*.  The x axis is limited to the min and max of the
        positive finite values (empty-bin NaNs are ignored).
    colors :
        Colormap name or a sequence of colour specs, one per curve.
        Default ``'Set1'``.
    linewidth :
        Width of each accuracy line.  Default ``1.0``.
    size :
        Figure size in inches.  ``None`` uses ``(6.0, 4.0)``.  A number
        is the figure height and the width keeps the 6:4 aspect.  A
        ``(width, height)`` pair is used as ``figsize``.
    title, subtitle :
        Optional texts centered above the axes.
    legend :
        Draw a legend to the right of the axes when ``True``.
    legend_title :
        Optional legend title.
    legend_labels :
        One label per curve.  ``None`` uses ``'1'``, ``'2'``, ...
    xlabel, ylabel :
        Axis labels.  ``None`` uses ``'Fraction of rivers'`` and
        ``'Weighted accuracy score'``.  Pass ``xlabel='Number of rivers'``
        for count-bin plots.
    yrange :
        Y-axis limits.  ``None`` uses ``(0.0, 1.0)``.  A ``(ymin, ymax)``
        pair is passed to ``set_ylim``.
    output :
        Path to write a PNG.  Default ``'accurs_plot.png'``.  ``None``
        skips saving.
    dpi :
        Resolution used when *output* is not ``None``.  Default ``300``.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes.  ``plt.show()`` is not called.

    Raises
    ------
    ValueError
        When *accurs* and *bins* differ in the number of curves or in the
        length of a pair, when *legend_labels* does not match the number
        of curves, when *size* is not a number or a ``(width, height)``
        pair, when *yrange* is not a ``(ymin, ymax)`` pair with distinct
        bounds, or when *bins* has no positive finite value.
    """
    y_curves = _normalize_curves(accurs, "accurs")
    x_curves = _normalize_curves(bins, "bins")
    if len(y_curves) != len(x_curves):
        raise ValueError(
            "accurs and bins must contain the same number of curves, "
            f"got {len(y_curves)} and {len(x_curves)}."
        )
    for i, (y, x) in enumerate(zip(y_curves, x_curves)):
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"accurs and bins must have the same length for curve {i}, "
                f"got {y.shape[0]} and {x.shape[0]}."
            )

    n = len(y_curves)
    if legend_labels is None:
        labels = [str(i) for i in range(1, n + 1)]
    else:
        labels = [str(label) for label in legend_labels]
        if len(labels) != n:
            raise ValueError(
                f"legend_labels must have length {n} (one per curve), "
                f"got {len(labels)}."
            )

    palette = resolve_colors(colors, n)
    figsize = _figsize(size)
    xmin, xmax = _positive_finite_xlim(x_curves)
    ymin, ymax = _ylim(yrange)
    x_label = xlabel if xlabel is not None else "Fraction of rivers"
    y_label = ylabel if ylabel is not None else "Weighted accuracy score"

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        for x, y, color in zip(x_curves, y_curves, palette):
            (line,) = ax.plot(x, y, color=color, linewidth=linewidth)
            handles.append(line)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if legend:
            legend_right(ax, handles, labels, legend_title)
        add_titles(ax, title, subtitle)
        save(fig, output, dpi)
        return fig, ax
