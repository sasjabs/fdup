"""Polar windrose of normalised 8-bin flow-direction distributions.

Public API
----------
windrose_plot(windrose, *, add_ref=None, ...) -> tuple[Figure, Axes]
    Grouped polar bars for one or more windrose distributions, with
    optional reference arcs.  Bins follow ``fdup._core.d8.COMPASS_ORDER``
    (E, SE, S, SW, W, NW, N, NE).  Returns ``(fig, ax)`` and does not
    call ``plt.show()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

from fdup._core.d8 import COMPASS_ORDER
from fdup.plots._style import RC, add_titles, legend_right, resolve_colors, save

__all__ = ["windrose_plot"]

# Azimuths in degrees, clockwise from north.  Keys match COMPASS_ORDER.
_AZIMUTH_DEG: dict[str, float] = {
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
    "N": 0.0,
    "NE": 45.0,
}

_CARDINAL = frozenset({"E", "N", "W", "S"})
_DIST_RADIUS = 0.125
_COUNT_CARDINAL = 0.147
_COUNT_DIAGONAL = 0.103

_DIRECTION_THETA = np.deg2rad(
    np.array([_AZIMUTH_DEG[name] for name in COMPASS_ORDER], dtype=np.float64)
)
_COUNT_RADII = np.array(
    [
        _COUNT_CARDINAL if name in _CARDINAL else _COUNT_DIAGONAL
        for name in COMPASS_ORDER
    ],
    dtype=np.float64,
)

_FULL_SECTOR_RAD = np.deg2rad(45.0)
_BAR_GROUP_RAD = _FULL_SECTOR_RAD * 0.8
_ARC_SAMPLES = 24
_RMAX_FACTOR = 1.05
_RLABEL_DEG = 67.5
_SUM_ATOL = 1e-6

_DEFAULT_WIDTH = 6.0
_DEFAULT_HEIGHT = 5.0

_GRID_COLOR = "0.85"
_GRID_LINESTYLE = "--"
_GRID_LINEWIDTH = 0.6
_SPINE_LINEWIDTH = 0.8

# Polar theta labels are center-anchored several points outside the spine,
# past the 4 pt offset used by add_titles.  Lift titles by this many points
# so they clear the north label the way the reference figure does.
_TITLE_CLEARANCE_PT = 28.0

_REF_CHOICES = (None, "dist", "count")

# Legend left edge in axes fraction; clears the "E" theta label.
_LEGEND_ANCHOR_X = 1.12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_distribution(values: Any, name: str) -> np.ndarray:
    """Require a length-8 distribution that sums to 1 within ``1e-6``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (8,):
        raise ValueError(
            f"{name} must have shape (8,) matching "
            f"fdup._core.d8.COMPASS_ORDER = {COMPASS_ORDER}.  "
            f"Got shape {arr.shape}."
        )
    total = float(arr.sum())
    if not np.isfinite(total) or abs(total - 1.0) > _SUM_ATOL:
        raise ValueError(
            f"{name} must be a normalised distribution summing to 1.0 "
            f"(±1e-6), but got sum={total:.8f}."
        )
    return arr


def _series_matrix(windrose: Any) -> np.ndarray:
    """Stack one or more windrose series into an ``(n, 8)`` array."""
    if isinstance(windrose, (str, bytes)):
        raise ValueError(
            "windrose must be an array of shape (8,) or (n, 8), "
            f"got {type(windrose).__name__}."
        )

    if isinstance(windrose, (list, tuple)):
        if len(windrose) == 0:
            raise ValueError("windrose must contain at least one series.")
        if all(np.ndim(item) == 0 for item in windrose):
            return _validate_distribution(windrose, "windrose")[None, :]
        rows = [
            _validate_distribution(item, f"windrose[{i}]")
            for i, item in enumerate(windrose)
        ]
        return np.stack(rows, axis=0)

    arr = np.asarray(windrose, dtype=np.float64)
    if arr.ndim == 1:
        return _validate_distribution(arr, "windrose")[None, :]
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("windrose must contain at least one series.")
        rows = [
            _validate_distribution(arr[i], f"windrose[{i}]")
            for i in range(arr.shape[0])
        ]
        return np.stack(rows, axis=0)
    raise ValueError(
        "windrose must be an array of shape (8,) or (n, 8), "
        f"got shape {arr.shape}."
    )


def _series_labels(legend_labels: Sequence[str] | None, n: int) -> list[str]:
    if legend_labels is None:
        return [str(i) for i in range(1, n + 1)]
    if isinstance(legend_labels, str):
        raise ValueError(
            f"legend_labels must have length {n}, got a string."
        )
    labels = [str(label) for label in legend_labels]
    if len(labels) != n:
        raise ValueError(
            f"legend_labels must have length {n}, got {len(labels)}."
        )
    return labels


def _reference_label_pair(ref_labels: Sequence[str]) -> tuple[str, str]:
    if isinstance(ref_labels, str):
        raise ValueError(
            f"ref_labels must have length 2, got {ref_labels!r}."
        )
    pair = tuple(ref_labels)
    if len(pair) != 2:
        raise ValueError(
            f"ref_labels must have length 2, got {ref_labels!r}."
        )
    return str(pair[0]), str(pair[1])


def _resolve_size(size: float | Sequence[float] | None) -> tuple[float, float]:
    if size is None:
        return (_DEFAULT_WIDTH, _DEFAULT_HEIGHT)
    if isinstance(size, bool) or isinstance(size, str):
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        )
    if isinstance(size, (int, float)):
        height = float(size)
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError(f"size must be positive, got {size!r}.")
        return (height * _DEFAULT_WIDTH / _DEFAULT_HEIGHT, height)
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
        width = float(seq[0])
        height = float(seq[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "size must be None, a number, or a (width, height) pair, "
            f"got {size!r}."
        ) from exc
    if (
        not np.isfinite(width)
        or not np.isfinite(height)
        or width <= 0.0
        or height <= 0.0
    ):
        raise ValueError(f"size dimensions must be positive, got {size!r}.")
    return (width, height)


def _rmax(data: np.ndarray, add_ref: str | None, custom: np.ndarray | None) -> float:
    peak = float(np.max(data))
    if add_ref == "dist":
        peak = max(peak, _DIST_RADIUS)
    elif add_ref == "count":
        peak = max(peak, float(np.max(_COUNT_RADII)))
    if custom is not None:
        peak = max(peak, float(np.max(custom)))
    return peak * _RMAX_FACTOR


def _draw_arcs(
    ax: Axes,
    radii: np.ndarray,
    linestyle: str,
    linewidth: float,
) -> None:
    """Draw one reference arc per compass sector at a constant radius."""
    half = _FULL_SECTOR_RAD / 2.0
    for theta, radius in zip(_DIRECTION_THETA, radii):
        angles = np.linspace(theta - half, theta + half, _ARC_SAMPLES)
        ax.plot(
            angles,
            np.full(angles.shape, float(radius)),
            color="black",
            linestyle=linestyle,
            linewidth=linewidth,
            solid_capstyle="butt",
        )


def _radial_ticks(rmax: float) -> np.ndarray:
    """A short run of nice radii inside *rmax* (about four rings)."""
    locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2)
    ticks = np.asarray(locator.tick_values(0.0, rmax), dtype=np.float64)
    ticks = ticks[(ticks > 0.0) & (ticks <= rmax + 1e-9)]
    if ticks.size == 0:
        return np.array([rmax], dtype=np.float64)
    return ticks


def _apply_radial_labels(ax: Axes, rmax: float) -> None:
    """Draw nice radial gridlines and label only the two largest."""
    ticks = _radial_ticks(rmax)
    ax.set_yticks(ticks)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    formatter.set_locs(ticks)
    keep = np.sort(ticks)[-2:]

    def _format(value: float, _pos: Any = None) -> str:
        if np.any(np.isclose(value, keep, rtol=0.0, atol=1e-8)):
            return formatter(float(value))
        return ""

    ax.yaxis.set_major_formatter(FuncFormatter(_format))


def _clear_polar_titles(ax: Axes) -> None:
    """Shift ``add_titles`` annotations above the north compass label."""
    for text in ax.texts:
        if getattr(text, "xycoords", None) != "axes fraction":
            continue
        x_off, y_off = text.xyann
        text.xyann = (x_off, float(y_off) + _TITLE_CLEARANCE_PT)


def _style_polar_frame(ax: Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color=_GRID_COLOR,
        linestyle=_GRID_LINESTYLE,
        linewidth=_GRID_LINEWIDTH,
    )
    spine = ax.spines["polar"]
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linestyle("solid")
    spine.set_linewidth(_SPINE_LINEWIDTH)


def _add_legend(
    ax: Axes,
    palette: Sequence[tuple],
    labels: Sequence[str],
    *,
    add_ref: str | None,
    has_custom: bool,
    ref_labels: tuple[str, str],
    linewidth: float,
    legend_title: str | None,
) -> None:
    handles: list[Any] = [
        Patch(facecolor=color, edgecolor="none") for color in palette
    ]
    texts = list(labels)
    ref_handles: list[Any] = []
    ref_texts: list[str] = []
    if add_ref is not None:
        ref_handles.append(
            Line2D([0], [0], color="black", linestyle="--", linewidth=linewidth)
        )
        ref_texts.append(ref_labels[0])
    if has_custom:
        ref_handles.append(
            Line2D([0], [0], color="black", linestyle=":", linewidth=linewidth)
        )
        ref_texts.append(ref_labels[1])
    if ref_handles:
        handles.append(Line2D([0], [0], color="none", linewidth=0))
        texts.append(" ")
        handles.extend(ref_handles)
        texts.extend(ref_texts)
    leg = legend_right(ax, handles, texts, legend_title)
    # The east compass label sits just outside the polar spine; shift the
    # legend right so it does not cover it.
    leg.set_bbox_to_anchor((_LEGEND_ANCHOR_X, 0.5))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def windrose_plot(
    windrose: Any,
    *,
    add_ref: str | None = None,
    add_custom_ref: Any = None,
    colors: str | Sequence[Any] = "set1",
    linewidth: float = 0.75,
    size: float | Sequence[float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    legend_labels: Sequence[str] | None = None,
    ref_labels: Sequence[str] = ("Theoretical distribution", "Reference distribution"),
    output: str | Path | None = "windrose_plot.png",
    dpi: float = 300,
) -> tuple[Figure, Axes]:
    """Draw a polar windrose of one or more 8-bin flow-direction distributions.

    Each direction is a sector of ``45° × 0.8`` centered on its compass
    azimuth.  Series share that sector as adjacent bars.  Optional
    reference arcs span the full 45° compass sector.

    Bin order is ``fdup._core.d8.COMPASS_ORDER``: E, SE, S, SW, W, NW, N, NE.
    Azimuths are degrees clockwise from north (E 90, SE 135, S 180, SW 225,
    W 270, NW 315, N 0, NE 45).

    Parameters
    ----------
    windrose :
        One distribution of shape ``(8,)``, a sequence of such arrays
        (one series each), or an array of shape ``(n, 8)``.  A sequence
        of eight scalars is a single series.  Every series must sum to 1
        within ``1e-6``.
    add_ref :
        Built-in reference arcs.  ``None`` draws none.  ``'dist'`` is a
        dashed arc at radius 0.125 in every direction.  ``'count'`` is a
        dashed arc at 0.147 on E, N, W, S and 0.103 on the diagonals.
    add_custom_ref :
        Optional length-8 distribution drawn as dotted arcs at those
        radii.  Same shape and sum checks as a windrose series.  May be
        combined with *add_ref*.
    colors :
        Colormap name (case-insensitive; default ``'set1'``) or a
        sequence of *n* colour specs.  Passed to
        :func:`fdup.plots._style.resolve_colors`.
    linewidth :
        Width of the reference arcs.  Default ``0.75``.
    size :
        Figure size in inches.  ``None`` uses ``(6.0, 5.0)``.  A number
        is the figure height; the width keeps the 6:5 aspect.  A
        ``(width, height)`` pair is used as ``figsize``.
    title, subtitle :
        Optional texts centered above the axes.
    legend :
        Draw the legend to the right of the axes.  Default ``True``.
    legend_title :
        Optional legend title.
    legend_labels :
        One label per series, in series order.  ``None`` uses ``'1'``,
        ``'2'``, and so on.
    ref_labels :
        Two strings.  Index 0 labels the *add_ref* arc and index 1 labels
        *add_custom_ref*.  A reference that is not drawn is omitted, and
        the blank spacer before the reference entries is omitted when no
        reference is drawn.
    output :
        Path to write a PNG.  Default ``'windrose_plot.png'``.  ``None``
        skips saving.
    dpi :
        Resolution used when *output* is not ``None``.  Default ``300``.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and polar axes.  ``plt.show()`` is not called and the
        figure is not closed.

    Raises
    ------
    ValueError
        When a series or *add_custom_ref* does not have shape ``(8,)`` or
        does not sum to 1 (±1e-6), when *add_ref* is not ``None``,
        ``'dist'``, or ``'count'``, when *legend_labels* or *ref_labels*
        have the wrong length, or when *size* is not ``None``, a positive
        number, or a positive ``(width, height)`` pair.
    """
    data = _series_matrix(windrose)
    n = int(data.shape[0])
    if add_ref not in _REF_CHOICES:
        raise ValueError(
            f"add_ref must be None, 'dist', or 'count', got {add_ref!r}."
        )
    custom = None
    if add_custom_ref is not None:
        custom = _validate_distribution(add_custom_ref, "add_custom_ref")
    labels = _series_labels(legend_labels, n)
    ref_pair = _reference_label_pair(ref_labels)
    palette = resolve_colors(colors, n)
    figsize = _resolve_size(size)
    rmax = _rmax(data, add_ref, custom)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": "polar"},
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(_DIRECTION_THETA)
        ax.set_xticklabels(list(COMPASS_ORDER))

        bar_width = _BAR_GROUP_RAD / n
        offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width
        for i in range(n):
            ax.bar(
                _DIRECTION_THETA + offsets[i],
                data[i],
                width=bar_width,
                align="center",
                edgecolor="none",
                color=palette[i],
            )

        if add_ref == "dist":
            _draw_arcs(ax, np.full(8, _DIST_RADIUS), "--", linewidth)
        elif add_ref == "count":
            _draw_arcs(ax, _COUNT_RADII, "--", linewidth)
        if custom is not None:
            _draw_arcs(ax, custom, ":", linewidth)

        ax.set_ylim(0.0, rmax)
        _apply_radial_labels(ax, rmax)
        ax.set_rlabel_position(_RLABEL_DEG)
        _style_polar_frame(ax)

        if legend:
            _add_legend(
                ax,
                palette,
                labels,
                add_ref=add_ref,
                has_custom=custom is not None,
                ref_labels=ref_pair,
                linewidth=linewidth,
                legend_title=legend_title,
            )
        add_titles(ax, title, subtitle)
        _clear_polar_titles(ax)
        save(fig, output, dpi)
        return fig, ax
