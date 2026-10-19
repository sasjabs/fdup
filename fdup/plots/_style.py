"""Shared academic style helpers for ``fdup.plots``.

Public API
----------
RC : dict
    matplotlib rcParams for a clean academic look.  Plot functions apply
    this via ``matplotlib.rc_context(RC)`` so the user's global rcParams
    are never changed.

add_titles(ax, title, subtitle) -> None
    Bold title and smaller gray subtitle, both centered on the axes (not
    the figure) so they stay over the plot when a legend sits to the right.

resolve_colors(colors, n) -> list[tuple]
    Resolve a colormap name or a sequence of color specs to *n* RGBA tuples.
    Palette names are matched case-insensitively (``'set1'`` → ``Set1``).

legend_right(ax, handles, labels, title) -> matplotlib.legend.Legend
    Vertical legend outside the axes, right side.

save(fig, output, dpi) -> None
    Write *fig* with ``bbox_inches='tight'``.  ``output=None`` skips saving.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.legend import Legend

__all__ = [
    "RC",
    "add_titles",
    "resolve_colors",
    "legend_right",
    "save",
]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "0.75",
    "legend.framealpha": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_TITLE_SIZE = 10
_SUBTITLE_SIZE = 9
_SUBTITLE_COLOR = "0.4"
_TITLE_STACK_GAP = 2.0
_TITLE_AXES_PAD = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_titles(
    ax: Axes,
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    """Place a bold title and a smaller gray subtitle centered on *ax*.

    Both texts are attached to the axes (not the figure) so they stay
    centered on the plot when a legend sits outside to the right.
    ``None`` and empty strings are skipped.
    """
    title = title if title else None
    subtitle = subtitle if subtitle else None
    if title is None and subtitle is None:
        return

    # Stack upward from the axes top: subtitle (closer) then title.
    offset = _TITLE_AXES_PAD
    if subtitle is not None:
        text = ax.annotate(
            str(subtitle),
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_SUBTITLE_SIZE,
            color=_SUBTITLE_COLOR,
            annotation_clip=False,
        )
        text.set_in_layout(True)
        offset += _SUBTITLE_SIZE + _TITLE_STACK_GAP
    if title is not None:
        text = ax.annotate(
            str(title),
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_TITLE_SIZE,
            fontweight="bold",
            annotation_clip=False,
        )
        text.set_in_layout(True)


def resolve_colors(colors: str | Sequence[Any], n: int) -> list[tuple]:
    """Resolve *colors* to a list of *n* RGBA tuples.

    Parameters
    ----------
    colors :
        A matplotlib colormap name (matched case-insensitively) or a
        sequence of color specifications.  HEXA strings are accepted.
    n :
        Number of colors to return.  A sequence must have this length.

    Returns
    -------
    list of tuple
        *n* ``(r, g, b, a)`` tuples.

    Notes
    -----
    Qualitative palettes (``ListedColormap`` with at most 20 swatches, e.g.
    Set1 / tab10) contribute their first *n* colors.  Continuous palettes
    are sampled with ``linspace(0, 1, n)``.  Matplotlib 3.8+ stores viridis
    and similar maps as 256-color ``ListedColormap`` instances, so the
    swatch count, not the class, is what distinguishes the two.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")

    if isinstance(colors, str):
        cmap = _get_cmap(colors)
        if _is_qualitative(cmap):
            palette = [to_rgba(c) for c in cmap.colors]
            if n > len(palette):
                raise ValueError(
                    f"Qualitative colormap {colors!r} has {len(palette)} "
                    f"colors, but {n} were requested."
                )
            return palette[:n]
        sampled = np.atleast_2d(cmap(np.linspace(0.0, 1.0, n)))
        return [tuple(float(x) for x in row) for row in sampled]

    try:
        seq = list(colors)
    except TypeError as exc:
        raise TypeError(
            "colors must be a colormap name or a sequence of colors, "
            f"got {type(colors).__name__}."
        ) from exc

    if len(seq) != n:
        raise ValueError(
            f"colors list must have length {n}, got {len(seq)}."
        )

    resolved: list[tuple] = []
    for i, spec in enumerate(seq):
        try:
            resolved.append(to_rgba(spec))
        except ValueError as exc:
            raise ValueError(f"Invalid color at index {i}: {spec!r}.") from exc
    return resolved


def legend_right(
    ax: Axes,
    handles: Sequence[Any],
    labels: Sequence[str],
    title: str | None = None,
) -> Legend:
    """Place a vertical legend outside *ax*, to the right.

    Uses ``loc='center left'`` and ``bbox_to_anchor=(1.02, 0.5)``.
    """
    handles = list(handles)
    labels = list(labels)
    if len(handles) != len(labels):
        raise ValueError(
            "handles and labels must have the same length, "
            f"got {len(handles)} and {len(labels)}."
        )
    return ax.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )


def save(
    fig: Figure,
    output: str | Path | None,
    dpi: float = 300,
) -> None:
    """Write *fig* to *output* with ``bbox_inches='tight'``.

    Passing ``None`` skips saving so legends outside the axes can still
    be inspected on the returned figure without writing a file.
    """
    if output is None:
        return
    fig.savefig(output, dpi=dpi, bbox_inches="tight")


# Largest builtin qualitative map is tab20 / tab20b / tab20c (20 swatches).
_QUALITATIVE_MAX_N = 20


def _is_qualitative(cmap) -> bool:
    """True for discrete palettes such as Set1, as opposed to viridis/jet."""
    return isinstance(cmap, ListedColormap) and cmap.N <= _QUALITATIVE_MAX_N


def _get_cmap(name: str):
    """Look up a matplotlib colormap, matching *name* case-insensitively."""
    if name in colormaps:
        return colormaps[name]
    target = name.lower()
    matches = [registered for registered in colormaps if registered.lower() == target]
    if len(matches) == 1:
        return colormaps[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            f"Colormap name {name!r} is ambiguous among {matches}."
        )
    raise ValueError(
        f"{name!r} is not a recognised matplotlib colormap."
    )
