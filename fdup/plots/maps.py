"""Map of upscaled watershed, stream network, and optional reference outline.

Public API
----------
upscaled_map(upscaled_watershed, *, upscaled_network, reference_watershed, ...)
        -> tuple[Figure, Axes]
    Draw the upscaled watershed fill over an optional web-tile basemap, with
    optional stream network and reference-watershed outline.  Layers are
    reprojected to EPSG:3857.  Returns ``(fig, ax)`` and does not call
    ``plt.show()``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

from fdup.plots._style import RC, add_titles, save

__all__ = ["upscaled_map"]

_WEB_MERCATOR = "EPSG:3857"
_WGS84 = "EPSG:4326"

_DEFAULT_COLORS: dict[str, str] = {
    "uw": "#6B9DD1B3",
    "un": "#1D56BEE6",
    "ref": "#CC250FE6",
}
_DEFAULT_LINEWIDTH: dict[str, float] = {
    "un": 0.75,
    "ref": 1.5,
}
_DEFAULT_SCALING: dict[str, Any] = {
    "field": "flowacc",
    "range": (0.5, 2),
    "log": True,
}
_DEFAULT_LABELS: tuple[str, str, str] = (
    "Upscaled watershed",
    "Upscaled stream network",
    "Reference watershed",
)

_DEFAULT_MAP_HEIGHT_IN = 5.0
_MIN_FIG_IN = 3.0
_MAX_FIG_IN = 14.0

_LAYER_ZORDER = {"uw": 1, "un": 2, "ref": 3}

_MAP_FRAME_COLOR = "black"
_MAP_FRAME_LINEWIDTH = 0.8
_MAP_FRAME_ZORDER = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_mapping(value: Mapping[str, Any] | None, default: Mapping[str, Any], name: str) -> dict[str, Any]:
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    out = dict(default)
    out.update(value)
    return out


def _own_crs(obj: Any) -> Any:
    """CRS attached to a GeoDataFrame / GeoSeries, or None."""
    crs = getattr(obj, "crs", None)
    return crs if crs is not None else None


def _resolve_layer_crs(
    obj: Any,
    *,
    crs: Any,
    network_crs: Any,
    layer_name: str,
) -> Any:
    """Resolve CRS: the layer's own CRS, then *crs*, then the network CRS."""
    own = _own_crs(obj)
    if own is not None:
        return own
    if crs is not None:
        return crs
    if network_crs is not None:
        return network_crs
    raise ValueError(
        f"Cannot determine CRS for {layer_name}. "
        "Pass crs=... or provide a GeoSeries/GeoDataFrame with a CRS, "
        "or pass upscaled_network with a CRS."
    )


def _to_gdf(obj: Any, crs: Any, layer_name: str) -> gpd.GeoDataFrame:
    """Wrap a shapely geometry / GeoSeries / GeoDataFrame as a GeoDataFrame."""
    if isinstance(obj, gpd.GeoDataFrame):
        gdf = obj.copy()
        if gdf.crs is None:
            if crs is None:
                raise ValueError(
                    f"Cannot determine CRS for {layer_name}. "
                    "Pass crs=... or set a CRS on the GeoDataFrame."
                )
            gdf = gdf.set_crs(crs)
        return gdf
    if isinstance(obj, gpd.GeoSeries):
        gs = obj.copy()
        if gs.crs is None:
            if crs is None:
                raise ValueError(
                    f"Cannot determine CRS for {layer_name}. "
                    "Pass crs=... or set a CRS on the GeoSeries."
                )
            gs = gs.set_crs(crs)
        return gpd.GeoDataFrame(geometry=gs, crs=gs.crs)
    if isinstance(obj, BaseGeometry):
        if crs is None:
            raise ValueError(
                f"Cannot determine CRS for {layer_name}. "
                "Pass crs=... or provide a GeoSeries/GeoDataFrame with a CRS, "
                "or pass upscaled_network with a CRS."
            )
        return gpd.GeoDataFrame({"geometry": [obj]}, crs=crs)
    raise TypeError(
        f"{layer_name} must be a shapely geometry, GeoSeries or GeoDataFrame, "
        f"got {type(obj).__name__}."
    )


def _to_web_mercator(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Cannot reproject a layer that has no CRS.")
    return gdf.to_crs(_WEB_MERCATOR)


def _is_drawable(gdf: gpd.GeoDataFrame | None) -> bool:
    if gdf is None or gdf.empty:
        return False
    return not bool(gdf.geometry.is_empty.all())


def _union_bounds(layers: Sequence[gpd.GeoDataFrame]) -> tuple[float, float, float, float]:
    boxes = []
    for gdf in layers:
        if not _is_drawable(gdf):
            continue
        bounds = np.asarray(gdf.total_bounds, dtype=np.float64)
        if np.all(np.isfinite(bounds)):
            boxes.append(bounds)
    if not boxes:
        raise ValueError("Cannot determine map extent: all input layers are empty.")
    stacked = np.vstack(boxes)
    return (
        float(stacked[:, 0].min()),
        float(stacked[:, 1].min()),
        float(stacked[:, 2].max()),
        float(stacked[:, 3].max()),
    )


def _pad_bounds(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    padding: float,
) -> tuple[float, float, float, float]:
    span_x = xmax - xmin
    span_y = ymax - ymin
    if span_x <= 0.0 and span_y > 0.0:
        span_x = span_y
    elif span_y <= 0.0 and span_x > 0.0:
        span_y = span_x
    elif span_x <= 0.0 and span_y <= 0.0:
        span_x = span_y = 1.0
    pad_x = (padding / 100.0) * span_x
    pad_y = (padding / 100.0) * span_y
    return xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y


def _map_figsize(
    size: float | Sequence[float] | None,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float]:
    width = abs(xmax - xmin)
    height = abs(ymax - ymin)
    aspect = width / height if height > 0.0 else 1.0
    if size is None:
        fig_h = _DEFAULT_MAP_HEIGHT_IN
        fig_w = fig_h * aspect
        if fig_w > _MAX_FIG_IN:
            fig_w = _MAX_FIG_IN
            fig_h = fig_w / aspect if aspect else fig_h
        if fig_w < _MIN_FIG_IN:
            fig_w = _MIN_FIG_IN
            fig_h = fig_w / aspect if aspect else fig_h
        if fig_h < _MIN_FIG_IN:
            fig_h = _MIN_FIG_IN
            fig_w = fig_h * aspect
            if fig_w > _MAX_FIG_IN:
                fig_w = _MAX_FIG_IN
        return (float(fig_w), float(fig_h))
    if isinstance(size, (int, float)):
        fig_h = float(size)
        if fig_h <= 0.0:
            raise ValueError(f"size must be positive, got {size!r}.")
        return (fig_h * aspect, fig_h)
    seq = tuple(size)
    if len(seq) != 2:
        raise ValueError(
            f"size must be None, a number, or a (width, height) pair, got {size!r}."
        )
    fig_w, fig_h = float(seq[0]), float(seq[1])
    if fig_w <= 0.0 or fig_h <= 0.0:
        raise ValueError(f"size dimensions must be positive, got {size!r}.")
    return (fig_w, fig_h)


def _network_linewidths(
    gdf: gpd.GeoDataFrame,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Stretch *field* onto ``settings['range']``, optionally in log10 space."""
    field = settings["field"]
    if field not in gdf.columns:
        raise ValueError(
            f"Scaling field {field!r} is not in upscaled_network. "
            f"Available columns: {list(gdf.columns)}."
        )
    rng = settings["range"]
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            "un_scaling_settings['range'] must be a (min, max) pair of line widths."
        ) from exc

    values = np.asarray(gdf[field], dtype=np.float64)
    if bool(settings.get("log", True)):
        out = np.full(values.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(values) & (values > 0.0)
        out[positive] = np.log10(values[positive])
        values = out

    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(
            f"Scaling field {field!r} has no finite values to stretch."
        )

    vmin = float(np.min(values[finite]))
    vmax = float(np.max(values[finite]))
    widths = np.full(values.shape, lo, dtype=np.float64)
    if vmax > vmin:
        widths[finite] = lo + (values[finite] - vmin) / (vmax - vmin) * (hi - lo)
    else:
        widths[finite] = 0.5 * (lo + hi)
    return widths


def _legend_labels(
    legend_labels: Sequence[str] | None,
    has_un: bool,
    has_ref: bool,
) -> list[str]:
    """Pick labels for drawn layers (uw, then un, then ref)."""
    slots = (True, has_un, has_ref)
    n_present = sum(slots)
    if legend_labels is None:
        return [lab for lab, keep in zip(_DEFAULT_LABELS, slots) if keep]
    source = list(legend_labels)
    if len(source) == 3:
        return [lab for lab, keep in zip(source, slots) if keep]
    if len(source) == n_present:
        return source
    raise ValueError(
        "legend_labels must have length "
        f"{n_present} (one per drawn layer) or 3 (uw, un, ref slots), "
        f"got {len(source)}."
    )


def _resolve_basemap_provider(basemap_provider: Any):
    """Map ``'OSM'`` / a provider name / a TileProvider to a tile source."""
    if basemap_provider is None:
        return None
    import contextily as cx

    if isinstance(basemap_provider, str):
        if basemap_provider.upper() == "OSM":
            return cx.providers.OpenStreetMap.Mapnik
        return cx.providers.query_name(basemap_provider)
    return basemap_provider


def _add_basemap(ax: Axes, source: Any) -> None:
    import contextily as cx

    cx.add_basemap(
        ax,
        source=source,
        crs=_WEB_MERCATOR,
        reset_extent=False,
    )


def _scalebar_dx(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    """True metres per Web Mercator unit at the map centre latitude."""
    x = 0.5 * (xmin + xmax)
    y = 0.5 * (ymin + ymax)
    transformer = Transformer.from_crs(_WEB_MERCATOR, _WGS84, always_xy=True)
    _lon, lat = transformer.transform(x, y)
    if not np.isfinite(lat):
        return 1.0
    return float(np.cos(np.deg2rad(lat)))


def _add_scalebar(ax: Axes, dx: float) -> None:
    from matplotlib_scalebar.scalebar import ScaleBar

    ax.add_artist(ScaleBar(dx=dx, units="m", location="lower right"))


def _frame_map(ax: Axes) -> None:
    """Draw a full rectangular border and hide ticks and the grid.

    Shared rcParams turn the top and right spines off and enable the grid,
    which suits the other plots but not a map.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_MAP_FRAME_COLOR)
        spine.set_linewidth(_MAP_FRAME_LINEWIDTH)
        spine.set_zorder(_MAP_FRAME_ZORDER)


def _add_legend_below(
    ax: Axes,
    handles: Sequence[Any],
    labels: Sequence[str],
) -> None:
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=max(len(handles), 1),
        frameon=False,
        borderaxespad=0.0,
    )


def _prepare_network(
    upscaled_network: Any,
    *,
    crs: Any,
    un_scaling: bool,
    un_scaling_settings: Mapping[str, Any],
) -> tuple[gpd.GeoDataFrame | None, np.ndarray | None]:
    if upscaled_network is None:
        return None, None
    if isinstance(upscaled_network, gpd.GeoSeries):
        upscaled_network = gpd.GeoDataFrame(
            geometry=upscaled_network,
            crs=upscaled_network.crs,
        )
    if not isinstance(upscaled_network, gpd.GeoDataFrame):
        raise TypeError(
            "upscaled_network must be a GeoDataFrame, "
            f"got {type(upscaled_network).__name__}."
        )
    network_crs = _resolve_layer_crs(
        upscaled_network,
        crs=crs,
        network_crs=None,
        layer_name="upscaled_network",
    )
    gdf = _to_web_mercator(_to_gdf(upscaled_network, network_crs, "upscaled_network"))
    if not un_scaling:
        return gdf, None
    widths = _network_linewidths(gdf, un_scaling_settings)
    order = np.argsort(widths, kind="stable")
    return gdf.iloc[order], widths[order]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upscaled_map(
    upscaled_watershed: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame,
    *,
    upscaled_network: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    reference_watershed: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    colors: Mapping[str, Any] | None = None,
    linewidth: Mapping[str, float] | None = None,
    un_scaling: bool = False,
    un_scaling_settings: Mapping[str, Any] | None = None,
    padding: float = 10,
    basemap_provider: Any = "OSM",
    size: float | Sequence[float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    legend_labels: Sequence[str] | None = None,
    crs: Any = None,
    output: str | Path | None = "upscaled_map.png",
    dpi: float = 300,
) -> tuple[Figure, Axes]:
    """Draw an upscaled watershed map with optional network and reference outline.

    Layers are reprojected to EPSG:3857 so they line up with web tiles.
    The upscaled watershed is filled (no outline), the stream network is
    drawn on top, and the reference watershed is an outline only.  A thin
    black frame outlines the map; tick marks and the grid are omitted.

    Parameters
    ----------
    upscaled_watershed :
        Watershed polygon: a shapely geometry, GeoSeries or GeoDataFrame.
        Shapely geometries carry no CRS; pass *crs* or *upscaled_network*
        with a CRS in that case.
    upscaled_network :
        Optional stream-network GeoDataFrame (LineString rows).  A GeoSeries
        is accepted and wrapped.  ``None`` skips this layer.
    reference_watershed :
        Optional reference watershed polygon (same types as
        *upscaled_watershed*).  Drawn as an outline with no fill.
    colors :
        Mapping with keys ``'uw'``, ``'un'``, ``'ref'``.  Values are matplotlib
        colour specs (HEXA strings work).  Missing keys keep the defaults
        ``#6B9DD1B3``, ``#1D56BEE6``, ``#CC250FE6``.
    linewidth :
        Mapping with keys ``'un'`` and ``'ref'``.  ``'un'`` is used only when
        *un_scaling* is ``False``.  Defaults ``{'un': 0.75, 'ref': 1.5}``.
    un_scaling :
        When ``True``, scale network line widths from a numeric column.
    un_scaling_settings :
        ``{'field': 'flowacc', 'range': (0.5, 2), 'log': True}``.  *field*
        is read from *upscaled_network*; positive values are ``log10``-transformed
        when *log* is ``True``, then stretched linearly onto *range*.  Equal
        values use the midpoint of *range*.  Segments are drawn thin-to-thick
        so large rivers sit on top.  Raises if *field* is missing.
    padding :
        Padding around the union of layer bounds, as a percent of bbox width
        (x) and height (y).  Default ``10``.
    basemap_provider :
        ``'OSM'`` (OpenStreetMap Mapnik), any name resolved by
        ``contextily.providers.query_name`` (e.g. ``'CartoDB.Positron'``),
        an ``xyzservices.TileProvider``, or ``None`` to skip tiles.
    size :
        Figure size in inches.  ``None`` derives it from the padded bbox
        aspect ratio.  A number is the figure height (width follows the
        aspect).  A ``(width, height)`` pair is used as ``figsize``.
    title, subtitle :
        Optional texts centered above the axes.
    legend_labels :
        Labels for drawn layers, in order uw / un / ref.  A 3-element
        sequence is matched by slot (absent layers are skipped).  ``None``
        uses ``'Upscaled watershed'``, ``'Upscaled stream network'``,
        ``'Reference watershed'``.
    crs :
        Fallback CRS for layers that do not carry one (in particular a
        shapely geometry from :func:`fdup.utils.vectorize_watershed`).
    output :
        Path to write a PNG.  Default ``'upscaled_map.png'``.  ``None``
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
        When no CRS can be resolved, a scaling field is missing, *padding*
        is negative, or the map extent cannot be computed.
    TypeError
        When an input is not a supported geometry type.
    """
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}.")

    colors_map = _as_mapping(colors, _DEFAULT_COLORS, "colors")
    lw_map = _as_mapping(linewidth, _DEFAULT_LINEWIDTH, "linewidth")
    scaling = _as_mapping(un_scaling_settings, _DEFAULT_SCALING, "un_scaling_settings")
    uw_color = to_rgba(colors_map["uw"])
    un_color = to_rgba(colors_map["un"])
    ref_color = to_rgba(colors_map["ref"])
    un_lw = float(lw_map["un"])
    ref_lw = float(lw_map["ref"])

    network_crs = _own_crs(upscaled_network) if upscaled_network is not None else None

    uw_crs = _resolve_layer_crs(
        upscaled_watershed,
        crs=crs,
        network_crs=network_crs,
        layer_name="upscaled_watershed",
    )
    uw = _to_web_mercator(_to_gdf(upscaled_watershed, uw_crs, "upscaled_watershed"))

    network, net_widths = _prepare_network(
        upscaled_network,
        crs=crs,
        un_scaling=un_scaling,
        un_scaling_settings=scaling,
    )

    ref = None
    if reference_watershed is not None:
        ref_crs = _resolve_layer_crs(
            reference_watershed,
            crs=crs,
            network_crs=network_crs,
            layer_name="reference_watershed",
        )
        ref = _to_web_mercator(
            _to_gdf(reference_watershed, ref_crs, "reference_watershed")
        )

    extent_layers = [uw]
    if network is not None:
        extent_layers.append(network)
    if ref is not None:
        extent_layers.append(ref)
    xmin, ymin, xmax, ymax = _pad_bounds(*_union_bounds(extent_layers), padding)
    figsize = _map_figsize(size, xmin, ymin, xmax, ymax)

    has_un = _is_drawable(network)
    has_ref = _is_drawable(ref)
    labels = _legend_labels(legend_labels, has_un, has_ref)

    source = _resolve_basemap_provider(basemap_provider)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)

        if _is_drawable(uw):
            uw.plot(
                ax=ax,
                facecolor=uw_color,
                edgecolor="none",
                legend=False,
                zorder=_LAYER_ZORDER["uw"],
            )
        if has_un:
            plot_kw: dict[str, Any] = {
                "ax": ax,
                "color": un_color,
                "legend": False,
                "zorder": _LAYER_ZORDER["un"],
            }
            if net_widths is not None:
                plot_kw["linewidth"] = net_widths
            else:
                plot_kw["linewidth"] = un_lw
            network.plot(**plot_kw)
        if has_ref:
            ref.plot(
                ax=ax,
                facecolor="none",
                edgecolor=ref_color,
                linewidth=ref_lw,
                legend=False,
                zorder=_LAYER_ZORDER["ref"],
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        if source is not None:
            _add_basemap(ax, source)

        _add_scalebar(ax, _scalebar_dx(xmin, ymin, xmax, ymax))

        handles: list[Any] = [
            Patch(facecolor=uw_color, edgecolor="none"),
        ]
        if has_un:
            handles.append(Line2D([0], [0], color=un_color, linewidth=un_lw))
        if has_ref:
            handles.append(
                Patch(facecolor="none", edgecolor=ref_color, linewidth=ref_lw)
            )
        _add_legend_below(ax, handles, labels)

        _frame_map(ax)

        add_titles(ax, title, subtitle)
        save(fig, output, dpi)
        return fig, ax
