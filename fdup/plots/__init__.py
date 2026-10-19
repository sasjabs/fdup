"""Plotting helpers for upscaling and evaluation results."""

from fdup.plots.accuracy import accurs_plot
from fdup.plots.huac import huac_scatterplot
from fdup.plots.maps import upscaled_map
from fdup.plots.windrose import windrose_plot

__all__ = [
    "upscaled_map",
    "accurs_plot",
    "huac_scatterplot",
    "windrose_plot",
]
