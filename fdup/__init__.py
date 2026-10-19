"""fdup — flow direction upscaling, evaluation, and utilities."""

from fdup import _core, io, upscalers, utils, evals, plots
from fdup._core import Grid, GridMeta, GridType, warmup

__all__ = [
    "_core",
    "io",
    "upscalers",
    "utils",
    "evals",
    "plots",
    "Grid",
    "GridMeta",
    "GridType",
    "warmup",
]

__version__ = "0.4.0"
