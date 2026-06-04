"""fdup — flow direction upscaling, evaluation, and utilities."""

from fdup import _core, io, upscalers, utils, evals
from fdup._core import Grid, GridMeta, GridType, warmup

__all__ = [
    "_core",
    "io",
    "upscalers",
    "utils",
    "evals",
    "Grid",
    "GridMeta",
    "GridType",
    "warmup",
]

__version__ = "0.2.0"
