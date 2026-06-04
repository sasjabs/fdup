"""End-to-end pipeline tests for fdup.evals (Phase 4, Step 4.7)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.evals import (
    compare_flowdir,
    compare_watersheds,
    flowdir_windrose,
    huac,
    windrose_emd,
)
from fdup.evals.huac import _TABLE_COLUMNS
from fdup.upscalers import DMM
from fdup.utils import (
    d8,
    delineate_watershed,
    disaggregate_mask,
    flow_accumulation,
    match_grids,
    river_tree,
)

# 1° pixels, origin at (0°E, 50°N) — valid geographic coordinates so that
# geodesy helpers return positive areas and distances.  spherical=False is
# used for D8 to keep results fully deterministic.
FINE_TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 50.0)

N = 16   # fine-grid edge length
K = 4    # upscaling factor (must be even for DMM)
NC = N // K  # coarse-grid edge length = 4


def _make_fine_dem() -> Grid:
    """16×16 float32 DEM sloping toward the SE corner (15, 15).

    Elevation at (r, c) = (N-1-r) + (N-1-c), making (15,15) the unique
    lowest cell.  The steepest D8 descent for interior cells is SE because
    the diagonal elevation drop (2 units over sqrt(2) pixels) exceeds any
    cardinal drop (1 unit over 1 pixel), so all paths ultimately converge
    to the outlet at (15, 15).
    """
    rows = np.arange(N, dtype=np.float32)[:, np.newaxis]
    cols = np.arange(N, dtype=np.float32)[np.newaxis, :]
    arr = (N - 1 - rows) + (N - 1 - cols)
    return Grid.create(
        array=arr,
        type=GridType.DEM,
        transform=FINE_TRANSFORM,
        crs=None,
        nodata=float("nan"),
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture — build the full fine/coarse pipeline once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline():
    dem_fine = _make_fine_dem()

    fd_fine = d8(dem_fine, spherical=False)
    fa_fine = flow_accumulation(fd_fine, area=False)

    # DMM upscaling: 16×16 fine → 4×4 coarse (k=4, even ✓)
    fd_coarse = DMM(fa_fine, k=K)
    fa_coarse = flow_accumulation(fd_coarse, area=False)

    # Watershed delineation at the fine and coarse outlets
    ws_fine = delineate_watershed(fd_fine, pour_row=N - 1, pour_col=N - 1)
    ws_coarse_disagg = disaggregate_mask(
        delineate_watershed(fd_coarse, pour_row=NC - 1, pour_col=NC - 1),
        k=K,
    )
    ws_coarse_matched = match_grids(reference=ws_fine, other=ws_coarse_disagg)

    tree_grid, seeds = river_tree(fd_fine, fa_fine)

    return {
        "fd_fine": fd_fine,
        "fa_fine": fa_fine,
        "fd_coarse": fd_coarse,
        "fa_coarse": fa_coarse,
        "ws_fine": ws_fine,
        "ws_coarse_matched": ws_coarse_matched,
        "tree_grid": tree_grid,
        "seeds": seeds,
    }


# ---------------------------------------------------------------------------
# Step 4.7.9  — compare_watersheds
# ---------------------------------------------------------------------------


def test_compare_watersheds(pipeline) -> None:
    p = pipeline
    ochiai, inter = compare_watersheds(p["ws_fine"], p["ws_coarse_matched"])

    assert 0.0 <= ochiai <= 1.0, f"Ochiai coefficient must be in [0, 1], got {ochiai}"
    assert inter.meta.type == GridType.Mask, (
        f"intersection must be Mask, got {inter.meta.type}"
    )
    assert inter.shape == p["ws_fine"].shape, (
        f"intersection shape {inter.shape} != fine shape {p['ws_fine'].shape}"
    )


# ---------------------------------------------------------------------------
# Step 4.7.10 — huac
# ---------------------------------------------------------------------------


def test_huac(pipeline) -> None:
    p = pipeline
    err_grid, df = huac(
        p["fa_fine"],
        p["fd_coarse"],
        p["fa_coarse"],
        pour_row=NC - 1,
        pour_col=NC - 1,
    )

    assert err_grid.shape == p["fd_coarse"].shape, (
        f"error raster shape {err_grid.shape} must match coarse shape "
        f"{p['fd_coarse'].shape}"
    )
    assert err_grid.meta.type == GridType.FlowAcc, (
        f"error raster type must be FlowAcc, got {err_grid.meta.type}"
    )
    assert list(df.columns) == list(_TABLE_COLUMNS), (
        f"DataFrame columns {list(df.columns)} != expected {list(_TABLE_COLUMNS)}"
    )


# ---------------------------------------------------------------------------
# Step 4.7.11 — compare_flowdir
# ---------------------------------------------------------------------------


def test_compare_flowdir(pipeline) -> None:
    p = pipeline
    scores = compare_flowdir(p["fd_fine"], p["fd_coarse"], p["seeds"])

    assert scores.dtype == np.float32, (
        f"scores dtype must be float32, got {scores.dtype}"
    )
    assert scores.shape == (len(p["seeds"]),), (
        f"scores shape {scores.shape} must equal (len(seeds),) = ({len(p['seeds'])},)"
    )

    # Degenerate single-coarse-cell seeds return -1; all others must be in [0, 1].
    non_degenerate = scores[scores >= 0.0]
    if len(non_degenerate) > 0:
        assert non_degenerate.min() >= 0.0
        assert non_degenerate.max() <= 1.0


# ---------------------------------------------------------------------------
# Step 4.7.12 — flowdir_windrose
# ---------------------------------------------------------------------------


def test_flowdir_windrose(pipeline) -> None:
    p = pipeline
    wr_fine = flowdir_windrose(p["fd_fine"])
    wr_coarse = flowdir_windrose(p["fd_coarse"])

    for name, wr in (("fine", wr_fine), ("coarse", wr_coarse)):
        absolute, distribution = wr
        assert absolute.shape == (8,), (
            f"{name} windrose absolute_sums shape must be (8,), got {absolute.shape}"
        )
        assert distribution.shape == (8,), (
            f"{name} windrose distribution shape must be (8,), got {distribution.shape}"
        )
        assert np.all(absolute >= 0.0), (
            f"{name} absolute_sums must be non-negative"
        )


# ---------------------------------------------------------------------------
# Step 4.7.13 — windrose_emd
# ---------------------------------------------------------------------------


def test_windrose_emd(pipeline) -> None:
    p = pipeline
    _, dist_fine = flowdir_windrose(p["fd_fine"])
    _, dist_coarse = flowdir_windrose(p["fd_coarse"])

    if dist_fine.sum() == 0.0 or dist_coarse.sum() == 0.0:
        pytest.skip("one or both windroses are all-zero (empty/nodata-only grid)")

    emd = windrose_emd(dist_fine, dist_coarse)
    assert emd >= 0.0, f"EMD must be non-negative, got {emd}"


# ---------------------------------------------------------------------------
# Step 4.7.14 — compare_flowdir raises TypeError for non-structured seeds
# ---------------------------------------------------------------------------


def test_compare_flowdir_bad_seeds_raises(pipeline) -> None:
    p = pipeline
    bad_seeds = np.array([[0, 1, 2, 3, 4]], dtype=np.uint32)
    with pytest.raises(TypeError, match="seeds.dtype.names"):
        compare_flowdir(p["fd_fine"], p["fd_coarse"], bad_seeds)


# ---------------------------------------------------------------------------
# Step 4.7.15 — compare_watersheds raises ValueError on shape mismatch
# ---------------------------------------------------------------------------


def test_compare_watersheds_shape_mismatch_raises(pipeline) -> None:
    p = pipeline
    small_mask = Grid.create(
        array=np.ones((4, 4), dtype=bool),
        type=GridType.Mask,
        transform=FINE_TRANSFORM,
        crs=None,
    )
    with pytest.raises(ValueError):
        compare_watersheds(p["ws_fine"], small_mask)
