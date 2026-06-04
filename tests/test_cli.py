"""CLI integration tests for fdup.cli.main([...]) (Phase 5, Step 5.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.cli import main
from fdup import io as fdup_io
from fdup.upscalers import DMM
from fdup.utils import (
    d8,
    delineate_watershed,
    disaggregate_mask,
    flow_accumulation,
    match_grids,
    river_tree,
)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

FINE_TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 50.0)
N = 16    # fine grid edge length
K = 4     # upscaling factor (even, required by DMM)
NC = N // K  # coarse edge = 4

# Center of fine outlet pixel (row=15, col=15):
#   x = col_center = 15 + 0.5 = 15.5
#   y = 50 - row_center = 50 - 15.5 = 34.5
FINE_X, FINE_Y = 15.5, 34.5

# Center of coarse outlet pixel (row=3, col=3), coarse pixel size = 4°:
#   x = (3 + 0.5) * 4.0 = 14.0
#   y = 50 + (3 + 0.5) * (-4.0) = 36.0
COARSE_X, COARSE_Y = 14.0, 36.0


# ---------------------------------------------------------------------------
# Module-scoped fixture: build the full fine/coarse pipeline and write input
# rasters once so all subcommand tests can share them.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rasters(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_rasters")

    # SE-draining 16×16 DEM: elevation at (r,c) = (N-1-r)+(N-1-c)
    r_idx = np.arange(N, dtype=np.float32)[:, np.newaxis]
    c_idx = np.arange(N, dtype=np.float32)[np.newaxis, :]
    dem_arr = (N - 1 - r_idx) + (N - 1 - c_idx)
    dem = Grid.create(dem_arr, type=GridType.DEM, transform=FINE_TRANSFORM)

    fd_fine = d8(dem, spherical=False)
    fa_fine = flow_accumulation(fd_fine, area=False)
    fd_coarse = DMM(fa_fine, k=K)
    fa_coarse = flow_accumulation(fd_coarse, area=False)

    ws_fine = delineate_watershed(fd_fine, pour_row=N - 1, pour_col=N - 1)
    ws_coarse_disagg = disaggregate_mask(
        delineate_watershed(fd_coarse, pour_row=NC - 1, pour_col=NC - 1),
        k=K,
    )
    ws_coarse_matched = match_grids(reference=ws_fine, other=ws_coarse_disagg)

    _tree_grid, seeds = river_tree(fd_fine, fa_fine)

    def wr(g, name):
        p = d / name
        fdup_io.write(g, str(p), overwrite=True)
        return str(p)

    paths = {
        "dem":                wr(dem,                 "dem.tif"),
        "fd_fine":            wr(fd_fine,             "fd_fine.tif"),
        "fa_fine":            wr(fa_fine,             "fa_fine.tif"),
        "fd_coarse":          wr(fd_coarse,           "fd_coarse.tif"),
        "fa_coarse":          wr(fa_coarse,           "fa_coarse.tif"),
        "ws_fine":            wr(ws_fine,             "ws_fine.tif"),
        "ws_coarse_matched":  wr(ws_coarse_matched,   "ws_coarse_matched.tif"),
    }
    seeds_p = str(d / "seeds.npz")
    np.savez(seeds_p, seeds=seeds)
    paths["seeds"] = seeds_p
    return paths


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(argv: list[str]) -> None:
    """Call main(), treating SystemExit(0) as success (argparse --help path)."""
    try:
        main(argv)
    except SystemExit as exc:
        assert exc.code == 0, f"main() exited with non-zero code {exc.code}"


# ---------------------------------------------------------------------------
# dmm
# ---------------------------------------------------------------------------


def test_cli_dmm(rasters, tmp_path):
    out = tmp_path / "fd_dmm.tif"
    _run(["dmm", "--flowacc", rasters["fa_fine"], "-o", str(out), "-k", str(K)])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.FlowDir)
    assert g.meta.type == GridType.FlowDir
    assert g.shape == (NC, NC)


# ---------------------------------------------------------------------------
# nsa
# ---------------------------------------------------------------------------


def test_cli_nsa(rasters, tmp_path):
    out = tmp_path / "fd_nsa.tif"
    _run(["nsa", "--flowacc", rasters["fa_fine"], "-o", str(out), "-k", str(K)])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.FlowDir)
    assert g.meta.type == GridType.FlowDir
    assert g.shape == (NC, NC)


# ---------------------------------------------------------------------------
# cotat
# ---------------------------------------------------------------------------


def test_cli_cotat(rasters, tmp_path):
    out = tmp_path / "fd_cotat.tif"
    _run([
        "cotat",
        "--flowdir", rasters["fd_fine"],
        "--flowacc", rasters["fa_fine"],
        "-o", str(out),
        "-k", str(K),
    ])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.FlowDir)
    assert g.meta.type == GridType.FlowDir
    assert g.shape == (NC, NC)


# ---------------------------------------------------------------------------
# d8
# ---------------------------------------------------------------------------


def test_cli_d8(rasters, tmp_path):
    out = tmp_path / "fd_d8.tif"
    _run(["d8", "--dem", rasters["dem"], "-o", str(out), "--no-spherical"])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.FlowDir)
    assert g.meta.type == GridType.FlowDir
    assert g.shape == (N, N)


# ---------------------------------------------------------------------------
# flowacc
# ---------------------------------------------------------------------------


def test_cli_flowacc_area(rasters, tmp_path):
    out = tmp_path / "fa_area.tif"
    _run(["flowacc", "--flowdir", rasters["fd_fine"], "-o", str(out)])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.FlowAcc)
    assert g.meta.type == GridType.FlowAcc
    assert g.shape == (N, N)


def test_cli_flowacc_cells(rasters, tmp_path):
    out = tmp_path / "fa_cells.tif"
    _run(["flowacc", "--flowdir", rasters["fd_fine"], "-o", str(out), "--cells"])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.FlowAcc)
    assert g.shape == (N, N)


# ---------------------------------------------------------------------------
# watershed  (direct coordinate → grid cell, no flowacc snapping)
# ---------------------------------------------------------------------------


def test_cli_watershed_direct(rasters, tmp_path):
    out = tmp_path / "ws_direct.tif"
    _run([
        "watershed",
        "--flowdir", rasters["fd_fine"],
        "--x", str(FINE_X),
        "--y", str(FINE_Y),
        "--radius", "1.0",
        "-o", str(out),
    ])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.Mask)
    assert g.meta.type == GridType.Mask
    assert g.shape == (N, N)


# ---------------------------------------------------------------------------
# watershed  (with flowacc snapping)
# ---------------------------------------------------------------------------


def test_cli_watershed_snap(rasters, tmp_path):
    out = tmp_path / "ws_snap.tif"
    _run([
        "watershed",
        "--flowdir", rasters["fd_fine"],
        "--flowacc", rasters["fa_fine"],
        "--x", str(FINE_X),
        "--y", str(FINE_Y),
        "--radius", "1.0",
        "-o", str(out),
    ])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.Mask)
    assert g.meta.type == GridType.Mask
    assert g.shape == (N, N)


# ---------------------------------------------------------------------------
# huac
# ---------------------------------------------------------------------------


def test_cli_huac(rasters, tmp_path):
    o_raster = tmp_path / "err.tif"
    o_csv = tmp_path / "err.csv"
    _run([
        "huac",
        "--flowacc-fine",    rasters["fa_fine"],
        "--flowdir-coarse",  rasters["fd_coarse"],
        "--flowacc-coarse",  rasters["fa_coarse"],
        "--x",               str(COARSE_X),
        "--y",               str(COARSE_Y),
        "--radius",          "5.0",
        "-o-raster",         str(o_raster),
        "-o-csv",            str(o_csv),
    ])
    assert o_raster.exists()
    assert o_csv.exists()
    g = fdup_io.read(str(o_raster), grid_type=GridType.FlowAcc)
    assert g.shape == (NC, NC)
    df = pd.read_csv(str(o_csv))
    assert len(df.columns) > 0


# ---------------------------------------------------------------------------
# compare-watersheds  (stdout only)
# ---------------------------------------------------------------------------


def test_cli_compare_watersheds_stdout(rasters, tmp_path, capsys):
    _run([
        "compare-watersheds",
        "--mask1", rasters["ws_fine"],
        "--mask2", rasters["ws_coarse_matched"],
    ])
    text = capsys.readouterr().out.strip()
    ochiai = float(text)
    assert 0.0 <= ochiai <= 1.0


# ---------------------------------------------------------------------------
# compare-watersheds  (with optional intersection output)
# ---------------------------------------------------------------------------


def test_cli_compare_watersheds_with_output(rasters, tmp_path, capsys):
    out = tmp_path / "intersection.tif"
    _run([
        "compare-watersheds",
        "--mask1", rasters["ws_fine"],
        "--mask2", rasters["ws_coarse_matched"],
        "-o", str(out),
    ])
    assert out.exists()
    g = fdup_io.read(str(out), grid_type=GridType.Mask)
    assert g.shape == (N, N)


# ---------------------------------------------------------------------------
# compare-flowdir
# ---------------------------------------------------------------------------


def test_cli_compare_flowdir(rasters, tmp_path):
    out = tmp_path / "scores.npy"
    _run([
        "compare-flowdir",
        "--flowdir-fine",   rasters["fd_fine"],
        "--flowdir-coarse", rasters["fd_coarse"],
        "--seeds",          rasters["seeds"],
        "-o",               str(out),
        "--no-shuffle",
    ])
    assert out.exists()
    scores = np.load(str(out))
    assert scores.dtype == np.float32


# ---------------------------------------------------------------------------
# river-tree
# ---------------------------------------------------------------------------


def test_cli_river_tree(rasters, tmp_path):
    o_tree  = tmp_path / "tree.tif"
    o_seeds = tmp_path / "seeds_rt.npz"
    _run([
        "river-tree",
        "--flowdir",  rasters["fd_fine"],
        "--flowacc",  rasters["fa_fine"],
        "--o-tree",   str(o_tree),
        "--o-seeds",  str(o_seeds),
    ])
    assert o_tree.exists()
    assert o_seeds.exists()
    g = fdup_io.read(str(o_tree), grid_type=GridType.Tree)
    assert g.shape == (N, N)
    seeds = np.load(str(o_seeds))["seeds"]
    assert seeds.dtype.names is not None
