"""Command-line interface for the fdup toolkit (functional API).

Usage examples::

    fdup dmm   --flowacc flowacc.tif -o out.tif -k 4
    fdup nsa   --flowacc flowacc.tif -o out.tif -k 8
    fdup cotat --flowdir fdir.tif --flowacc facc.tif -o out.tif -k 4
    fdup d8    --dem dem.tif -o flowdir.tif
    fdup flowacc --flowdir flowdir.tif -o flowacc.tif
    fdup watershed --flowdir flowdir.tif --x 10.5 --y 48.2 --radius 0.1 -o mask.tif
    fdup huac --flowacc-fine fa_fine.tif --flowdir-coarse fd_coarse.tif \\
              --flowacc-coarse fa_coarse.tif --x 10.5 --y 48.2 --radius 0.1 \\
              -o-raster err.tif -o-csv err.csv
    fdup compare-watersheds --mask1 ws1.tif --mask2 ws2.tif
    fdup compare-flowdir --flowdir-fine fd_fine.tif --flowdir-coarse fd_coarse.tif \\
                         --seeds seeds.npz -o scores.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from fdup._core.types import GridType
from fdup._core.warmup import warmup
from fdup import io as fdup_io
from fdup import upscalers, utils, evals


# ---------------------------------------------------------------------------
# Subcommand helpers
# ---------------------------------------------------------------------------


def _run_dmm(args):
    warmup()
    fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
    fd_coarse = upscalers.DMM(fa, k=args.k)
    fdup_io.write(fd_coarse, args.output, overwrite=True)


def _run_nsa(args):
    warmup()
    fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
    fd_coarse = upscalers.NSA(fa, k=args.k)
    fdup_io.write(fd_coarse, args.output, overwrite=True)


def _run_cotat(args):
    warmup()
    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
    fd_coarse = upscalers.COTAT(
        fd,
        fa,
        k=args.k,
        area_threshold=args.area_threshold,
        mufp=args.mufp,
    )
    fdup_io.write(fd_coarse, args.output, overwrite=True)


def _run_d8(args):
    warmup()
    dem = fdup_io.read(args.dem, grid_type=GridType.DEM)
    fd = utils.d8(dem, spherical=args.spherical)
    fdup_io.write(fd, args.output, overwrite=True)


def _run_flowacc(args):
    warmup()
    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    area = not args.cells
    fa = utils.flow_accumulation(fd, area=area)
    fdup_io.write(fa, args.output, overwrite=True)


def _run_watershed(args):
    warmup()
    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)

    if args.flowacc is not None:
        fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
        pour_row, pour_col = utils.snap_pour_cell(fa, x=args.x, y=args.y, radius=args.radius)
    else:
        # No flowacc provided: convert world coordinate directly to grid row/col.
        col_f, row_f = ~fd.meta.transform * (args.x, args.y)
        pour_row = int(round(row_f - 0.5))
        pour_col = int(round(col_f - 0.5))
        nrows, ncols = fd.shape
        if not (0 <= pour_row < nrows and 0 <= pour_col < ncols):
            raise ValueError(
                f"Pour point ({args.x}, {args.y}) maps to cell ({pour_row}, {pour_col}) "
                f"which is outside the grid (shape {nrows}×{ncols}). "
                "Provide --flowacc for radius-based snapping."
            )

    mask = utils.delineate_watershed(fd, pour_row=pour_row, pour_col=pour_col)
    fdup_io.write(mask, args.output, overwrite=True)


def _run_huac(args):
    warmup()
    fa_fine = fdup_io.read(args.flowacc_fine, grid_type=GridType.FlowAcc)
    fd_coarse = fdup_io.read(args.flowdir_coarse, grid_type=GridType.FlowDir)
    fa_coarse = fdup_io.read(args.flowacc_coarse, grid_type=GridType.FlowAcc)

    pour_row, pour_col = utils.snap_pour_cell(
        fa_coarse, x=args.x, y=args.y, radius=args.radius
    )

    err_grid, df = evals.huac(
        fa_fine,
        fd_coarse,
        fa_coarse,
        pour_row=pour_row,
        pour_col=pour_col,
        upstream_area_threshold=args.upstream_area_threshold,
    )

    fdup_io.write(err_grid, args.o_raster, overwrite=True)
    df.to_csv(args.o_csv, index=False)


def _run_compare_watersheds(args):
    warmup()
    mask1 = fdup_io.read(args.mask1, grid_type=GridType.Mask)
    mask2 = fdup_io.read(args.mask2, grid_type=GridType.Mask)
    ochiai, intersection = evals.compare_watersheds(mask1, mask2)
    print(f"{ochiai:.4f}")
    if args.output is not None:
        fdup_io.write(intersection, args.output, overwrite=True)


def _run_compare_flowdir(args):
    warmup()
    seeds_path = Path(args.seeds)
    if seeds_path.suffix.lower() != ".npz":
        raise ValueError(
            "expected an .npz file with a 'seeds' array "
            "(produced by fdup ... or numpy savez)"
        )
    loaded = np.load(seeds_path, allow_pickle=False)
    if "seeds" not in loaded:
        raise ValueError(
            "expected an .npz file with a 'seeds' array "
            "(produced by fdup ... or numpy savez)"
        )
    seeds = loaded["seeds"]

    fd_fine = fdup_io.read(args.flowdir_fine, grid_type=GridType.FlowDir)
    fd_coarse = fdup_io.read(args.flowdir_coarse, grid_type=GridType.FlowDir)

    scores = evals.compare_flowdir(
        fd_fine,
        fd_coarse,
        seeds,
        shuffle=not args.no_shuffle,
        alpha=args.alpha,
        strict_upstream=args.strict_upstream,
        cascade=args.cascade,
    )
    np.save(args.output, scores)


def _run_river_tree(args):
    warmup()
    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
    mask = None
    if args.mask is not None:
        mask = fdup_io.read(args.mask, grid_type=GridType.Mask)
    tree_grid, seeds = utils.river_tree(fd, fa, mask=mask)
    fdup_io.write(tree_grid, args.o_tree, overwrite=True)
    np.savez(args.o_seeds, seeds=seeds)


def _run_strahler(args):
    warmup()
    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    strahler_grid = utils.strahler_order(fd)
    fdup_io.write(strahler_grid, args.output, overwrite=True)


def _run_mask_seeds(args):
    warmup()
    seeds_path = Path(args.seeds)
    if seeds_path.suffix.lower() != ".npz":
        raise ValueError(
            "expected an .npz file with a 'seeds' array "
            "(produced by fdup river-tree or numpy savez)"
        )
    loaded = np.load(seeds_path, allow_pickle=False)
    if "seeds" not in loaded:
        raise ValueError(
            "expected an .npz file with a 'seeds' array "
            "(produced by fdup river-tree or numpy savez)"
        )
    seeds = loaded["seeds"]

    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
    mask = fdup_io.read(args.mask, grid_type=GridType.Mask)
    out_seeds = utils.mask_seeds(seeds, fd, fa, mask)
    np.savez(args.output, seeds=out_seeds)


# ---------------------------------------------------------------------------
# CLI type helpers
# ---------------------------------------------------------------------------

_CLI_GRID_TYPES: dict[str, GridType] = {
    "flowdir":  GridType.FlowDir,
    "flowacc":  GridType.FlowAcc,
    "dem":      GridType.DEM,
    "mask":     GridType.Mask,
    "strahler": GridType.Strahler,
    "tree":     GridType.Tree,
}


def _parse_grid_type(s: str) -> GridType:
    key = s.lower()
    if key not in _CLI_GRID_TYPES:
        raise argparse.ArgumentTypeError(
            f"Unknown grid type {s!r}. "
            f"Choose from: {', '.join(_CLI_GRID_TYPES)}"
        )
    return _CLI_GRID_TYPES[key]


def _run_threshold_mask(args):
    warmup()
    if args.flowacc is not None:
        grid = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc)
    else:
        grid = fdup_io.read(args.strahler, grid_type=GridType.Strahler)
    mask = utils.threshold_mask(grid, args.cutoff)
    fdup_io.write(mask, args.output, overwrite=True)


def _run_mask_grid(args):
    warmup()
    grid_type = _parse_grid_type(args.grid_type)
    grid = fdup_io.read(args.grid, grid_type=grid_type)
    mask = fdup_io.read(args.mask, grid_type=GridType.Mask)
    result = utils.mask_grid(grid, mask)
    fdup_io.write(result, args.output, overwrite=True)


def _run_crop_grid(args):
    warmup()
    grid_type = _parse_grid_type(args.grid_type)
    grid = fdup_io.read(args.grid, grid_type=grid_type)
    result = utils.crop_grid(grid)
    fdup_io.write(result, args.output, overwrite=True)


def _run_vectorize_network(args):
    warmup()
    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    fa = fdup_io.read(args.flowacc, grid_type=GridType.FlowAcc) if args.flowacc else None
    gdf = utils.vectorize_network(fd, flowacc=fa)
    gdf.to_file(args.output, driver="GPKG")


def _run_vectorize_tree(args):
    warmup()
    seeds_path = Path(args.seeds)
    if seeds_path.suffix.lower() != ".npz":
        raise ValueError(
            "expected an .npz file with a 'seeds' array "
            "(produced by fdup river-tree or numpy savez)"
        )
    loaded = np.load(seeds_path, allow_pickle=False)
    if "seeds" not in loaded:
        raise ValueError(
            "expected an .npz file with a 'seeds' array "
            "(produced by fdup river-tree or numpy savez)"
        )
    seeds = loaded["seeds"]

    fd = fdup_io.read(args.flowdir, grid_type=GridType.FlowDir)
    accuracy = np.load(args.accuracy) if args.accuracy else None

    gdf = utils.vectorize_tree(
        seeds,
        fd,
        cutoff=args.cutoff,
        rank=args.rank,
        accuracy=accuracy,
    )
    gdf.to_file(args.output, driver="GPKG")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fdup",
        description="Flow direction upscaling, evaluation, and utilities (functional API)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ------------------------------------------------------------------
    # Shared parent: flowacc + output + k (for legacy upscalers)
    # ------------------------------------------------------------------
    upscaler_parent = argparse.ArgumentParser(add_help=False)
    upscaler_parent.add_argument(
        "--flowacc", required=True, help="Path to flow accumulation raster"
    )
    upscaler_parent.add_argument(
        "-o", "--output", required=True, help="Output flow direction raster path"
    )
    upscaler_parent.add_argument("-k", type=int, required=True, help="Scaling factor")

    # ------------------------------------------------------------------
    # dmm
    # ------------------------------------------------------------------
    p_dmm = subparsers.add_parser(
        "dmm", parents=[upscaler_parent], help="Double Maximum Method (DMM)"
    )
    p_dmm.set_defaults(func=_run_dmm)

    # ------------------------------------------------------------------
    # nsa
    # ------------------------------------------------------------------
    p_nsa = subparsers.add_parser(
        "nsa", parents=[upscaler_parent], help="Network Scaling Algorithm (NSA)"
    )
    p_nsa.set_defaults(func=_run_nsa)

    # ------------------------------------------------------------------
    # cotat
    # ------------------------------------------------------------------
    p_cotat = subparsers.add_parser(
        "cotat",
        parents=[upscaler_parent],
        help="Cell Outlet Tracing with an Area Threshold (COTAT)",
    )
    p_cotat.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_cotat.add_argument(
        "--area-threshold",
        type=float,
        default=0.0,
        dest="area_threshold",
        help="Area threshold for COTAT tracing (default: 0.0)",
    )
    p_cotat.add_argument(
        "--mufp",
        type=float,
        default=None,
        help="Minimum Upstream Flow Path in metres enabling the COTAT+ "
             "outlet-selection scheme (default: disabled / plain COTAT)",
    )
    p_cotat.set_defaults(func=_run_cotat)

    # ------------------------------------------------------------------
    # d8
    # ------------------------------------------------------------------
    p_d8 = subparsers.add_parser("d8", help="Compute D8 flow directions from a DEM")
    p_d8.add_argument("--dem", required=True, help="Path to DEM raster")
    p_d8.add_argument("-o", "--output", required=True, help="Output flow direction raster path")
    p_d8.add_argument(
        "--spherical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use spherical (geographic) distance weighting (default: --spherical)",
    )
    p_d8.set_defaults(func=_run_d8)

    # ------------------------------------------------------------------
    # flowacc
    # ------------------------------------------------------------------
    p_flowacc = subparsers.add_parser(
        "flowacc", help="Compute flow accumulation from a D8 flow direction grid"
    )
    p_flowacc.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_flowacc.add_argument(
        "-o", "--output", required=True, help="Output flow accumulation raster path"
    )
    p_flowacc.add_argument(
        "--cells",
        action="store_true",
        default=False,
        help="Output cell count instead of area (default: area)",
    )
    p_flowacc.set_defaults(func=_run_flowacc)

    # ------------------------------------------------------------------
    # watershed
    # ------------------------------------------------------------------
    p_ws = subparsers.add_parser(
        "watershed", help="Delineate a watershed from a pour point"
    )
    p_ws.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_ws.add_argument("--x", type=float, required=True, help="Pour point X coordinate (CRS units)")
    p_ws.add_argument("--y", type=float, required=True, help="Pour point Y coordinate (CRS units)")
    p_ws.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Snap radius in CRS units (degrees for EPSG:4326)",
    )
    p_ws.add_argument("-o", "--output", required=True, help="Output mask raster path")
    p_ws.add_argument(
        "--flowacc",
        default=None,
        help="Optional flow accumulation raster for snapping the pour point",
    )
    p_ws.set_defaults(func=_run_watershed)

    # ------------------------------------------------------------------
    # huac
    # ------------------------------------------------------------------
    p_huac = subparsers.add_parser(
        "huac", help="Hierarchical Upstream Area Comparison (HUAC)"
    )
    p_huac.add_argument(
        "--flowacc-fine", required=True, dest="flowacc_fine",
        help="Fine-resolution flow accumulation raster"
    )
    p_huac.add_argument(
        "--flowdir-coarse", required=True, dest="flowdir_coarse",
        help="Coarse-resolution flow direction raster"
    )
    p_huac.add_argument(
        "--flowacc-coarse", required=True, dest="flowacc_coarse",
        help="Coarse-resolution flow accumulation raster"
    )
    p_huac.add_argument("--x", type=float, required=True, help="Pour point X coordinate (CRS units)")
    p_huac.add_argument("--y", type=float, required=True, help="Pour point Y coordinate (CRS units)")
    p_huac.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Snap radius in CRS units for pour point snapping",
    )
    p_huac.add_argument(
        "-o-raster", "--o-raster", required=True, dest="o_raster",
        help="Output error raster path (GeoTIFF)"
    )
    p_huac.add_argument(
        "-o-csv", "--o-csv", required=True, dest="o_csv",
        help="Output per-cell DataFrame CSV path"
    )
    p_huac.add_argument(
        "--upstream-area-threshold",
        type=float,
        default=0.0,
        dest="upstream_area_threshold",
        help="Coarse cells below this accumulation value are skipped in the "
             "error raster but traversal continues through them (default: 0.0)",
    )
    p_huac.set_defaults(func=_run_huac)

    # ------------------------------------------------------------------
    # compare-watersheds
    # ------------------------------------------------------------------
    p_cws = subparsers.add_parser(
        "compare-watersheds", help="Compute the squared Ochiai overlap index for two masks"
    )
    p_cws.add_argument("--mask1", required=True, help="First watershed mask raster")
    p_cws.add_argument("--mask2", required=True, help="Second watershed mask raster")
    p_cws.add_argument(
        "-o", "--output", default=None,
        help="Optional output path for the intersection mask raster"
    )
    p_cws.set_defaults(func=_run_compare_watersheds)

    # ------------------------------------------------------------------
    # compare-flowdir
    # ------------------------------------------------------------------
    p_cfd = subparsers.add_parser(
        "compare-flowdir",
        help="Per-seed flow direction accuracy scoring (d8comp)",
    )
    p_cfd.add_argument(
        "--flowdir-fine", required=True, dest="flowdir_fine",
        help="Fine-resolution flow direction raster"
    )
    p_cfd.add_argument(
        "--flowdir-coarse", required=True, dest="flowdir_coarse",
        help="Coarse-resolution flow direction raster"
    )
    p_cfd.add_argument(
        "--seeds", required=True,
        help=".npz file with a 'seeds' structured array (from fdup river-tree or np.savez)"
    )
    p_cfd.add_argument("-o", "--output", required=True, help="Output .npy file for per-seed scores")
    p_cfd.add_argument(
        "--alpha", type=float, default=0.0,
        help="Harmonic decay coefficient for off-path deviation penalty (default: 0.0)"
    )
    p_cfd.add_argument(
        "--no-shuffle", action="store_true", default=False,
        help="Disable randomised seed processing order"
    )
    p_cfd.add_argument(
        "--strict-upstream", action="store_true", default=False, dest="strict_upstream",
        help="Treat coarse cells landing upstream as incorrect"
    )
    p_cfd.add_argument(
        "--cascade", action="store_true", default=False,
        help="Propagate correctness in upstream order (implies --strict-upstream)"
    )
    p_cfd.set_defaults(func=_run_compare_flowdir)

    # ------------------------------------------------------------------
    # river-tree (optional; produces seeds for compare-flowdir)
    # ------------------------------------------------------------------
    p_rt = subparsers.add_parser(
        "river-tree",
        help="Extract a river tree and seed list from a flow direction + flow accumulation grid",
    )
    p_rt.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_rt.add_argument("--flowacc", required=True, help="Path to flow accumulation raster")
    p_rt.add_argument("--mask", default=None, help="Optional mask raster to restrict the tree")
    p_rt.add_argument("--o-tree", required=True, dest="o_tree", help="Output tree raster path")
    p_rt.add_argument(
        "--o-seeds", required=True, dest="o_seeds",
        help="Output seeds .npz path (use with compare-flowdir --seeds)"
    )
    p_rt.set_defaults(func=_run_river_tree)

    # ------------------------------------------------------------------
    # strahler
    # ------------------------------------------------------------------
    p_strahler = subparsers.add_parser(
        "strahler",
        help="Compute Strahler stream orders from a D8 flow direction grid",
    )
    p_strahler.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_strahler.add_argument(
        "-o", "--output", required=True, help="Output Strahler order raster path (uint8)"
    )
    p_strahler.set_defaults(func=_run_strahler)

    # ------------------------------------------------------------------
    # mask-seeds
    # ------------------------------------------------------------------
    p_ms = subparsers.add_parser(
        "mask-seeds",
        help="Prune a seeds .npz to in-mask sub-segments, re-computing acc and length_m",
    )
    p_ms.add_argument(
        "--seeds", required=True,
        help=".npz file with a 'seeds' structured array (from fdup river-tree or np.savez)"
    )
    p_ms.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_ms.add_argument("--flowacc", required=True, help="Path to flow accumulation raster")
    p_ms.add_argument("--mask", required=True, help="Path to mask raster (GridType.Mask)")
    p_ms.add_argument("-o", "--output", required=True, help="Output seeds .npz path")
    p_ms.set_defaults(func=_run_mask_seeds)

    # ------------------------------------------------------------------
    # threshold-mask
    # ------------------------------------------------------------------
    p_tm = subparsers.add_parser(
        "threshold-mask",
        help="Create a boolean mask where grid values are >= cutoff",
    )
    _tm_group = p_tm.add_mutually_exclusive_group(required=True)
    _tm_group.add_argument(
        "--flowacc", default=None,
        help="Path to a flow accumulation raster (GridType.FlowAcc)",
    )
    _tm_group.add_argument(
        "--strahler", default=None,
        help="Path to a Strahler order raster (GridType.Strahler)",
    )
    p_tm.add_argument(
        "--cutoff", type=float, required=True,
        help="Threshold value (inclusive); cells >= cutoff become True",
    )
    p_tm.add_argument("-o", "--output", required=True, help="Output mask raster path")
    p_tm.set_defaults(func=_run_threshold_mask)

    # ------------------------------------------------------------------
    # mask-grid
    # ------------------------------------------------------------------
    p_mg = subparsers.add_parser(
        "mask-grid",
        help="Apply a boolean mask to a grid, setting nodata where mask is False",
    )
    p_mg.add_argument("--grid", required=True, help="Path to the input raster")
    p_mg.add_argument(
        "--grid-type", required=True, dest="grid_type",
        metavar="TYPE",
        help=(
            "Semantic type of --grid. "
            f"Choices: {', '.join(_CLI_GRID_TYPES)}"
        ),
    )
    p_mg.add_argument("--mask", required=True, help="Path to the mask raster (GridType.Mask)")
    p_mg.add_argument("-o", "--output", required=True, help="Output raster path")
    p_mg.set_defaults(func=_run_mask_grid)

    # ------------------------------------------------------------------
    # crop-grid
    # ------------------------------------------------------------------
    p_cg = subparsers.add_parser(
        "crop-grid",
        help="Trim a grid to the minimal bounding box of data (non-nodata) cells",
    )
    p_cg.add_argument("--grid", required=True, help="Path to the input raster")
    p_cg.add_argument(
        "--grid-type", required=True, dest="grid_type",
        metavar="TYPE",
        help=(
            "Semantic type of --grid. "
            f"Choices: {', '.join(_CLI_GRID_TYPES)}"
        ),
    )
    p_cg.add_argument("-o", "--output", required=True, help="Output raster path")
    p_cg.set_defaults(func=_run_crop_grid)

    # ------------------------------------------------------------------
    # vectorize-network
    # ------------------------------------------------------------------
    p_vn = subparsers.add_parser(
        "vectorize-network",
        help="Vectorize a D8 network into LineString segments (GeoPackage output)",
    )
    p_vn.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_vn.add_argument(
        "--flowacc", default=None,
        help="Optional flow accumulation raster; attached as 'flowacc' attribute",
    )
    p_vn.add_argument("-o", "--output", required=True, help="Output GeoPackage path (.gpkg)")
    p_vn.set_defaults(func=_run_vectorize_network)

    # ------------------------------------------------------------------
    # vectorize-tree
    # ------------------------------------------------------------------
    p_vt = subparsers.add_parser(
        "vectorize-tree",
        help="Convert a seeds .npz to a LineString GeoDataFrame (GeoPackage output)",
    )
    p_vt.add_argument(
        "--seeds", required=True,
        help=".npz file with a 'seeds' structured array (from fdup river-tree)",
    )
    p_vt.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_vt.add_argument(
        "--cutoff", type=float, default=None,
        help="Keep only seeds with acc >= cutoff (mutually exclusive with --rank)",
    )
    p_vt.add_argument(
        "--rank", type=int, default=None,
        help="Keep the top-N seeds by acc (mutually exclusive with --cutoff)",
    )
    p_vt.add_argument(
        "--accuracy", default=None,
        help="Optional .npy file of per-seed accuracy values (same length as seeds)",
    )
    p_vt.add_argument("-o", "--output", required=True, help="Output GeoPackage path (.gpkg)")
    p_vt.set_defaults(func=_run_vectorize_tree)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
