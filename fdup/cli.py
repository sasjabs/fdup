"""Command-line interface for the fdup toolkit.

Usage examples::

    fdup dmm   --flowacc flowacc.tif -o out.tif -k 4
    fdup nsa   --flowacc flowacc.tif -o out.tif -k 8
    fdup cotat --flowdir fdir.tif --flowacc facc.tif -o out.tif -k 4 --area-threshold 0.5
"""

import argparse

from fdup.upscalers import COTAT, DMM, NSA


def _run_dmm(args):
    upscaler = DMM()
    upscaler.load_flowacc(args.flowacc)
    upscaler.upscale(k=args.k)
    upscaler.save(args.output)


def _run_nsa(args):
    upscaler = NSA()
    upscaler.load_flowacc(args.flowacc)
    upscaler.upscale(k=args.k)
    upscaler.save(args.output)


def _run_cotat(args):
    upscaler = COTAT()
    upscaler.load_flowdir(args.flowdir)
    upscaler.load_flowacc(args.flowacc)
    upscaler.upscale(k=args.k, area_threshold=args.area_threshold)
    upscaler.save(args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fdup",
        description="Flow direction upscaling toolkit (DMM, NSA, COTAT)",
    )
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    # Shared arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--flowacc", required=True, help="Path to flow accumulation raster")
    common.add_argument("-o", "--output", required=True, help="Output flow direction raster path")
    common.add_argument("-k", type=int, required=True, help="Scaling factor")

    # DMM
    p_dmm = subparsers.add_parser("dmm", parents=[common], help="Double Maximum Method (DMM)")
    p_dmm.set_defaults(func=_run_dmm)

    # NSA
    p_nsa = subparsers.add_parser("nsa", parents=[common], help="Network Scaling Algorithm (NSA)")
    p_nsa.set_defaults(func=_run_nsa)

    # COTAT
    p_cotat = subparsers.add_parser(
        "cotat",
        parents=[common],
        help="Cell Outlet Tracing with an Area Threshold (COTAT)",
    )
    p_cotat.add_argument("--flowdir", required=True, help="Path to flow direction raster")
    p_cotat.add_argument(
        "--area-threshold",
        type=float,
        default=0.0,
        help="Area threshold for COTAT tracing (default: 0.0)",
    )
    p_cotat.set_defaults(func=_run_cotat)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
