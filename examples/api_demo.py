"""Full-pipeline demo using the fdup functional API.

No external data required — a synthetic DEM is built from numpy arrays.

Run:
    python examples/api_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine

import fdup
from fdup._core.types import Grid, GridType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K = 4                   # upscaling factor (must be even for DMM)
GRID_SIZE = 32          # fine-grid side length (32x32 → 8x8 coarse)
OUT_DIR = Path("examples/outputs")

# Pour point in CRS units (geographic degrees for this synthetic grid)
POUR_X = -80.05         # longitude
POUR_Y =  35.05         # latitude
SNAP_RADIUS = 0.5       # degrees (synthetic EPSG:4326 grid)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transform(n: int, *, origin_x: float, origin_y: float, res: float) -> Affine:
    """Build an affine transform for an n×n grid starting at (origin_x, origin_y)."""
    return Affine(res, 0.0, origin_x, 0.0, -res, origin_y)


def _make_dem(n: int, transform: Affine) -> Grid:
    """Create a synthetic DEM: a south-to-north slope with an east-to-west tilt."""
    rows = np.linspace(100.0, 0.0, n)          # decreasing elevation N→S
    cols = np.linspace(0.0, 50.0, n)           # slight W→E tilt
    arr = np.outer(rows, np.ones(n)) + np.outer(np.ones(n), cols)
    arr = arr.astype(np.float32)
    return Grid.create(array=arr, type=GridType.DEM, transform=transform)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== fdup functional-API demo ===\n")

    # -- 1. Warmup JIT kernels -----------------------------------------------
    print("[1/10] Warming up JIT kernels (first run compiles; subsequent runs are fast)…")
    fdup.warmup()
    print("       done.\n")

    # -- 2. Build synthetic DEM -----------------------------------------------
    res_fine = 0.1        # 0.1° pixel spacing
    origin_x = -81.6      # western edge
    origin_y =  38.2      # northern edge

    transform_fine = _make_transform(GRID_SIZE, origin_x=origin_x, origin_y=origin_y, res=res_fine)
    dem = _make_dem(GRID_SIZE, transform_fine)

    print(f"[2/10] Synthetic DEM: shape={dem.shape}, dtype={dem.array.dtype}, "
          f"transform={dem.meta.transform}")
    fdup.io.write(dem, OUT_DIR / "dem.tif", overwrite=True)
    print()

    # -- 3. Compute fine-resolution D8 flow direction -------------------------
    print("[3/10] Computing fine-resolution D8 flow directions…")
    fd_fine = fdup.utils.d8(dem, spherical=True)
    print(f"       FlowDir shape={fd_fine.shape}, dtype={fd_fine.array.dtype}")
    fdup.io.write(fd_fine, OUT_DIR / "flowdir_fine.tif", overwrite=True)
    print()

    # -- 4. Compute fine-resolution flow accumulation -------------------------
    print("[4/10] Computing fine-resolution flow accumulation (cell counts)…")
    fa_fine = fdup.utils.flow_accumulation(fd_fine, area=False)
    print(f"       FlowAcc shape={fa_fine.shape}, dtype={fa_fine.array.dtype}, "
          f"max={np.nanmax(fa_fine.array):.0f} cells")
    fdup.io.write(fa_fine, OUT_DIR / "flowacc_fine.tif", overwrite=True)
    print()

    # -- 5. Upscale with DMM --------------------------------------------------
    print(f"[5/10] Upscaling with DMM (k={K})…")
    fd_coarse = fdup.upscalers.DMM(fa_fine, k=K)
    print(f"       Coarse FlowDir shape={fd_coarse.shape}, dtype={fd_coarse.array.dtype}")
    fdup.io.write(fd_coarse, OUT_DIR / "flowdir_coarse_dmm.tif", overwrite=True)
    print()

    # -- 6. Compute coarse flow accumulation ----------------------------------
    print("[6/10] Computing coarse flow accumulation (cell counts)…")
    fa_coarse = fdup.utils.flow_accumulation(fd_coarse, area=False)
    print(f"       FlowAcc shape={fa_coarse.shape}, dtype={fa_coarse.array.dtype}, "
          f"max={np.nanmax(fa_coarse.array):.0f} cells")
    fdup.io.write(fa_coarse, OUT_DIR / "flowacc_coarse.tif", overwrite=True)
    print()

    # -- 7. Snap pour points --------------------------------------------------
    print(f"[7/10] Snapping pour points near ({POUR_X}, {POUR_Y}) "
          f"within radius={SNAP_RADIUS} degrees…")
    pour_row_fine, pour_col_fine = fdup.utils.snap_pour_cell(
        fa_fine, x=POUR_X, y=POUR_Y, radius=SNAP_RADIUS
    )
    pour_row_coarse, pour_col_coarse = fdup.utils.snap_pour_cell(
        fa_coarse, x=POUR_X, y=POUR_Y, radius=SNAP_RADIUS
    )
    print(f"       Fine pour cell:   row={pour_row_fine}, col={pour_col_fine}")
    print(f"       Coarse pour cell: row={pour_row_coarse}, col={pour_col_coarse}")
    print()

    # -- 8. Delineate watersheds ----------------------------------------------
    print("[8/10] Delineating watersheds…")
    ws_fine = fdup.utils.delineate_watershed(fd_fine, pour_row_fine, pour_col_fine)
    ws_coarse = fdup.utils.delineate_watershed(fd_coarse, pour_row_coarse, pour_col_coarse)
    fine_cells = int(ws_fine.array.sum())
    coarse_cells = int(ws_coarse.array.sum())
    print(f"       Fine watershed:   {fine_cells} cells  (shape={ws_fine.shape})")
    print(f"       Coarse watershed: {coarse_cells} cells (shape={ws_coarse.shape})")
    fdup.io.write(ws_fine, OUT_DIR / "watershed_fine.tif", overwrite=True)
    fdup.io.write(ws_coarse, OUT_DIR / "watershed_coarse.tif", overwrite=True)
    print()

    # -- 9. Disaggregate + align coarse watershed to fine grid ----------------
    print(f"[9/10] Disaggregating coarse mask by k={K} and aligning to fine grid…")
    ws_coarse_disagg = fdup.utils.disaggregate_mask(ws_coarse, k=K)
    ws_coarse_matched = fdup.utils.match_grids(reference=ws_fine, other=ws_coarse_disagg)
    print(f"       Disaggregated shape: {ws_coarse_disagg.shape}")
    print(f"       After match_grids:   {ws_coarse_matched.shape} (same as fine)")
    fdup.io.write(ws_coarse_matched, OUT_DIR / "watershed_coarse_matched.tif", overwrite=True)
    print()

    # -- 10. Compare watersheds -----------------------------------------------
    print("[10/10] Comparing fine vs coarse-upscaled watershed (squared Ochiai index)…")
    ochiai, intersection = fdup.evals.compare_watersheds(ws_fine, ws_coarse_matched)
    inter_cells = int(intersection.array.sum())
    fdup.io.write(intersection, OUT_DIR / "watershed_intersection.tif", overwrite=True)
    print(f"\n  Squared Ochiai index : {ochiai:.4f}")
    print(f"  Intersection cells   : {inter_cells}")
    print()

    print(f"All outputs written to '{OUT_DIR}/'")
    print("  dem.tif")
    print("  flowdir_fine.tif")
    print("  flowacc_fine.tif")
    print("  flowdir_coarse_dmm.tif")
    print("  flowacc_coarse.tif")
    print("  watershed_fine.tif")
    print("  watershed_coarse.tif")
    print("  watershed_coarse_matched.tif")
    print("  watershed_intersection.tif")


if __name__ == "__main__":
    main()
