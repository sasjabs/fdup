# fdup

`fdup` is a Python toolkit for upscaling D8 flow direction grids, computing flow accumulation and watersheds, and evaluating the accuracy of upscaled results.

## Implemented upscaling algorithms

- **DMM** (Double Maximum Method) — [Olivera et al., 2002](https://doi.org/10.1029/2001WR000726)
- **NSA** (Network Scaling Algorithm) — [Fekete et al., 2001](https://doi.org/10.1029/2001WR900024)
- **COTAT / COTAT+** (Cell Outlet Tracing with an Area Threshold) — [Reed, 2003](https://doi.org/10.1029/2003WR001989)

---

## Install

From PyPI:

```bash
pip install fdup
```

For development (editable install with test dependencies):

```bash
git clone https://github.com/sasjabs/fdup
cd fdup
pip install -e .[dev]
```

Requirements: Python `>=3.10`, `numpy`, `numba`, `rasterio`, `pandas`, `affine`, `geopandas>=0.14`, `shapely>=2.0`.

---



## Quickstart

```python
import fdup

# Pre-compile all numba kernels (~30 s on first run; cached afterwards).
fdup.warmup()

fa = fdup.io.read("flowacc.tif", grid_type=fdup.GridType.FlowAcc)
fd_coarse = fdup.upscalers.DMM(fa, k=4)
fdup.io.write(fd_coarse, "flowdir_coarse.tif", overwrite=True)
```

See `examples/api_demo.py` for a fully self-contained pipeline that runs on a synthetic DEM.

---



## Submodule reference



### `fdup.io`


| Function                                    | Description                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| `read(path, grid_type)`                     | Read a GeoTIFF into a `Grid`. Validates dtype against the requested `GridType`. |
| `write(grid, path, *, overwrite, compress)` | Write a `Grid` to a GeoTIFF.                                                    |




### `fdup.upscalers`


| Function                                              | Description                                                           |
| ----------------------------------------------------- | --------------------------------------------------------------------- |
| `DMM(flowacc, k)`                                     | Double Maximum Method. Requires even `k`. Returns `GridType.FlowDir`. |
| `NSA(flowacc, k)`                                     | Network Scaling Algorithm. Returns `GridType.FlowDir`.                |
| `COTAT(flowdir, flowacc, k, *, area_threshold, mufp)` | COTAT / COTAT+. Returns `GridType.FlowDir`.                           |




### `fdup.utils`


| Function                                                                   | Description                                                                                                           |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `d8(dem, spherical=True)`                                                  | Compute ESRI D8 flow directions from a DEM.                                                                           |
| `flow_accumulation(flowdir, *, area=True)`                                 | Compute upstream flow accumulation (area in km² for geographic CRS, CRS units² for projected CRS, or raw cell count). |
| `strahler_order(flowdir)`                                                  | Compute Strahler stream orders via BFS. Returns `GridType.Strahler` (uint8; 0 = nodata/sink).                         |
| `snap_pour_cell(flowacc, x, y, radius)`                                    | Snap a pour point to the highest flow-accumulation cell within `radius`. Returns `(row, col)`.                        |
| `delineate_watershed(flowdir, pour_row, pour_col)`                         | BFS upstream delineation from a pour cell. Returns `GridType.Mask`.                                                   |
| `disaggregate_mask(mask, k)`                                               | Expand a coarse mask by factor `k` (nearest-neighbour).                                                               |
| `match_grids(reference, other)`                                            | Crop/pad `other` to the same extent as `reference`.                                                                   |
| `mask_area(mask)`                                                          | Total area of True-valued cells: km² for geographic CRS, CRS units² for projected CRS.                                |
| `threshold_mask(grid, cutoff)`                                             | Boolean mask where `value >= cutoff`; nodata/NaN cells → False. Accepts `FlowAcc` or `Strahler`.                      |
| `mask_grid(grid, mask)`                                                    | Set nodata on `grid` wherever `mask` is False. Returns a new Grid of the same type.                                   |
| `crop_grid(grid)`                                                          | Trim `grid` to the minimal bounding box of data cells; raises `ValueError` for all-nodata grids.                      |
| `river_tree(flowdir, flowacc, *, mask, min_upstream_area)`                 | Extract the river network as a `GridType.Tree` grid plus an array of seed coordinates.                                |
| `mask_seeds(seeds, flowdir, flowacc, mask)`                                | Prune a seeds array to mask-intersecting sub-segments; returns a new structured array with recomputed lengths.        |
| `vectorize_network(flowdir, flowacc=None)`                                 | Decompose D8 grid into river-segment LineStrings (GeoDataFrame). Requires `geopandas` and `shapely`.                  |
| `vectorize_tree(seeds, flowdir, *, cutoff=None, rank=None, accuracy=None)` | Trace each seed from headwater to mouth and return one LineString per seed (GeoDataFrame).                            |




### `fdup.evals`


| Function                                                                                            | Description                                                                    |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `compare_watersheds(mask1, mask2)`                                                                  | Squared Ochiai overlap index + intersection mask.                              |
| `compare_flowdir(flowdir_fine, flowdir_coarse, seeds, *, alpha, shuffle, strict_upstream, cascade)` | Per-seed flow-direction accuracy scores.                                       |
| `huac(flowacc_fine, flowdir_coarse, flowacc_coarse, x, y, radius, *, upstream_area_threshold)`      | Hiearchical upstream-area comparison: returns an error raster and a DataFrame. |
| `flowdir_windrose(flowdir, mask)`                                                                   | Direction-frequency windrose for a flow direction grid.                        |
| `windrose_emd(wr1, wr2)`                                                                            | Earth Mover's Distance between two windroses.                                  |




### Top-level names

```python
import fdup

fdup.warmup()            # pre-compile all numba kernels
fdup.Grid                # Grid value object
fdup.GridMeta            # GridMeta value object
fdup.GridType            # GridType enum (DEM, FlowDir, FlowAcc, Mask, Tree, Strahler)
```

---



## CLI reference

After installation the `fdup` command is available on the PATH.

```bash
fdup --help
```



### Upscaling

```bash
# DMM
fdup dmm --flowacc flowacc.tif -o flowdir_coarse.tif -k 4

# NSA
fdup nsa --flowacc flowacc.tif -o flowdir_coarse.tif -k 4

# COTAT
fdup cotat --flowdir flowdir.tif --flowacc flowacc.tif -o flowdir_coarse.tif -k 4 \
     --area-threshold 10

# COTAT+ (enable MUFP outlet selection; threshold in metres)
fdup cotat --flowdir flowdir.tif --flowacc flowacc.tif -o flowdir_coarse.tif -k 4 \
     --area-threshold 10 --mufp 5000
```



### Derivation utilities

```bash
# D8 flow directions from a DEM
fdup d8 --dem dem.tif -o flowdir.tif

# Flow accumulation
fdup flowacc --flowdir flowdir.tif -o flowacc.tif          # area in km² / CRS units²
fdup flowacc --flowdir flowdir.tif -o flowacc.tif --cells  # raw cell count

# Strahler stream order
fdup strahler --flowdir flowdir.tif -o strahler.tif

# Watershed delineation (snap to flowacc if --flowacc is provided)
fdup watershed --flowdir flowdir.tif --x -80.05 --y 35.05 --radius 5000 \
     -o watershed.tif --flowacc flowacc.tif
```



### Masking

```bash
# Boolean mask: cells with flow accumulation >= 500 km² (or Strahler order >= 4)
fdup threshold-mask --flowacc flowacc.tif --cutoff 500 -o streams.tif
fdup threshold-mask --strahler strahler.tif --cutoff 4 -o streams.tif

# Apply mask to another grid (sets nodata outside the mask)
fdup mask-grid --grid flowacc.tif --mask streams.tif -o flowacc_masked.tif

# Trim grid to the minimal bounding box of data cells
fdup crop-grid --grid flowacc.tif -o flowacc_cropped.tif
```



### River tree and vectorization

```bash
# River tree + seed export
fdup river-tree --flowdir flowdir.tif --flowacc flowacc.tif \
     --o-tree tree.tif --o-seeds seeds.npz

# Prune seeds to a spatial mask
fdup mask-seeds --seeds seeds.npz --flowdir flowdir.tif --flowacc flowacc.tif \
     --mask streams.tif -o seeds_masked.npz

# Vectorize full D8 network to GeoPackage
fdup vectorize-network --flowdir flowdir.tif -o network.gpkg
fdup vectorize-network --flowdir flowdir.tif --flowacc flowacc.tif -o network.gpkg

# Vectorize river-tree seeds to GeoPackage
fdup vectorize-tree --flowdir flowdir.tif --seeds seeds.npz -o tree.gpkg
fdup vectorize-tree --flowdir flowdir.tif --seeds seeds.npz --cutoff 1000 -o tree.gpkg
fdup vectorize-tree --flowdir flowdir.tif --seeds seeds.npz --rank 10 -o tree.gpkg
```



### Evaluation

```bash
# Ochiai overlap between two watershed masks
fdup compare-watersheds --mask1 ws1.tif --mask2 ws2.tif

# Per-seed flow-direction accuracy (seeds file: .npz with 'seeds' array)
fdup compare-flowdir --flowdir-fine fd_fine.tif --flowdir-coarse fd_coarse.tif \
     --seeds seeds.npz -o scores.npy

# HUAC upstream-area error
fdup huac --flowacc-fine fa_fine.tif --flowdir-coarse fd_coarse.tif \
     --flowacc-coarse fa_coarse.tif --x -80.05 --y 35.05 --radius 5000 \
     --o-raster huac_errors.tif --o-csv huac_errors.csv

```

---



## Caveats



### `snap_pour_cell`: radius is in CRS units

`radius` is deliberately a **Euclidean distance in the grid's CRS units**, not necessarily metres.
For geographic grids (e.g. EPSG:4326) the unit is **degrees**.
A radius of `0.5` on an EPSG:4326 grid means 0.5 degrees, not 0.5 km.
To use metric radii, reproject the grid to a projected CRS first.

### Area units depend on CRS type

`mask_area` and `flow_accumulation(..., area=True)` return areas in different
units depending on the CRS:

- **Geographic CRS** (lat/lon, e.g. EPSG:4326): areas are in **km²**, computed
via the spherical-trapezoid formula, so cell area varies by latitude.
- **Projected CRS** (e.g. UTM): areas are in **CRS units²**.  For a
metre-based CRS this is **m²**; for a foot-based CRS this is **ft²**.
Callers are responsible for interpreting the unit based on their CRS.
Reproject to a metre-based CRS (e.g. a UTM zone) if m² output is required.



### COTAT+ MUFP is in metres

The `mufp` parameter of `COTAT()` (and `--mufp` on the CLI) is always in **metres**,
regardless of the grid's CRS.  The internal distance accumulation uses the
spherical-distance formula for geographic grids and the Euclidean metric
(scaled by the CRS unit) for projected grids.

### Float64 flow accumulation precision above 2⁵³

When flow accumulation is computed in `float64`, the internal numba kernels
accumulate values as 64-bit floats.  Values above `2**53` (~9 × 10¹⁵) cannot be
represented exactly, and cell-count precision is lost.  For realistic grid sizes
this is not a concern, but very large global grids at fine resolution may be
affected.  Use `uint64` accumulation (cell counts only) if exact integer results
are required.

---



## Migration guide from v0.1

The object-oriented `BaseUpscaler` surface has been removed.  Replace the old
call pattern:

```python
# old (v0.1)
from fdup.upscalers import DMM
dmm = DMM()
dmm.load_flowacc("flowacc.tif")
dmm.upscale(k=4)
dmm.save("flowdir_coarse.tif")
```

with the new functional API:

```python
# new (v0.2+)
import fdup
fd = fdup.upscalers.DMM(fdup.io.read("flowacc.tif", grid_type=fdup.GridType.FlowAcc), k=4)
fdup.io.write(fd, "flowdir_coarse.tif", overwrite=True)
```

The same pattern applies to `NSA` and `COTAT`.

---



## Input data conventions

Flow direction grids use the [ESRI D8 encoding](https://pro.arcgis.com/en/pro-app/3.4/tool-reference/spatial-analyst/how-flow-direction-works.htm):
`1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE, 255=nodata`.

Supported flow accumulation dtypes: `int32`, `uint32`, `int64`, `uint64`, `float32`, `float64`.