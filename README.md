# fdup

`fdup` is a [numba](https://numba.pydata.org/)-accelerated Python toolkit containing algorithms for upscaling D8 flow direction grids. 

## Currently implemented algorithms

- **DMM** (Double Maximum Method) by [Olivera et al., 2002](https://doi.org/10.1029/2001WR000726)
- **NSA** (Network Scaling Algorithm) by [Fekete et al., 2001](https://doi.org/10.1029/2001WR900024)
- **COTAT** (Cell Outlet Tracing with an Area Threshold) by [Reed, 2003](https://doi.org/10.1029/2003WR001989)

## Installation

From the repository root:

```bash
pip install .
```

Requirements:

- Python `>=3.10`
- `numpy >= 1.21.3`
- `numba >= 0.55.0`
- `rasterio >= 1.3.0`

## Python API usage

```python
# Import call can take 10-15s since all @njit functions are compiled at import time
from fdup.upscalers import DMM, NSA, COTAT

# DMM
dmm = DMM()
dmm.load_flowacc("flowacc.tif")
dmm.upscale(k=20)
dmm.save("out_dmm.tif")

# NSA
nsa = NSA()
nsa.load_flowacc("flowacc.tif")
nsa.upscale(k=20)
nsa.save("out_nsa.tif")

# COTAT
cotat = COTAT()
cotat.load_flowdir("flowdir.tif")
cotat.load_flowacc("flowacc.tif")
cotat.upscale(k=20, area_threshold=10)
cotat.save("out_cotat.tif")
```

## CLI

Besides Python API, `fdup` features a command-line interface:

```bash
fdup --help
```

#### DMM

```bash
fdup dmm --flowacc flowacc.tif -o out_dmm.tif -k 20
```

#### NSA

```bash
fdup nsa --flowacc flowacc.tif -o out_nsa.tif -k 20
```

#### COTAT

```bash
fdup cotat --flowdir flowdir.tif --flowacc flowacc.tif -o out_cotat.tif -k 20 --area-threshold 10
```

## Parameters

- `k`: positive integer scaling factor for resulting grid. Output grid cells will be `k` times larger than input ones. Note: for DMM, `k` should be an even number.

- `area-threshold`/`area_threshold`: tracing area threshold (for COTAT only). Defines when to stop tracing original fine-resolution flow directions: larger values of threshold tend to increase the number of diagonal flow directions.

## Input data conventions

The tool works with [ESRI-style](https://pro.arcgis.com/en/pro-app/3.4/tool-reference/spatial-analyst/how-flow-direction-works.htm) D8 flow direction encoding for input as well as output flow directions. 

For input flow accumulation, the supported data types are `uint32`, `float32`, `float64`/`double`