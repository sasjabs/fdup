"""Golden-output and peak-memory checks for the flow-direction upscalers.

``tests/data/upscaler_golden.npz`` is produced by
``tests/_make_upscaler_golden.py`` against unchanged upscaler code. These
tests rerun NSA, COTAT and DMM on the stored inputs and require bit-identical
arrays and transforms. A separate tracemalloc bound guards the copy-heavy
wrappers; tighten the ratios as those copies are removed.
"""

from __future__ import annotations

import gc
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.upscalers import COTAT, DMM, NSA
from tests._make_upscaler_golden import make_flow_pair

_NPZ = Path(__file__).resolve().parent / "data" / "upscaler_golden.npz"

# Peak traced bytes / nbytes of the arrays passed into the call.
# Median of 3 warmed-up runs on a 192×256 float32 flowacc (uint8 flowdir
# for COTAT), k=4. tracemalloc does not include the input arrays, which
# were allocated before tracing; adding them back matches the working-set
# estimates in the cleanup plan (~4 / ~6 / ~11 bytes per fine pixel).
#
#   NSA    0.11×   ~0.43 B/px traced  (coarse outputs only; ~4.4 B/px with input)
#   COTAT  0.19×   ~0.94 B/px traced  (row-distance table + coarse outputs;
#                                      ~5.9 B/px with input)
#   DMM    0.28×   ~1.1  B/px traced  (one unpadded bool mask; ~5.1 B/px with input)
#
# Ceilings sit just high enough to absorb allocator noise and still fail
# if a full-size extra copy appears. Lower them as each copy-elimination
# step lands. COTAT's 0.35 fails a fine-grid uint8 or bool array (~0.2×)
# and a float32 accumulation copy (~0.8×).
# DMM's 0.50 fails an extra fine-grid bool or uint8 array (~0.25×)
# and the removed padded float32 copy (~1.0×).
_NSA_PEAK_RATIO = 0.6
_COTAT_PEAK_RATIO = 0.35
_DMM_PEAK_RATIO = 0.50

_MEMORY_SHAPE = (192, 256)
_MEMORY_K = 4


def _load():
    if not _NPZ.is_file():
        raise FileNotFoundError(
            f"Missing {_NPZ}. Generate it with: python -m tests._make_upscaler_golden"
        )
    with np.load(_NPZ) as archive:
        manifest = json.loads(bytes(archive["manifest"]).decode("utf-8"))
        arrays = {
            key: archive[key].copy()
            for key in archive.files
            if key != "manifest"
        }
    return manifest, arrays


_MANIFEST, _ARRAYS = _load()
_CASES = {case["name"]: case for case in _MANIFEST["cases"]}


def _nodata_from_token(token: str):
    if token == "none":
        return None
    if token == "nan":
        return float("nan")
    return int(token)


def _transform_from_coeffs(coeffs) -> Affine:
    a, b, c, d, e, f = (float(v) for v in coeffs)
    return Affine(a, b, c, d, e, f)


def _grids(case: dict) -> tuple[Grid, Grid]:
    name = case["name"]
    transform = _transform_from_coeffs(_MANIFEST["transform"])
    nodata = _nodata_from_token(case["nodata"])
    # Copy before Grid.create. FlowAcc keeps the caller's array, and a later
    # copy=False change on FlowDir would too; the fixture buffers must stay
    # pristine for the rest of the session.
    flowdir = Grid.create(
        array=_ARRAYS[f"{name}__fd"].copy(),
        type=GridType.FlowDir,
        transform=transform,
        crs=None,
    )
    flowacc = Grid.create(
        array=_ARRAYS[f"{name}__fa"].copy(),
        type=GridType.FlowAcc,
        transform=transform,
        crs=None,
        nodata=nodata,
    )
    return flowdir, flowacc


def _k_arg(case: dict) -> int | tuple[int, int]:
    kx, ky = case["k"]
    return kx if kx == ky else (kx, ky)


def _run(algorithm: str, flowdir: Grid, flowacc: Grid, case: dict) -> Grid:
    k = _k_arg(case)
    if algorithm == "NSA":
        return NSA(flowacc, k)
    if algorithm == "DMM":
        return DMM(flowacc, k)
    if algorithm == "COTAT":
        mufp = case["mufp"]
        return COTAT(
            flowdir,
            flowacc,
            k,
            area_threshold=float(case["area_threshold"]),
            mufp=None if mufp is None else float(mufp),
        )
    raise AssertionError(algorithm)


def _case_params():
    params = []
    for case in _MANIFEST["cases"]:
        for algorithm in case["algorithms"]:
            params.append(pytest.param(case["name"], algorithm, id=f"{case['name']}-{algorithm}"))
    return params


def test_golden_fixture_covers_plan_matrix():
    """The stored cases hit every shape, k, dtype and nodata variant in Step 0."""
    cases = _MANIFEST["cases"]
    shapes = {tuple(case["shape"]) for case in cases}
    assert shapes >= {(37, 53), (64, 64), (41, 30)}

    ks = {tuple(case["k"]) for case in cases}
    assert ks >= {(2, 2), (4, 4), (4, 2), (2, 6), (5, 5)}

    dtypes = {case["fa_dtype"] for case in cases}
    assert dtypes >= {"uint32", "int64", "float32", "float64"}

    nodata = {case["nodata"] for case in cases}
    assert "none" in nodata
    assert "nan" in nodata
    assert any(token not in {"none", "nan"} for token in nodata)

    assert any(case["disagree"] and case["nodata"] == "nan" for case in cases)
    assert any(case["disagree"] and case["nodata"] not in {"none", "nan"} for case in cases)

    assert any(case["mufp"] is None for case in cases)
    assert any(case["mufp"] is not None and case["mufp"] > 0 for case in cases)
    assert any(case["area_threshold"] == 0 for case in cases)
    assert any(case["area_threshold"] > 0 for case in cases)

    odd = [case for case in cases if any(axis % 2 for axis in case["k"])]
    assert odd and all("DMM" not in case["algorithms"] for case in odd)
    assert any("DMM" in case["algorithms"] for case in cases)
    assert all({"NSA", "COTAT"} <= set(case["algorithms"]) for case in cases)

    for case in cases:
        name = case["name"]
        shape = tuple(case["shape"])
        assert _ARRAYS[f"{name}__fd"].shape == shape
        assert _ARRAYS[f"{name}__fa"].shape == shape
        assert str(_ARRAYS[f"{name}__fa"].dtype) == case["fa_dtype"]
        for algorithm in case["algorithms"]:
            out = _ARRAYS[f"{name}__{algorithm}"]
            assert out.dtype == np.uint8
            assert out.ndim == 2 and out.size > 0
            assert np.any(out != np.uint8(255)), f"{name} {algorithm} is entirely nodata"


@pytest.mark.parametrize("name,algorithm", _case_params())
def test_upscaler_matches_golden(name: str, algorithm: str):
    case = _CASES[name]
    flowdir, flowacc = _grids(case)
    fd_before = flowdir.array.copy()
    fa_before = flowacc.array.copy()

    out = _run(algorithm, flowdir, flowacc, case)

    np.testing.assert_array_equal(flowdir.array, fd_before)
    np.testing.assert_array_equal(flowacc.array, fa_before)
    np.testing.assert_array_equal(out.array, _ARRAYS[f"{name}__{algorithm}"])
    np.testing.assert_array_equal(
        np.array(
            [out.meta.transform.a, out.meta.transform.b, out.meta.transform.c,
             out.meta.transform.d, out.meta.transform.e, out.meta.transform.f],
            dtype=np.float64,
        ),
        _ARRAYS[f"{name}__{algorithm}_transform"],
    )
    assert out.meta.type == GridType.FlowDir
    assert out.array.dtype == np.uint8
    assert out.meta.nodata == 255
    assert out.meta.crs is None


def _peak_bytes(fn) -> int:
    """Median peak of three warmed-up calls. NumPy allocations are traced."""
    peaks = []
    for _ in range(3):
        gc.collect()
        tracemalloc.start()
        try:
            fn()
        finally:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        peaks.append(peak)
    peaks.sort()
    return peaks[1]


def test_upscaler_peak_memory_ratio():
    """Peak tracemalloc bytes stay within the per-upscaler input-size budget.

    The grid is larger than the golden cases so fixed Python overhead does
    not dominate the ratio. JIT compilation is warmed up before tracing.
    """
    flowdir, flowacc = make_flow_pair(
        _MEMORY_SHAPE, seed=7, hole=True, area=True,
    )
    NSA(flowacc, _MEMORY_K)
    DMM(flowacc, _MEMORY_K)
    COTAT(flowdir, flowacc, _MEMORY_K)

    nsa_peak = _peak_bytes(lambda: NSA(flowacc, _MEMORY_K))
    dmm_peak = _peak_bytes(lambda: DMM(flowacc, _MEMORY_K))
    cotat_peak = _peak_bytes(lambda: COTAT(flowdir, flowacc, _MEMORY_K))

    nsa_in = flowacc.array.nbytes
    dmm_in = flowacc.array.nbytes
    cotat_in = flowdir.array.nbytes + flowacc.array.nbytes
    nsa_ratio = nsa_peak / nsa_in
    dmm_ratio = dmm_peak / dmm_in
    cotat_ratio = cotat_peak / cotat_in

    assert nsa_ratio < _NSA_PEAK_RATIO, (
        f"NSA peak/input {nsa_ratio:.3f} exceeds {_NSA_PEAK_RATIO} "
        f"(peak={nsa_peak}, input={nsa_in})"
    )
    assert dmm_ratio < _DMM_PEAK_RATIO, (
        f"DMM peak/input {dmm_ratio:.3f} exceeds {_DMM_PEAK_RATIO} "
        f"(peak={dmm_peak}, input={dmm_in})"
    )
    assert cotat_ratio < _COTAT_PEAK_RATIO, (
        f"COTAT peak/input {cotat_ratio:.3f} exceeds {_COTAT_PEAK_RATIO} "
        f"(peak={cotat_peak}, input={cotat_in})"
    )
