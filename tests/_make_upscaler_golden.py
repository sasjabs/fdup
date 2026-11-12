"""Build ``tests/data/upscaler_golden.npz`` from the current upscalers.

The fixture stores both inputs and outputs so later changes to ``d8`` or
``flow_accumulation`` cannot move the golden results. Regenerate only when
the upscaler contract itself is supposed to change:

    python -m tests._make_upscaler_golden

Inputs are a seeded, smoothed DEM passed through ``d8`` (flat-Earth) and
``flow_accumulation``, which yields an acyclic D8 field. Random direction
codes are not used: a cycle can hang COTAT's downstream trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.upscalers import COTAT, DMM, NSA
from fdup.utils.d8 import d8
from fdup.utils.flowacc import flow_accumulation

# 0.01° pixels, origin at (0°E, 10°N). CRS is left unset, which the library
# treats as geographic, so COTAT+ path lengths are real metre distances.
FINE_TRANSFORM = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)

_OUTPUT = Path(__file__).resolve().parent / "data" / "upscaler_golden.npz"

_MUFP_CANDIDATES = (
    1.0,
    50.0,
    200.0,
    800.0,
    1_500.0,
    4_000.0,
    10_000.0,
    40_000.0,
    200_000.0,
)

# (name, shape, (kx, ky), kind, hole, disagree, mufp, area_threshold)
# kind is the flow-accumulation variant. mufp / area_threshold are either a
# number, None, or "search" (pick a value that actually changes COTAT).
_SPECS: tuple[dict, ...] = (
    dict(name="s37x53_k2_f32_nan", shape=(37, 53), k=(2, 2),
         kind="f32", hole=True, disagree=False, mufp=None, area_threshold=0.0),
    dict(name="s64x64_k4_u32_sentinel", shape=(64, 64), k=(4, 4),
         kind="u32", hole=True, disagree=False, mufp=None, area_threshold=0.0),
    dict(name="s41x30_k4x2_f64_none", shape=(41, 30), k=(4, 2),
         kind="f64_none", hole=False, disagree=False, mufp=None, area_threshold=0.0),
    dict(name="s37x53_k2x6_i64_sentinel", shape=(37, 53), k=(2, 6),
         kind="i64", hole=True, disagree=False, mufp="search", area_threshold=0.0),
    dict(name="s41x30_k5_f32_nan", shape=(41, 30), k=(5, 5),
         kind="f32", hole=True, disagree=False, mufp=None, area_threshold="search"),
    dict(name="s37x53_k2_disagree_f32", shape=(37, 53), k=(2, 2),
         kind="f32", hole=True, disagree=True, mufp="search", area_threshold="search"),
    dict(name="s64x64_k4_disagree_i64", shape=(64, 64), k=(4, 4),
         kind="i64", hole=True, disagree=True, mufp=None, area_threshold=0.0),
)

_PAIR_CACHE: dict[tuple, tuple[Grid, Grid]] = {}


def _smooth(noise: np.ndarray, passes: int = 4) -> np.ndarray:
    out = noise.astype(np.float64, copy=True)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return out


def _dem_array(shape: tuple[int, int], seed: int, *, hole: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = _smooth(rng.random(shape))
    rows = np.linspace(0.0, 1.0, shape[0], dtype=np.float64)[:, None]
    cols = np.linspace(0.0, 1.0, shape[1], dtype=np.float64)[None, :]
    # Tilt dominates the noise so the network drains, but not so strongly
    # that every pixel shares one D8 code.
    dem = (0.55 * noise + 0.9 * (1.0 - rows) + 0.7 * (1.0 - cols)).astype(np.float32)
    if hole:
        dem[4:12, 6:16] = np.nan
    return dem


def _clone(grid: Grid) -> Grid:
    return Grid.create(
        array=grid.array.copy(),
        type=grid.meta.type,
        transform=grid.meta.transform,
        crs=grid.meta.crs,
        nodata=grid.meta.nodata,
    )


def make_flow_pair(
    shape: tuple[int, int],
    *,
    seed: int,
    hole: bool,
    area: bool,
) -> tuple[Grid, Grid]:
    """Acyclic fine flowdir plus flow accumulation for ``shape``.

    ``area=True`` matches ``flow_accumulation``'s default (float32, NaN
    nodata). ``area=False`` is an upstream-cell count (uint32, integer
    sentinel). Returned grids are copies and may be mutated by the caller.
    """
    key = (shape, seed, hole, area)
    cached = _PAIR_CACHE.get(key)
    if cached is None:
        dem = Grid.create(
            array=_dem_array(shape, seed, hole=hole),
            type=GridType.DEM,
            transform=FINE_TRANSFORM,
            crs=None,
            nodata=float("nan"),
        )
        flowdir = d8(dem, spherical=False)
        flowacc = flow_accumulation(flowdir, area=area)
        cached = (flowdir, flowacc)
        _PAIR_CACHE[key] = cached
    return _clone(cached[0]), _clone(cached[1])


def _seed_for(shape: tuple[int, int]) -> int:
    return 20260312 + shape[0] * 1_000 + shape[1]


def _as_int64(flowacc: Grid) -> tuple[np.ndarray, int]:
    """Cell counts as int64, with the uint32 sentinel remapped to int64 max."""
    src = flowacc.array
    src_sentinel = np.iinfo(np.uint32).max
    sentinel = int(np.iinfo(np.int64).max)
    out = src.astype(np.int64)
    out[src == src_sentinel] = sentinel
    return out, sentinel


def _as_float64_none(flowacc: Grid) -> tuple[np.ndarray, None]:
    """Finite float64 accumulation and ``nodata=None`` (NaNs become 0)."""
    out = flowacc.array.astype(np.float64)
    out[np.isnan(out)] = 0.0
    return out, None


def _punch_disagreement(
    flowdir: np.ndarray,
    flowacc: np.ndarray,
    nodata,
) -> tuple[np.ndarray, np.ndarray]:
    """Nodata patches that do not match between flowdir and flowacc.

    ``[14:18, 2:8]`` is flowdir-only nodata. ``[20:26, 20:28]`` is
    flowacc-only nodata. The DEM hole, when present, stays agreed nodata.
    """
    flowdir = flowdir.copy()
    flowacc = flowacc.copy()
    flowdir[14:18, 2:8] = np.uint8(255)
    if isinstance(nodata, float) and np.isnan(nodata):
        flowacc[20:26, 20:28] = np.nan
    else:
        flowacc[20:26, 20:28] = np.array(nodata, dtype=flowacc.dtype)
    return flowdir, flowacc


def _nodata_mask(flowacc: np.ndarray, nodata) -> np.ndarray:
    if nodata is None:
        return np.zeros(flowacc.shape, dtype=bool)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isnan(flowacc)
    return flowacc == nodata


def _materialize(spec: dict) -> tuple[np.ndarray, np.ndarray, object]:
    shape = spec["shape"]
    seed = _seed_for(shape)
    kind = spec["kind"]
    if kind == "f32":
        flowdir, flowacc = make_flow_pair(shape, seed=seed, hole=spec["hole"], area=True)
        fa = flowacc.array.copy()
        nodata: object = float("nan")
    elif kind == "u32":
        flowdir, flowacc = make_flow_pair(shape, seed=seed, hole=spec["hole"], area=False)
        fa = flowacc.array.copy()
        nodata = int(flowacc.meta.nodata)
    elif kind == "i64":
        flowdir, flowacc = make_flow_pair(shape, seed=seed, hole=spec["hole"], area=False)
        fa, nodata = _as_int64(flowacc)
    elif kind == "f64_none":
        flowdir, flowacc = make_flow_pair(shape, seed=seed, hole=spec["hole"], area=True)
        fa, nodata = _as_float64_none(flowacc)
    else:
        raise AssertionError(f"unknown kind {kind!r}")

    fd = flowdir.array.copy()
    if spec["disagree"]:
        fd, fa = _punch_disagreement(fd, fa, nodata)
        fd_nd = fd == np.uint8(255)
        fa_nd = _nodata_mask(fa, nodata)
        if not ((fd_nd & ~fa_nd).any() and (fa_nd & ~fd_nd).any()):
            raise RuntimeError(f"{spec['name']}: disagreement patches do not overlap differently")
    elif spec["hole"]:
        fd_nd = fd == np.uint8(255)
        fa_nd = _nodata_mask(fa, nodata)
        if not np.array_equal(fd_nd, fa_nd):
            raise RuntimeError(f"{spec['name']}: expected flowdir and flowacc nodata to agree")
    return fd, fa, nodata


def _grids(fd: np.ndarray, fa: np.ndarray, nodata) -> tuple[Grid, Grid]:
    flowdir = Grid.create(
        array=fd,
        type=GridType.FlowDir,
        transform=FINE_TRANSFORM,
        crs=None,
    )
    flowacc = Grid.create(
        array=fa,
        type=GridType.FlowAcc,
        transform=FINE_TRANSFORM,
        crs=None,
        nodata=nodata,
    )
    return flowdir, flowacc


def _k_arg(k: tuple[int, int]) -> int | tuple[int, int]:
    kx, ky = k
    return kx if kx == ky else (kx, ky)


def _algorithms(k: tuple[int, int]) -> list[str]:
    names = ["NSA", "COTAT"]
    kx, ky = k
    if kx % 2 == 0 and ky % 2 == 0:
        names.append("DMM")
    return names


def _run(name: str, flowdir: Grid, flowacc: Grid, k: tuple[int, int],
         area_threshold: float, mufp: float | None) -> Grid:
    k_arg = _k_arg(k)
    if name == "NSA":
        return NSA(flowacc, k_arg)
    if name == "DMM":
        return DMM(flowacc, k_arg)
    if name == "COTAT":
        return COTAT(
            flowdir, flowacc, k_arg,
            area_threshold=area_threshold,
            mufp=mufp,
        )
    raise AssertionError(name)


def _coeffs(transform: Affine) -> np.ndarray:
    return np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype=np.float64,
    )


def _threshold_candidates(flowacc: np.ndarray, nodata) -> list[float]:
    valid = flowacc[~_nodata_mask(flowacc, nodata)]
    finite = valid[np.isfinite(valid)] if np.issubdtype(valid.dtype, np.floating) else valid
    vmax = float(np.max(finite)) if finite.size else 1.0
    candidates: list[float] = []
    value = vmax
    while value > 1e-3 and len(candidates) < 16:
        candidates.append(float(value))
        value *= 0.5
    return candidates


def _closest_change(
    baseline: np.ndarray,
    candidates: list[float],
    run,
) -> float:
    """Candidate whose COTAT result moves, preferring ~25% of cells changed."""
    target = max(1, baseline.size // 4)
    best: tuple[int, float] | None = None
    for value in candidates:
        diff = int(np.count_nonzero(run(value).array != baseline))
        if diff == 0:
            continue
        distance = abs(diff - target)
        if best is None or distance < best[0]:
            best = (distance, float(value))
    if best is None:
        raise RuntimeError("no candidate changed the COTAT result")
    return best[1]


def _resolve_cotat_params(
    spec: dict,
    flowdir: Grid,
    flowacc: Grid,
) -> tuple[float | None, float]:
    k = spec["k"]
    mufp = spec["mufp"]
    area_threshold = spec["area_threshold"]

    def cotat(mufp_val: float | None, threshold: float) -> Grid:
        return _run("COTAT", flowdir, flowacc, k, threshold, mufp_val)

    if mufp == "search" and area_threshold == "search":
        plain = cotat(None, 0.0)
        thresholds = _threshold_candidates(flowacc.array, flowacc.meta.nodata)
        found = None
        for candidate_mufp in _MUFP_CANDIDATES:
            with_mufp = cotat(candidate_mufp, 0.0)
            if np.array_equal(with_mufp.array, plain.array):
                continue
            for candidate_threshold in thresholds:
                both = cotat(candidate_mufp, candidate_threshold)
                only_threshold = cotat(None, candidate_threshold)
                mufp_matters = not np.array_equal(both.array, only_threshold.array)
                threshold_matters = not np.array_equal(both.array, with_mufp.array)
                if mufp_matters and threshold_matters:
                    found = (float(candidate_mufp), float(candidate_threshold))
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"{spec['name']}: no mufp/area_threshold pair changes COTAT")
        return found

    if mufp == "search":
        baseline = cotat(None, float(area_threshold))
        mufp = _closest_change(
            baseline.array,
            list(_MUFP_CANDIDATES),
            lambda value: cotat(value, float(area_threshold)),
        )
    if area_threshold == "search":
        baseline = cotat(mufp, 0.0)
        area_threshold = _closest_change(
            baseline.array,
            _threshold_candidates(flowacc.array, flowacc.meta.nodata),
            lambda value: cotat(mufp, value),
        )
    return mufp, float(area_threshold)


def _nodata_token(nodata) -> str:
    if nodata is None:
        return "none"
    if isinstance(nodata, float) and np.isnan(nodata):
        return "nan"
    return str(int(nodata))


def build_fixture(path: Path = _OUTPUT) -> dict:
    """Run every spec on the current code and write inputs plus outputs."""
    manifest_cases: list[dict] = []
    payload: dict[str, np.ndarray] = {}

    for spec in _SPECS:
        fd_arr, fa_arr, nodata = _materialize(spec)
        flowdir, flowacc = _grids(fd_arr, fa_arr, nodata)
        mufp, area_threshold = _resolve_cotat_params(spec, flowdir, flowacc)
        algorithms = _algorithms(spec["k"])

        if spec["disagree"]:
            # The flowacc-only patch must be visible to NSA, which never reads flowdir.
            fd_raw, fa_raw, _nodata_raw = _materialize({**spec, "disagree": False})
            raw_dir, raw_acc = _grids(fd_raw, fa_raw, _nodata_raw)
            punched_nsa = _run("NSA", flowdir, flowacc, spec["k"], area_threshold, mufp)
            raw_nsa = _run("NSA", raw_dir, raw_acc, spec["k"], area_threshold, mufp)
            if np.array_equal(punched_nsa.array, raw_nsa.array):
                raise RuntimeError(f"{spec['name']}: flowacc nodata patch did not affect NSA")

        case = {
            "name": spec["name"],
            "shape": list(spec["shape"]),
            "k": list(spec["k"]),
            "fa_dtype": str(np.dtype(flowacc.array.dtype)),
            "nodata": _nodata_token(flowacc.meta.nodata),
            "disagree": bool(spec["disagree"]),
            "mufp": mufp,
            "area_threshold": area_threshold,
            "algorithms": algorithms,
        }
        manifest_cases.append(case)
        payload[f"{spec['name']}__fd"] = np.ascontiguousarray(flowdir.array)
        payload[f"{spec['name']}__fa"] = np.ascontiguousarray(flowacc.array)

        for algorithm in algorithms:
            out = _run(algorithm, flowdir, flowacc, spec["k"], area_threshold, mufp)
            if out.array.dtype != np.uint8 or out.meta.type != GridType.FlowDir:
                raise RuntimeError(f"{spec['name']} {algorithm}: unexpected output grid")
            payload[f"{spec['name']}__{algorithm}"] = np.ascontiguousarray(out.array)
            payload[f"{spec['name']}__{algorithm}_transform"] = _coeffs(out.meta.transform)

    manifest = {
        "version": 1,
        "transform": _coeffs(FINE_TRANSFORM).tolist(),
        "cases": manifest_cases,
    }
    payload["manifest"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"),
        dtype=np.uint8,
    ).copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return manifest


def main() -> None:
    manifest = build_fixture()
    print(f"wrote {_OUTPUT}")
    for case in manifest["cases"]:
        mufp = "None" if case["mufp"] is None else case["mufp"]
        algos = ",".join(case["algorithms"])
        print(
            f"  {case['name']}: {algos} nodata={case['nodata']} "
            f"mufp={mufp} area_threshold={case['area_threshold']}"
        )


if __name__ == "__main__":
    main()
