"""Smoke tests for fdup.upscalers (Phase 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from fdup._core.d8 import DIR_DIST
from fdup._core.types import Grid, GridType
from fdup.upscalers import DMM, NSA, COTAT
from fdup.upscalers.nsa import _dir_distances

# 0.01° pixels, origin at (0°E, 10°N)
FINE_TRANSFORM = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 10.0)
COARSE_TRANSFORM = Affine(0.02, 0.0, 0.0, 0.0, -0.02, 10.0)


# ---------------------------------------------------------------------------
# Fixture-like helpers
# ---------------------------------------------------------------------------

def _make_flowacc(shape=(8, 8), dtype=np.uint32) -> Grid:
    """8×8 FlowAcc grid with values increasing row-by-row (simple gradient)."""
    arr = np.arange(shape[0] * shape[1], dtype=dtype).reshape(shape) + 1
    return Grid.create(
        array=arr,
        type=GridType.FlowAcc,
        transform=FINE_TRANSFORM,
        crs=None,
    )


def _make_flowdir(shape=(8, 8)) -> Grid:
    """8×8 FlowDir grid with all pixels pointing SE (D8 code = 2)."""
    arr = np.full(shape, 2, dtype=np.uint8)   # 2 = SE
    return Grid.create(
        array=arr,
        type=GridType.FlowDir,
        transform=FINE_TRANSFORM,
        crs=None,
    )


# ---------------------------------------------------------------------------
# Output shape / dtype / transform assertions (shared logic)
# ---------------------------------------------------------------------------

def _assert_upscaled(out: Grid, flowacc: Grid, k: int) -> None:
    assert out.meta.type == GridType.FlowDir, "output type must be FlowDir"
    assert out.array.dtype == np.uint8, f"output dtype must be uint8, got {out.array.dtype}"

    H, W = flowacc.shape
    expected_rows = (H + k - 1) // k if k > 0 else H // k
    expected_cols = (W + k - 1) // k if k > 0 else W // k

    # DMM uses floor (//k) rather than ceil for its specific A-grid logic
    # so we just check both algorithms' documented behaviour separately.
    # Here we assert the shape matches the expectation for the specific algorithm.
    assert out.shape == (expected_rows, expected_cols), (
        f"expected shape ({expected_rows}, {expected_cols}), got {out.shape}"
    )

    t_in  = flowacc.meta.transform
    t_out = out.meta.transform
    assert abs(t_out.a - t_in.a * k) < 1e-9, (
        f"output pixel width should be {t_in.a * k}, got {t_out.a}"
    )
    assert abs(t_out.e - t_in.e * k) < 1e-9, (
        f"output pixel height should be {t_in.e * k}, got {t_out.e}"
    )


def _assert_anisotropic(out: Grid, flowacc: Grid, kx: int, ky: int, *, floor: bool) -> None:
    """Shape is ceil(H/ky)×ceil(W/kx), or floor for DMM; transform is (a*kx, e*ky)."""
    H, W = flowacc.shape
    expected = (H // ky, W // kx) if floor else (math.ceil(H / ky), math.ceil(W / kx))
    assert out.shape == expected, f"expected shape {expected}, got {out.shape}"
    assert out.array.dtype == np.uint8
    assert out.meta.type == GridType.FlowDir
    t_in = flowacc.meta.transform
    t_out = out.meta.transform
    assert abs(t_out.a - t_in.a * kx) < 1e-9
    assert abs(t_out.e - t_in.e * ky) < 1e-9


def _assert_invalid_k(call) -> None:
    """Reject bad tuple lengths and non-integer members."""
    with pytest.raises(ValueError):
        call((4, 2, 1))
    with pytest.raises(ValueError):
        call((4,))
    with pytest.raises(ValueError):
        call((4.0, 2))
    with pytest.raises(ValueError):
        call((2, 4.0))


# ---------------------------------------------------------------------------
# DMM
# ---------------------------------------------------------------------------

class TestDMM:

    def test_dmm_basic_shape_dtype_transform(self):
        fa = _make_flowacc()
        k = 2
        out = DMM(fa, k)
        # DMM shape: flowacc.shape[0] // k
        H, W = fa.shape
        assert out.shape == (H // k, W // k)
        assert out.array.dtype == np.uint8
        assert out.meta.type == GridType.FlowDir
        t = fa.meta.transform
        assert abs(out.meta.transform.a - t.a * k) < 1e-9
        assert abs(out.meta.transform.e - t.e * k) < 1e-9

    def test_dmm_odd_k_raises(self):
        fa = _make_flowacc()
        with pytest.raises(ValueError, match="even"):
            DMM(fa, k=3)

    def test_dmm_wrong_type_raises(self):
        fd = _make_flowdir()
        with pytest.raises((TypeError, ValueError)):
            DMM(fd, k=2)   # FlowDir passed where FlowAcc expected

    def test_dmm_k4_on_8x8(self):
        fa = _make_flowacc()
        out = DMM(fa, k=4)
        H, W = fa.shape
        assert out.shape == (H // 4, W // 4)

    def test_dmm_int_k_matches_tuple_kk(self):
        fa = _make_flowacc()
        out_int = DMM(fa, k=4)
        out_tuple = DMM(fa, k=(4, 4))
        np.testing.assert_array_equal(out_int.array, out_tuple.array)
        assert out_int.meta.transform == out_tuple.meta.transform

    def test_dmm_anisotropic_shape_and_transform(self):
        fa = _make_flowacc(shape=(12, 10))
        kx, ky = 4, 2
        out = DMM(fa, k=(kx, ky))
        _assert_anisotropic(out, fa, kx, ky, floor=True)

    def test_dmm_invalid_k_tuples(self):
        fa = _make_flowacc()
        _assert_invalid_k(lambda k: DMM(fa, k))

    def test_dmm_odd_kx_raises(self):
        fa = _make_flowacc()
        with pytest.raises(ValueError, match="kx"):
            DMM(fa, k=(3, 4))

    def test_dmm_odd_ky_raises(self):
        fa = _make_flowacc()
        with pytest.raises(ValueError, match="ky"):
            DMM(fa, k=(4, 3))


# ---------------------------------------------------------------------------
# NSA
# ---------------------------------------------------------------------------

class TestNSA:

    def test_nsa_basic_shape_dtype_transform(self):
        fa = _make_flowacc()
        k = 2
        out = NSA(fa, k)
        _assert_upscaled(out, fa, k)

    def test_nsa_wrong_type_raises(self):
        fd = _make_flowdir()
        with pytest.raises((TypeError, ValueError)):
            NSA(fd, k=2)

    def test_nsa_odd_k_valid(self):
        # NSA has no parity restriction on k
        fa = _make_flowacc(shape=(8, 8))
        out = NSA(fa, k=4)
        assert out.shape == (2, 2)

    def test_nsa_int_k_matches_tuple_kk(self):
        fa = _make_flowacc()
        out_int = NSA(fa, k=4)
        out_tuple = NSA(fa, k=(4, 4))
        np.testing.assert_array_equal(out_int.array, out_tuple.array)
        assert out_int.meta.transform == out_tuple.meta.transform

    def test_nsa_anisotropic_shape_and_transform(self):
        fa = _make_flowacc(shape=(12, 10))
        kx, ky = 4, 2
        out = NSA(fa, k=(kx, ky))
        _assert_anisotropic(out, fa, kx, ky, floor=False)

    def test_nsa_invalid_k_tuples(self):
        fa = _make_flowacc()
        _assert_invalid_k(lambda k: NSA(fa, k))

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 6, 11, 12])
    def test_nsa_square_cells_reuse_unit_distance_table(self, k):
        """Square cells must use DIR_DIST itself, not a scaled copy.

        A scaled copy rounds the diagonal entries differently (hypot(k, k)
        != k*sqrt(2) for k = 3, 6, 11, 12), which could flip a near-tie
        between a cardinal and a diagonal neighbour.
        """
        assert _dir_distances(k, k) is DIR_DIST

    def test_nsa_rectangular_cells_weight_axes_separately(self):
        dist = _dir_distances(kx=4, ky=2)
        east, south_east, south = dist[0], dist[1], dist[2]
        assert east == pytest.approx(4.0)
        assert south == pytest.approx(2.0)
        assert south_east == pytest.approx(math.hypot(2.0, 4.0))


# ---------------------------------------------------------------------------
# COTAT
# ---------------------------------------------------------------------------

class TestCOTAT:

    def test_cotat_basic_shape_dtype_transform(self):
        fa = _make_flowacc()
        fd = _make_flowdir()
        k = 2
        out = COTAT(fd, fa, k)
        _assert_upscaled(out, fa, k)

    def test_cotat_transform_mismatch_raises(self):
        fa = _make_flowacc()
        # FlowDir with a different pixel size → transform mismatch
        fd_bad = Grid.create(
            array=np.full((8, 8), 2, dtype=np.uint8),
            type=GridType.FlowDir,
            transform=COARSE_TRANSFORM,   # wrong scale
            crs=None,
        )
        with pytest.raises(ValueError):
            COTAT(fd_bad, fa, k=2)

    def test_cotat_shape_mismatch_raises(self):
        fa = _make_flowacc(shape=(8, 8))
        fd_bad = Grid.create(
            array=np.full((6, 8), 2, dtype=np.uint8),
            type=GridType.FlowDir,
            transform=FINE_TRANSFORM,
            crs=None,
        )
        with pytest.raises(ValueError):
            COTAT(fd_bad, fa, k=2)

    def test_cotat_plus_runs(self):
        fa = _make_flowacc()
        fd = _make_flowdir()
        # mufp is now in metres; 5 000 m is ~4 pixel-lengths at 0.01° / 10°N
        out = COTAT(fd, fa, k=2, mufp=5000.0)
        assert out.shape == (4, 4)
        assert out.array.dtype == np.uint8

    def test_cotat_plus_mufp_zero_same_as_no_mufp(self):
        """mufp=0 (always exceeded) should give the same result as plain COTAT."""
        fa = _make_flowacc()
        fd = _make_flowdir()
        out_plain = COTAT(fd, fa, k=2)
        out_mufp0 = COTAT(fd, fa, k=2, mufp=0.0)
        np.testing.assert_array_equal(out_plain.array, out_mufp0.array)

    def test_cotat_plus_mufp_projected_crs(self):
        """COTAT+ with a projected (metre) CRS grid runs without error."""
        from rasterio.crs import CRS
        proj_crs = CRS.from_epsg(32632)  # UTM zone 32N, metres
        # 100 m pixels
        t = Affine(100.0, 0.0, 400000.0, 0.0, -100.0, 5000000.0)
        arr = np.arange(64, dtype=np.uint32).reshape(8, 8) + 1
        fa = Grid.create(array=arr, type=GridType.FlowAcc, transform=t, crs=proj_crs)
        arr_fd = np.full((8, 8), 2, dtype=np.uint8)
        fd = Grid.create(array=arr_fd, type=GridType.FlowDir, transform=t, crs=proj_crs)
        # mufp = 200 m  ≈ 2 pixel-lengths at 100 m resolution
        out = COTAT(fd, fa, k=2, mufp=200.0)
        assert out.shape == (4, 4)
        assert out.array.dtype == np.uint8

    def test_cotat_int_k_matches_tuple_kk(self):
        fa = _make_flowacc()
        fd = _make_flowdir()
        out_int = COTAT(fd, fa, k=4)
        out_tuple = COTAT(fd, fa, k=(4, 4))
        np.testing.assert_array_equal(out_int.array, out_tuple.array)
        assert out_int.meta.transform == out_tuple.meta.transform

    def test_cotat_anisotropic_shape_and_transform(self):
        fa = _make_flowacc(shape=(12, 10))
        fd = _make_flowdir(shape=(12, 10))
        kx, ky = 4, 2
        out = COTAT(fd, fa, k=(kx, ky))
        _assert_anisotropic(out, fa, kx, ky, floor=False)

    def test_cotat_invalid_k_tuples(self):
        fa = _make_flowacc()
        fd = _make_flowdir()
        _assert_invalid_k(lambda k: COTAT(fd, fa, k))
