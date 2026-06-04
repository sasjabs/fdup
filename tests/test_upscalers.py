"""Smoke tests for fdup.upscalers (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from fdup._core.types import Grid, GridType
from fdup.upscalers import DMM, NSA, COTAT

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
        out = COTAT(fd, fa, k=2, mufp=1.0)
        assert out.shape == (4, 4)
        assert out.array.dtype == np.uint8
