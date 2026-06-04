"""Package surface smoke tests (Phase 5, Step 5.7)."""
from __future__ import annotations

import importlib

import pytest


def test_top_level_imports():
    import fdup
    assert hasattr(fdup, "Grid")
    assert hasattr(fdup, "GridMeta")
    assert hasattr(fdup, "GridType")
    assert callable(fdup.warmup)


def test_submodule_imports():
    import fdup
    assert callable(fdup.io.read)
    assert callable(fdup.io.write)
    assert callable(fdup.upscalers.DMM)
    assert callable(fdup.upscalers.NSA)
    assert callable(fdup.upscalers.COTAT)
    assert callable(fdup.utils.d8)
    assert callable(fdup.utils.flow_accumulation)
    assert callable(fdup.utils.snap_pour_cell)
    assert callable(fdup.utils.delineate_watershed)
    assert callable(fdup.utils.disaggregate_mask)
    assert callable(fdup.utils.mask_area)
    assert callable(fdup.utils.match_grids)
    assert callable(fdup.utils.river_tree)
    assert callable(fdup.evals.compare_watersheds)
    assert callable(fdup.evals.compare_flowdir)
    assert callable(fdup.evals.huac)
    assert callable(fdup.evals.flowdir_windrose)
    assert callable(fdup.evals.windrose_emd)


def test_warmup_runs():
    import fdup
    fdup.warmup()  # must not raise; covers all numba kernels × FlowAcc dtypes


def test_no_legacy_base():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fdup.base")
