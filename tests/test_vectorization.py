"""Tests for fdup.utils.vectorization: vectorize_network and vectorize_tree.

Synthetic network (5×5 grid):

    A(0,0)--SE--> a(1,1)--SE--> J(2,2) <--SW--b(1,3) <--SW-- B(0,4)
                                  |S
                                m(3,2)
                                  |S
                                O(4,2)-->S--> [out of bounds]

D8 codes: SE=2, SW=8, S=4.

In-degree:
    A=0 (headwater), B=0 (headwater), a=1, b=1,
    J=2 (junction), m=1, O=1.

Expected vectorize_network segments (3 total):
    1. A(0,0) → a(1,1) → J(2,2)
    2. B(0,4) → b(1,3) → J(2,2)
    3. J(2,2) → m(3,2) → O(4,2)

Expected vectorize_tree seeds from river_tree (2 seeds, cell-count accumulation):
    A=1, a=2, B=1, b=2, J=5, m=6, O=7
    river_tree picks (4,2) first (acc=7), traces upstream through
    (3,2)→(2,2)→(1,3)→(0,4) because (1,3) ties (1,1) and NE wins with >=.
    Seed 1: mouth=(4,2), headwater=(0,4), ncells=5, acc=7.
    (1,1) is unclaimed → Seed 2: mouth=(1,1), headwater=(0,0), ncells=2, acc=2.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

pytest.importorskip("geopandas")
pytest.importorskip("shapely")

import geopandas as gpd  # noqa: E402  (after importorskip guard)
from shapely.geometry import LineString  # noqa: E402

from fdup._core.types import Grid, GridType  # noqa: E402
from fdup.utils.flowacc import flow_accumulation  # noqa: E402
from fdup.utils.tree import river_tree  # noqa: E402
from fdup.utils.vectorization import vectorize_network, vectorize_tree  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)  # 1-unit pixels, top-left at (0,5)


def _make_flowdir() -> Grid:
    """5×5 synthetic D8 grid with one confluence."""
    arr = np.zeros((5, 5), dtype=np.uint8)
    arr[0, 0] = 2   # A: SE → (1,1)
    arr[1, 1] = 2   # a: SE → (2,2)
    arr[0, 4] = 8   # B: SW → (1,3)
    arr[1, 3] = 8   # b: SW → (2,2)
    arr[2, 2] = 4   # J: S  → (3,2)
    arr[3, 2] = 4   # m: S  → (4,2)
    arr[4, 2] = 4   # O: S  → out-of-bounds
    return Grid.create(array=arr, type=GridType.FlowDir, transform=TRANSFORM, crs=None)


@pytest.fixture(scope="module")
def fd() -> Grid:
    return _make_flowdir()


@pytest.fixture(scope="module")
def fa(fd: Grid) -> Grid:
    """Cell-count flow accumulation on the synthetic grid."""
    return flow_accumulation(fd, area=False)


@pytest.fixture(scope="module")
def seeds(fd: Grid, fa: Grid) -> np.ndarray:
    """Seeds from river_tree on the synthetic grid."""
    _tree, s = river_tree(fd, fa)
    return s


# ---------------------------------------------------------------------------
# vectorize_network — basic correctness
# ---------------------------------------------------------------------------


class TestVectorizeNetwork:

    def test_segment_count(self, fd):
        gdf = vectorize_network(fd)
        assert len(gdf) == 3, f"expected 3 segments, got {len(gdf)}"

    def test_returns_geodataframe(self, fd):
        gdf = vectorize_network(fd)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_all_geometries_are_linestrings(self, fd):
        gdf = vectorize_network(fd)
        assert all(isinstance(geom, LineString) for geom in gdf.geometry)

    def test_each_segment_has_at_least_two_points(self, fd):
        gdf = vectorize_network(fd)
        for geom in gdf.geometry:
            assert len(geom.coords) >= 2

    def test_junction_is_shared_endpoint(self, fd):
        """All three segments should share (2.5, 2.5), the cell-centre of J(2,2)."""
        gdf = vectorize_network(fd)
        # J(2,2): x = 0 + (2 + 0.5)*1 = 2.5, y = 5 + (2 + 0.5)*(-1) = 2.5
        junction_xy = (2.5, 2.5)
        shared = [
            any(abs(x - junction_xy[0]) < 1e-9 and abs(y - junction_xy[1]) < 1e-9
                for x, y in geom.coords)
            for geom in gdf.geometry
        ]
        assert all(shared), "junction (2.5, 2.5) should appear in every segment"

    def test_no_flowacc_column_without_arg(self, fd):
        gdf = vectorize_network(fd)
        assert "flowacc" not in gdf.columns

    def test_flowacc_column_present_with_arg(self, fd, fa):
        gdf = vectorize_network(fd, flowacc=fa)
        assert "flowacc" in gdf.columns

    def test_flowacc_values_at_mouths(self, fd, fa):
        """Segments ending at J have flowacc=5; segment ending at O has flowacc=7."""
        gdf = vectorize_network(fd, flowacc=fa)
        fa_vals = sorted(gdf["flowacc"].tolist())
        # Two segments end at J (acc=5), one at O (acc=7)
        assert fa_vals == pytest.approx([5.0, 5.0, 7.0], rel=1e-5)

    def test_crs_none_when_no_crs(self, fd):
        gdf = vectorize_network(fd)
        assert gdf.crs is None

    def test_segment_coordinates_match_cell_centres(self, fd):
        """A(0,0) → a(1,1) → J(2,2): cell centres (0.5,4.5), (1.5,3.5), (2.5,2.5)."""
        gdf = vectorize_network(fd)
        all_coords: set[tuple[float, float]] = set()
        for geom in gdf.geometry:
            all_coords.update(geom.coords)
        # All valid stream cells should appear as cell centres
        expected = {
            (0.5, 4.5),   # A(0,0)
            (1.5, 3.5),   # a(1,1)
            (4.5, 4.5),   # B(0,4)
            (3.5, 3.5),   # b(1,3)
            (2.5, 2.5),   # J(2,2)
            (2.5, 1.5),   # m(3,2)
            (2.5, 0.5),   # O(4,2)
        }
        for pt in expected:
            assert any(abs(x - pt[0]) < 1e-9 and abs(y - pt[1]) < 1e-9
                       for x, y in all_coords), f"cell-centre {pt} not found in output"

    def test_empty_grid_returns_empty_gdf(self):
        arr = np.zeros((4, 4), dtype=np.uint8)  # all sinks
        fd_empty = Grid.create(array=arr, type=GridType.FlowDir, transform=TRANSFORM)
        gdf = vectorize_network(fd_empty)
        assert len(gdf) == 0

    def test_wrong_type_raises(self, fa):
        with pytest.raises(ValueError):
            vectorize_network(fa)  # type: ignore[arg-type]

    def test_flowacc_shape_mismatch_raises(self, fd):
        arr2 = np.ones((3, 3), dtype=np.float32)
        fa_bad = Grid.create(array=arr2, type=GridType.FlowAcc, transform=TRANSFORM)
        with pytest.raises(ValueError):
            vectorize_network(fd, flowacc=fa_bad)

    def test_straight_line_network(self):
        """Single headwater → terminus: two cells, one segment."""
        arr = np.zeros((3, 1), dtype=np.uint8)
        arr[0, 0] = 4   # S → (1,0)
        arr[1, 0] = 4   # S → (2,0)
        fd_line = Grid.create(array=arr, type=GridType.FlowDir, transform=TRANSFORM)
        gdf = vectorize_network(fd_line)
        assert len(gdf) == 1
        assert len(gdf.iloc[0].geometry.coords) == 3  # (0,0), (1,0), (2,0)


# ---------------------------------------------------------------------------
# vectorize_tree — basic correctness
# ---------------------------------------------------------------------------


class TestVectorizeTree:

    def test_default_returns_all_seeds(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        assert len(gdf) == len(seeds)

    def test_returns_geodataframe(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_flowacc_column_present(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        assert "flowacc" in gdf.columns

    def test_flowacc_values_match_seed_acc(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        expected = sorted(float(a) for a in seeds["acc"])
        actual   = sorted(gdf["flowacc"].tolist())
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_all_geometries_are_linestrings(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        assert all(isinstance(geom, LineString) for geom in gdf.geometry)

    def test_seed_ncells_matches_linestring_length(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        for row, seed in zip(gdf.itertuples(), seeds):
            assert len(row.geometry.coords) == int(seed["ncells"]), (
                f"seed ncells={seed['ncells']} but geometry has "
                f"{len(row.geometry.coords)} points"
            )

    def test_headwater_is_first_coord(self, seeds, fd):
        """The LineString should start at the headwater cell centre."""
        gdf = vectorize_tree(seeds, fd)
        t   = TRANSFORM
        for geom, seed in zip(gdf.geometry, seeds):
            hw_r = int(seed["headwater_row"])
            hw_c = int(seed["headwater_col"])
            expected_x = t.c + (hw_c + 0.5) * t.a
            expected_y = t.f + (hw_r + 0.5) * t.e
            x0, y0 = geom.coords[0]
            assert abs(x0 - expected_x) < 1e-9
            assert abs(y0 - expected_y) < 1e-9

    def test_mouth_is_last_coord(self, seeds, fd):
        """The LineString should end at the mouth cell centre."""
        gdf = vectorize_tree(seeds, fd)
        t   = TRANSFORM
        for geom, seed in zip(gdf.geometry, seeds):
            m_r = int(seed["mouth_row"])
            m_c = int(seed["mouth_col"])
            expected_x = t.c + (m_c + 0.5) * t.a
            expected_y = t.f + (m_r + 0.5) * t.e
            xn, yn = geom.coords[-1]
            assert abs(xn - expected_x) < 1e-9
            assert abs(yn - expected_y) < 1e-9

    def test_cutoff_filters_seeds(self, seeds, fd):
        """cutoff=3 keeps only seeds with acc >= 3 (only the main trunk, acc=7)."""
        gdf = vectorize_tree(seeds, fd, cutoff=3.0)
        assert all(v >= 3.0 for v in gdf["flowacc"])
        assert len(gdf) == sum(1 for s in seeds if float(s["acc"]) >= 3.0)

    def test_rank_keeps_top_seeds(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd, rank=1)
        assert len(gdf) == 1
        assert float(gdf.iloc[0]["flowacc"]) == pytest.approx(
            float(seeds["acc"].max()), rel=1e-5
        )

    def test_cutoff_and_rank_exclusive(self, seeds, fd):
        with pytest.raises(ValueError, match="mutually exclusive"):
            vectorize_tree(seeds, fd, cutoff=1.0, rank=1)

    def test_accuracy_column_attached(self, seeds, fd):
        acc_arr = np.arange(len(seeds), dtype=np.float64) * 10.0
        gdf = vectorize_tree(seeds, fd, accuracy=acc_arr)
        assert "accur" in gdf.columns
        assert len(gdf["accur"]) == len(seeds)

    def test_accuracy_subset_matches_selection(self, seeds, fd):
        """Accuracy values for selected subset must match the original indices."""
        acc_arr = np.arange(len(seeds), dtype=np.float64) * 100.0
        gdf = vectorize_tree(seeds, fd, cutoff=3.0, accuracy=acc_arr)
        sel_idx = np.where(seeds["acc"] >= 3.0)[0]
        expected_accur = [float(acc_arr[i]) for i in sel_idx]
        assert gdf["accur"].tolist() == pytest.approx(expected_accur, rel=1e-9)

    def test_accuracy_wrong_length_raises(self, seeds, fd):
        with pytest.raises(ValueError, match="same length as seeds"):
            vectorize_tree(seeds, fd, accuracy=np.zeros(len(seeds) + 1))

    def test_empty_selection_returns_empty_gdf(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd, cutoff=1e18)
        assert len(gdf) == 0
        assert "flowacc" in gdf.columns

    def test_crs_none_when_no_crs(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        assert gdf.crs is None

    def test_no_accur_column_without_accuracy(self, seeds, fd):
        gdf = vectorize_tree(seeds, fd)
        assert "accur" not in gdf.columns

    def test_wrong_type_raises(self, seeds, fa):
        with pytest.raises(ValueError):
            vectorize_tree(seeds, fa)  # type: ignore[arg-type]

    def test_empty_seeds_returns_empty_gdf(self, fd, seeds):
        empty = seeds[:0]
        gdf = vectorize_tree(empty, fd)
        assert len(gdf) == 0


# ---------------------------------------------------------------------------
# vectorize_network — with CRS
# ---------------------------------------------------------------------------


def test_crs_propagated():
    """CRS from flowdir is set on the output GeoDataFrame."""
    try:
        from rasterio.crs import CRS
        crs = CRS.from_epsg(4326)
    except Exception:
        pytest.skip("rasterio.crs not available")

    arr = np.zeros((3, 3), dtype=np.uint8)
    arr[0, 0] = 4  # S → (1,0)
    arr[1, 0] = 4  # S → (2,0)
    fd_crs = Grid.create(array=arr, type=GridType.FlowDir, transform=TRANSFORM, crs=crs)
    gdf = vectorize_network(fd_crs)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# Public API export check
# ---------------------------------------------------------------------------


def test_public_api_exports():
    from fdup import utils
    assert hasattr(utils, "vectorize_network")
    assert hasattr(utils, "vectorize_tree")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_vectorize_network(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    fd = _make_flowdir()
    fd_path = str(tmp_path / "fd.tif")
    fdup_io.write(fd, fd_path, overwrite=True)
    out_path = str(tmp_path / "net.gpkg")

    main(["vectorize-network", "--flowdir", fd_path, "-o", out_path])

    gdf = gpd.read_file(out_path)
    assert len(gdf) == 3
    assert all(isinstance(geom, LineString) for geom in gdf.geometry)


def test_cli_vectorize_network_with_flowacc(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    fd = _make_flowdir()
    fa = flow_accumulation(fd, area=False)
    fd_path = str(tmp_path / "fd.tif")
    fa_path = str(tmp_path / "fa.tif")
    fdup_io.write(fd, fd_path, overwrite=True)
    fdup_io.write(fa, fa_path, overwrite=True)
    out_path = str(tmp_path / "net.gpkg")

    main(["vectorize-network", "--flowdir", fd_path, "--flowacc", fa_path, "-o", out_path])

    gdf = gpd.read_file(out_path)
    assert len(gdf) == 3
    assert "flowacc" in gdf.columns


def test_cli_vectorize_tree(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    fd = _make_flowdir()
    fa = flow_accumulation(fd, area=False)
    _, seeds = river_tree(fd, fa)

    fd_path    = str(tmp_path / "fd.tif")
    seeds_path = str(tmp_path / "seeds.npz")
    out_path   = str(tmp_path / "tree.gpkg")

    fdup_io.write(fd, fd_path, overwrite=True)
    np.savez(seeds_path, seeds=seeds)

    main(["vectorize-tree", "--seeds", seeds_path, "--flowdir", fd_path, "-o", out_path])

    gdf = gpd.read_file(out_path)
    assert len(gdf) == len(seeds)
    assert "flowacc" in gdf.columns


def test_cli_vectorize_tree_with_cutoff(tmp_path):
    from fdup import io as fdup_io
    from fdup.cli import main

    fd = _make_flowdir()
    fa = flow_accumulation(fd, area=False)
    _, seeds = river_tree(fd, fa)

    fd_path    = str(tmp_path / "fd.tif")
    seeds_path = str(tmp_path / "seeds.npz")
    out_path   = str(tmp_path / "tree_cut.gpkg")

    fdup_io.write(fd, fd_path, overwrite=True)
    np.savez(seeds_path, seeds=seeds)

    main([
        "vectorize-tree", "--seeds", seeds_path, "--flowdir", fd_path,
        "--cutoff", "5", "-o", out_path,
    ])

    gdf = gpd.read_file(out_path)
    assert all(v >= 5.0 for v in gdf["flowacc"])
