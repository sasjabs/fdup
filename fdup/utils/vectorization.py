"""River network vectorization from D8 flow direction.

Public API
----------
vectorize_network(flowdir, flowacc=None) -> geopandas.GeoDataFrame
    Decompose a D8 grid into river-segment LineStrings.  Segment boundaries
    are headwaters (in-degree 0) and junctions (in-degree >= 2).  Junction
    cells are included as the shared endpoint of incoming segments and the
    start of the outgoing segment, producing topologically connected geometry.

vectorize_tree(seeds, flowdir, *, cutoff=None, rank=None, accuracy=None)
    -> geopandas.GeoDataFrame
    Trace each seed (from :func:`~fdup.utils.tree.river_tree`) from headwater
    to mouth and return one LineString per seed, with optional ``flowacc`` and
    ``accur`` columns.

_warmup(dtype=None) -> None
    Pre-compile all Numba kernels (dtype-agnostic).

Notes
-----
Both functions require ``geopandas >= 0.14`` and ``shapely >= 2.0``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fdup._core.types import Grid, GridType
from fdup._core.validation import (
    check_crs_match,
    check_shape_match,
    check_transform_match,
    check_type,
)

_REQUIRED_SEED_FIELDS = frozenset(
    {"mouth_row", "mouth_col", "headwater_row", "headwater_col", "ncells"}
)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nb_in_degree(dir_arr: np.ndarray) -> np.ndarray:
    """Compute D8 in-degree for every cell.

    Only valid stream cells (1 <= dir <= 128) contribute to downstream
    in-degree counts.  Sinks (dir == 0) and nodata cells (dir > 128) are
    ignored as sources.

    Parameters
    ----------
    dir_arr : uint8, (nrows, ncols)

    Returns
    -------
    np.ndarray, int32, (nrows, ncols)
    """
    nrow = np.int64(dir_arr.shape[0])
    ncol = np.int64(dir_arr.shape[1])

    di = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    idx = np.zeros(256, dtype=np.int64)
    idx[1] = np.int64(0); idx[2] = np.int64(1); idx[4] = np.int64(2); idx[8] = np.int64(3)   # noqa: E702
    idx[16] = np.int64(4); idx[32] = np.int64(5); idx[64] = np.int64(6); idx[128] = np.int64(7)  # noqa: E702

    in_deg = np.zeros((int(nrow), int(ncol)), dtype=np.int32)
    for i in range(int(nrow)):
        for j in range(int(ncol)):
            d = np.int64(dir_arr[i, j])
            if d <= np.int64(0) or d > np.int64(128):
                continue
            k = idx[d]
            ni = np.int64(i) + di[k]
            nj = np.int64(j) + dj[k]
            if np.int64(0) <= ni < nrow and np.int64(0) <= nj < ncol:
                in_deg[int(ni), int(nj)] += 1
    return in_deg


@njit(cache=True)
def _nb_collect_segments(
    dir_arr: np.ndarray,
    in_degree: np.ndarray,
) -> tuple:
    """Walk each river segment downstream and collect cell-coordinate paths.

    A segment starts at every valid cell where ``in_degree == 0``
    (headwater) or ``in_degree >= 2`` (junction).  The walk follows D8
    downstream, appending each cell to the current path.  The walk stops
    when:

    * The current cell is a sink (dir == 0) or nodata (dir > 128).
    * The downstream step would leave the grid.
    * The downstream cell is nodata (dir > 128) — not included.
    * The downstream cell is a junction (in_degree >= 2) — included as the
      last path cell; junction coordinates therefore appear at the end of
      each incoming segment and at the start of the outgoing segment.

    Paths with fewer than 2 cells are discarded (can't form a LineString).

    Parameters
    ----------
    dir_arr   : uint8, (nrows, ncols)
    in_degree : int32, (nrows, ncols)

    Returns
    -------
    all_coords : int32 (total_coords, 2) — flat (row, col) buffer
    seg_start  : int64 (n_segs,)         — offset into all_coords per segment
    seg_len    : int64 (n_segs,)         — cell count per segment
    n_segs     : int64                   — number of valid segments
    coord_ptr  : int64                   — total cells written to all_coords
    """
    nrow = np.int64(dir_arr.shape[0])
    ncol = np.int64(dir_arr.shape[1])

    di = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    idx = np.zeros(256, dtype=np.int64)
    idx[1] = np.int64(0); idx[2] = np.int64(1); idx[4] = np.int64(2); idx[8] = np.int64(3)   # noqa: E702
    idx[16] = np.int64(4); idx[32] = np.int64(5); idx[64] = np.int64(6); idx[128] = np.int64(7)  # noqa: E702

    # Count valid stream cells for initial buffer sizing
    n_valid = np.int64(0)
    for i in range(int(nrow)):
        for j in range(int(ncol)):
            d = np.int64(dir_arr[i, j])
            if d > np.int64(0) and d <= np.int64(128):
                n_valid += np.int64(1)

    max_coords = max(np.int64(64), n_valid // np.int64(4) + np.int64(1))
    max_segs   = max(np.int64(16), n_valid // np.int64(4) + np.int64(1))

    all_coords = np.empty((int(max_coords), 2), dtype=np.int32)
    seg_start  = np.empty(int(max_segs), dtype=np.int64)
    seg_len    = np.empty(int(max_segs), dtype=np.int64)

    n_segs    = np.int64(0)
    coord_ptr = np.int64(0)

    max_walk = nrow * ncol + np.int64(1)   # cycle-safety bound

    for si in range(int(nrow)):
        for sj in range(int(ncol)):
            d_start = np.int64(dir_arr[si, sj])
            if d_start <= np.int64(0) or d_start > np.int64(128):
                continue
            indeg = in_degree[si, sj]
            if indeg != 0 and indeg < 2:
                continue  # interior cell (in-degree 1) — not a segment start

            seg_offset = coord_ptr
            ci = np.int64(si)
            cj = np.int64(sj)

            for _ in range(int(max_walk)):
                # Grow coord buffer if needed
                if coord_ptr >= max_coords:
                    max_coords *= np.int64(2)
                    new_c = np.empty((int(max_coords), 2), dtype=np.int32)
                    new_c[:coord_ptr] = all_coords[:coord_ptr]
                    all_coords = new_c

                all_coords[coord_ptr, 0] = np.int32(ci)
                all_coords[coord_ptr, 1] = np.int32(cj)
                coord_ptr += np.int64(1)

                cd = np.int64(dir_arr[int(ci), int(cj)])
                if cd <= np.int64(0) or cd > np.int64(128):
                    break  # sink (0) or nodata (>128): nowhere to step

                k  = idx[cd]
                ni = ci + di[k]
                nj = cj + dj[k]

                if ni < np.int64(0) or ni >= nrow or nj < np.int64(0) or nj >= ncol:
                    break  # downstream step leaves the grid

                nd = np.int64(dir_arr[int(ni), int(nj)])
                if nd > np.int64(128):
                    break  # downstream is nodata — do not include

                if in_degree[int(ni), int(nj)] >= 2:
                    # Junction: include as segment endpoint then stop
                    if coord_ptr >= max_coords:
                        max_coords *= np.int64(2)
                        new_c = np.empty((int(max_coords), 2), dtype=np.int32)
                        new_c[:coord_ptr] = all_coords[:coord_ptr]
                        all_coords = new_c
                    all_coords[coord_ptr, 0] = np.int32(ni)
                    all_coords[coord_ptr, 1] = np.int32(nj)
                    coord_ptr += np.int64(1)
                    break

                ci = ni
                cj = nj

            length = coord_ptr - seg_offset
            if length >= np.int64(2):
                if n_segs >= max_segs:
                    max_segs *= np.int64(2)
                    new_ss = np.empty(int(max_segs), dtype=np.int64)
                    new_ss[:n_segs] = seg_start[:n_segs]
                    seg_start = new_ss
                    new_sl = np.empty(int(max_segs), dtype=np.int64)
                    new_sl[:n_segs] = seg_len[:n_segs]
                    seg_len = new_sl
                seg_start[n_segs] = seg_offset
                seg_len[n_segs]   = length
                n_segs += np.int64(1)
            else:
                coord_ptr = seg_offset  # discard degenerate 1-cell path

    return all_coords, seg_start, seg_len, n_segs, coord_ptr


@njit(cache=True)
def _nb_trace_seed_paths(
    seeds_geom: np.ndarray,  # uint32 (nseed, 5): [mouth_row, mouth_col, hw_row, hw_col, ncells]
    nseed: np.int64,
    dir_arr: np.ndarray,     # uint8 (nrows, ncols)
) -> tuple:
    """Trace each seed from headwater to mouth, collecting (row, col) paths.

    Follows D8 ``dir_arr`` starting from headwater until the mouth cell is
    reached or the trace cannot continue (out of bounds, sink, nodata).

    Parameters
    ----------
    seeds_geom : uint32 (nseed, 5)
        Per-seed layout (matching :func:`~fdup.utils.tree.mask_seeds`):
        ``[mouth_row, mouth_col, headwater_row, headwater_col, ncells]``.
    nseed : int64
    dir_arr : uint8 (nrows, ncols)

    Returns
    -------
    all_coords : int32 (total_coords, 2)
    seg_start  : int64 (nseed,)
    seg_len    : int64 (nseed,)
    coord_ptr  : int64
    """
    nrow = np.int64(dir_arr.shape[0])
    ncol = np.int64(dir_arr.shape[1])

    di = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    idx = np.zeros(256, dtype=np.int64)
    idx[1] = np.int64(0); idx[2] = np.int64(1); idx[4] = np.int64(2); idx[8] = np.int64(3)   # noqa: E702
    idx[16] = np.int64(4); idx[32] = np.int64(5); idx[64] = np.int64(6); idx[128] = np.int64(7)  # noqa: E702

    max_coords = max(np.int64(64), nseed * np.int64(8))
    all_coords = np.empty((int(max_coords), 2), dtype=np.int32)
    seg_start  = np.empty(int(nseed), dtype=np.int64)
    seg_len    = np.empty(int(nseed), dtype=np.int64)
    coord_ptr  = np.int64(0)

    for s in range(int(nseed)):
        hw_r      = np.int64(seeds_geom[s, 2])
        hw_c      = np.int64(seeds_geom[s, 3])
        m_r       = np.int64(seeds_geom[s, 0])
        m_c       = np.int64(seeds_geom[s, 1])
        max_steps = np.int64(seeds_geom[s, 4]) + np.int64(2)  # ncells + safety

        seg_offset = coord_ptr
        row = hw_r
        col = hw_c

        for _ in range(int(max_steps)):
            if coord_ptr >= max_coords:
                max_coords *= np.int64(2)
                new_c = np.empty((int(max_coords), 2), dtype=np.int32)
                new_c[:coord_ptr] = all_coords[:coord_ptr]
                all_coords = new_c

            all_coords[coord_ptr, 0] = np.int32(row)
            all_coords[coord_ptr, 1] = np.int32(col)
            coord_ptr += np.int64(1)

            if row == m_r and col == m_c:
                break

            d = np.int64(dir_arr[int(row), int(col)])
            if d <= np.int64(0) or d > np.int64(128):
                break
            k  = idx[d]
            nr = row + di[k]
            nc = col + dj[k]
            if nr < np.int64(0) or nr >= nrow or nc < np.int64(0) or nc >= ncol:
                break
            row = nr
            col = nc

        seg_start[s] = seg_offset
        seg_len[s]   = coord_ptr - seg_offset

    return all_coords, seg_start, seg_len, coord_ptr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vectorize_network(
    flowdir: Grid,
    flowacc: Grid | None = None,
) -> "geopandas.GeoDataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Decompose a D8 flow-direction grid into a river-segment GeoDataFrame.

    Segment boundaries are defined by headwater cells (in-degree 0) and
    junction cells (in-degree >= 2).  Each segment is returned as a
    ``LineString`` built from cell-centre coordinates
    (``transform * (col + 0.5, row + 0.5)``).  Junction cells appear as
    the shared endpoint of incoming segments and the start of the outgoing
    segment, so all LineStrings are topologically connected at junctions.

    Parameters
    ----------
    flowdir :
        ``GridType.FlowDir``, uint8.
    flowacc :
        Optional ``GridType.FlowAcc``, same shape/transform/CRS as
        *flowdir*.  When provided, a ``flowacc`` column is added with the
        flow-accumulation value at each segment's most-downstream cell.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``geometry`` (LineString); optionally ``flowacc``.
        CRS is taken from ``flowdir.meta.crs``.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "vectorize_network requires geopandas and shapely. "
            "Install them with: pip install 'geopandas>=0.14' 'shapely>=2.0'"
        ) from exc

    check_type(flowdir, GridType.FlowDir)
    if flowacc is not None:
        check_type(flowacc, GridType.FlowAcc)
        check_shape_match(flowdir, flowacc)
        check_transform_match(flowdir, flowacc)
        check_crs_match(flowdir, flowacc)

    dir_arr   = flowdir.array
    transform = flowdir.meta.transform
    crs       = flowdir.meta.crs

    in_degree = _nb_in_degree(dir_arr)
    all_coords, seg_start, seg_len, n_segs, _ = _nb_collect_segments(dir_arr, in_degree)

    n          = int(n_segs)
    geometries = []
    flowacc_vals: list[float] | None = [] if flowacc is not None else None

    for s in range(n):
        start  = int(seg_start[s])
        length = int(seg_len[s])
        rows   = all_coords[start : start + length, 0]
        cols   = all_coords[start : start + length, 1]
        xs     = transform.c + (cols + 0.5) * transform.a
        ys     = transform.f + (rows + 0.5) * transform.e
        geometries.append(LineString(list(zip(xs.tolist(), ys.tolist()))))
        if flowacc_vals is not None:
            mouth_r = int(rows[-1])
            mouth_c = int(cols[-1])
            flowacc_vals.append(float(flowacc.array[mouth_r, mouth_c]))  # type: ignore[union-attr]

    data: dict = {"geometry": geometries}
    if flowacc_vals is not None:
        data["flowacc"] = flowacc_vals

    return gpd.GeoDataFrame(data, crs=crs)


def vectorize_tree(
    seeds: np.ndarray,
    flowdir: Grid,
    *,
    cutoff: float | None = None,
    rank: int | None = None,
    accuracy: np.ndarray | None = None,
) -> "geopandas.GeoDataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Convert a seeds array to a GeoDataFrame of river-segment LineStrings.

    Each seed (as produced by :func:`~fdup.utils.tree.river_tree`) is
    traced from headwater to mouth by following *flowdir* downstream.  The
    result is one row per seed, with a ``flowacc`` column taken from
    ``seeds["acc"]`` and an optional ``accur`` column from *accuracy*.

    Parameters
    ----------
    seeds :
        Structured array with at least the fields
        ``mouth_row``, ``mouth_col``, ``headwater_row``, ``headwater_col``,
        ``ncells``, and ``acc`` (as returned by :func:`~fdup.utils.tree.river_tree`).
    flowdir :
        ``GridType.FlowDir``, uint8.
    cutoff :
        Keep only seeds where ``seeds["acc"] >= cutoff``.
        Mutually exclusive with *rank*.
    rank :
        Keep the top-*rank* seeds by ``acc`` (largest first).
        Mutually exclusive with *cutoff*.
    accuracy :
        1-D array of per-seed accuracy values, same length as *seeds*.
        The subset corresponding to selected seeds is attached as the
        ``accur`` column.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``geometry`` (LineString), ``flowacc``; optionally
        ``accur``.  CRS is taken from ``flowdir.meta.crs``.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "vectorize_tree requires geopandas and shapely. "
            "Install them with: pip install 'geopandas>=0.14' 'shapely>=2.0'"
        ) from exc

    check_type(flowdir, GridType.FlowDir)

    if seeds.dtype.names is None or not _REQUIRED_SEED_FIELDS.issubset(seeds.dtype.names):
        missing = _REQUIRED_SEED_FIELDS - set(seeds.dtype.names or [])
        raise ValueError(
            f"seeds array is missing required fields: {sorted(missing)}"
        )

    if cutoff is not None and rank is not None:
        raise ValueError("cutoff and rank are mutually exclusive")

    # --- seed selection ---
    if cutoff is not None:
        mask = seeds["acc"] >= cutoff
        sel_idx = np.where(mask)[0]
    elif rank is not None:
        order   = np.argsort(seeds["acc"])[::-1]
        n_take  = min(int(rank), len(seeds))
        sel_idx = order[:n_take]
    else:
        sel_idx = np.arange(len(seeds), dtype=np.int64)

    selected = seeds[sel_idx]

    if accuracy is not None:
        accuracy = np.asarray(accuracy)
        if len(accuracy) != len(seeds):
            raise ValueError(
                f"accuracy must have the same length as seeds ({len(seeds)}), "
                f"got {len(accuracy)}"
            )

    transform = flowdir.meta.transform
    crs       = flowdir.meta.crs
    n         = len(selected)

    if n == 0:
        cols: dict = {"geometry": [], "flowacc": []}
        if accuracy is not None:
            cols["accur"] = []
        return gpd.GeoDataFrame(cols, crs=crs)

    # --- build raw geometry buffer for the kernel ---
    seeds_geom = np.empty((n, 5), dtype=np.uint32)
    seeds_geom[:, 0] = selected["mouth_row"].astype(np.uint32)
    seeds_geom[:, 1] = selected["mouth_col"].astype(np.uint32)
    seeds_geom[:, 2] = selected["headwater_row"].astype(np.uint32)
    seeds_geom[:, 3] = selected["headwater_col"].astype(np.uint32)
    seeds_geom[:, 4] = selected["ncells"].astype(np.uint32)

    all_coords, seg_start, seg_len, _ = _nb_trace_seed_paths(
        seeds_geom, np.int64(n), flowdir.array
    )

    geometries: list = []
    flowacc_out: list[float] = []

    for s in range(n):
        start  = int(seg_start[s])
        length = int(seg_len[s])
        rows   = all_coords[start : start + length, 0]
        cols   = all_coords[start : start + length, 1]
        xs     = transform.c + (cols + 0.5) * transform.a
        ys     = transform.f + (rows + 0.5) * transform.e
        if length >= 2:
            line = LineString(list(zip(xs.tolist(), ys.tolist())))
        elif length == 1:
            # Degenerate single-cell seed: create a zero-length segment
            line = LineString([(xs[0], ys[0]), (xs[0], ys[0])])
        else:
            line = LineString()
        geometries.append(line)
        flowacc_out.append(float(selected["acc"][s]))

    data: dict = {"geometry": geometries, "flowacc": flowacc_out}
    if accuracy is not None:
        data["accur"] = [float(accuracy[int(i)]) for i in sel_idx]

    return gpd.GeoDataFrame(data, crs=crs)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup(dtype: object = None) -> None:  # noqa: ARG001
    """Pre-compile all Numba kernels.

    *dtype* is accepted for registry uniformity but ignored; both kernels
    are dtype-agnostic (FlowDir is always uint8).
    """
    # Tiny 2×2 flow-direction grid: (0,0) flows SE → (1,1).
    _dir = np.array([[2, 0], [0, 0]], dtype=np.uint8)
    _in_deg = _nb_in_degree(_dir)
    _nb_collect_segments(_dir, _in_deg)

    # Seed: mouth=(1,1), headwater=(0,0), ncells=2.
    _seeds_geom = np.array([[1, 1, 0, 0, 2]], dtype=np.uint32)
    _nb_trace_seed_paths(_seeds_geom, np.int64(1), _dir)
