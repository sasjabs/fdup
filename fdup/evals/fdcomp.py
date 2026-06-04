"""Flow direction comparison via per-seed river accuracy scoring.

Public API
----------
compare_flowdir(flowdir_fine, flowdir_coarse, seeds, *, shuffle, alpha,
                strict_upstream, cascade) -> np.ndarray
    Port of ``d8comp`` from the raw ``fdcompv2.py`` script, adapted for
    ``Grid``-based inputs.  Returns a float32 array of per-seed accuracy
    scores in [0, 1].

    The fine→coarse cell mapping uses ``check_aligned`` offsets
    ``(kx, ky, off_r, off_c)`` instead of the raw script's silent
    ``int64(dir1.shape[0] // nrow2)`` ratio, which silently produced wrong
    results when the two grids were not co-registered at (0, 0).

_warmup(dtype)
    Pre-compile the ``_d8comp`` numba kernel.  The *dtype* argument is
    accepted for warmup-registry uniformity; the kernel signature does not
    depend on the FlowAcc dtype.
"""

from __future__ import annotations

import numpy as np
from numba import int64, njit, prange

from fdup._core.types import Grid, GridType
from fdup._core.validation import check_aligned, check_crs_match, check_type

# Expected field names for the structured seeds array produced by river_tree.
_SEEDS_FIELDS = (
    "mouth_row",
    "mouth_col",
    "headwater_row",
    "headwater_col",
    "ncells",
    "acc",
    "length_m",
)
_SEEDS_HINT = (
    "Hint: seeds must be the structured array returned by fdup.utils.river_tree()."
)


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _d8comp(
    dir1,
    dir2,
    seeds_geom,
    kx,
    ky,
    off_r,
    off_c,
    shuffle=True,
    alpha=0.0,
    strict_upstream=False,
    cascade=False,
):
    """Parallel per-seed flow direction accuracy scoring.

    This is a direct port of ``d8comp`` from ``fdcompv2.py`` with the
    single change that the fine→coarse pixel mapping uses the explicit
    alignment offsets ``(kx, ky, off_r, off_c)`` supplied by
    ``check_aligned`` instead of the implicit ratio ``shape[0] // nrow2``.

    Parameters
    ----------
    dir1 : uint8 array (nrow1, ncol1)
        Fine-resolution flow direction grid.
    dir2 : uint8 array (nrow2, ncol2)
        Coarse-resolution flow direction grid.
    seeds_geom : uint32 array (nseeds, 5)
        Columns: [headwater_row, headwater_col, mouth_row, mouth_col, ncells].
        *headwater* is the upstream starting pixel; *mouth* is the downstream
        destination.  This layout matches the raw ``d8tree`` column order.
    kx, ky : int64
        Number of fine pixels per coarse pixel along X (columns) and Y (rows).
    off_r, off_c : int64
        Fine-pixel row/column index of the top-left corner of the coarse grid
        within the fine grid.
    shuffle : bool
        Randomise seed processing order for better parallel load balancing.
    alpha : float
        Harmonic decay coefficient for off-path deviation penalty.
        ``weight = 1 / (1 + steps_off_path * alpha)``; 0 disables the penalty.
    strict_upstream : bool
        When True a coarse cell whose dir2 trace lands upstream of the current
        cell is treated as incorrect rather than correct.
    cascade : bool
        When True correctness is propagated in upstream order; a cell is
        correct only if it traces to a downstream path cell that is itself
        correct.  Forces strict-upstream semantics internally.

    Returns
    -------
    np.ndarray float32, shape (nseeds,)
        Per-seed river accuracy score in [0, 1], or -1 for degenerate
        single-coarse-cell paths.
    """
    # D8 direction offsets (ESRI power-of-two convention)
    dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    di = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

    # Decode LUT: D8 byte → neighbor index
    idx = np.zeros(256, dtype=np.int64)
    idx[1]   = 0
    idx[2]   = 1
    idx[4]   = 2
    idx[8]   = 3
    idx[16]  = 4
    idx[32]  = 5
    idx[64]  = 6
    idx[128] = 7

    nrow2 = int64(dir2.shape[0])
    ncol2 = int64(dir2.shape[1])
    nrow1 = int64(dir1.shape[0])
    ncol1 = int64(dir1.shape[1])

    nseeds = int64(seeds_geom.shape[0])
    res = np.zeros(nseeds, dtype=np.float32)

    use_strict = strict_upstream or cascade

    perm = np.arange(nseeds, dtype=np.uint32)
    if shuffle:
        for i in range(nseeds - 1, 0, -1):
            j_swap = np.random.randint(0, i + 1)
            perm[i], perm[j_swap] = perm[j_swap], perm[i]

    for k in prange(nseeds):
        pk = int64(perm[k])
        # seeds_geom layout: [hw_row, hw_col, mouth_row, mouth_col, ncells]
        i1 = int64(seeds_geom[pk, 0])   # headwater (upstream start)
        j1 = int64(seeds_geom[pk, 1])
        i2 = int64(seeds_geom[pk, 2])   # mouth (downstream destination)
        j2 = int64(seeds_geom[pk, 3])
        max_cells = int64(seeds_geom[pk, 4])

        d2i = np.empty(max_cells, dtype=np.int64)
        d2j = np.empty(max_cells, dtype=np.int64)
        d2n = np.empty(max_cells, dtype=np.int64)

        i = i1
        j = j1
        # Fine-to-coarse mapping: (row - off_r) // ky
        icur = (i - off_r) // ky
        jcur = (j - off_c) // kx
        ncur = int64(1)
        cc   = int64(0)

        max_steps = max_cells + int64(10)
        step_count = int64(0)

        # ---- trace fine path downstream toward mouth ----------------------
        while i != i2 or j != j2:
            dv = dir1[i, j]
            if dv == 0 or dv == 255 or dv > 128:
                break
            d_offset = idx[dv]
            if d_offset == 0 and dv != 1:
                break
            i = i + di[d_offset]
            j = j + dj[d_offset]

            if i < 0 or i >= nrow1 or j < 0 or j >= ncol1:
                break

            igen = (i - off_r) // ky
            jgen = (j - off_c) // kx

            # Clamp to coarse grid bounds (handle fine pixels outside coarse extent)
            if igen < 0 or igen >= nrow2 or jgen < 0 or jgen >= ncol2:
                break

            if igen == icur and jgen == jcur:
                ncur += 1
            else:
                d2i[cc] = icur
                d2j[cc] = jcur
                d2n[cc] = ncur
                cc += 1
                ncur = int64(1)
                icur = igen
                jcur = jgen

            step_count += 1
            if step_count > max_steps:
                break

        d2i[cc] = icur
        d2j[cc] = jcur
        d2n[cc] = ncur
        cc += 1

        if cc == 1:
            res[pk] = np.float32(-1.0)
            continue

        # ---- build path-position sorted lookup for coarse cells ----------
        flat_unsorted = np.empty(cc, dtype=np.int64)
        for m in range(cc):
            flat_unsorted[m] = int64(d2i[m]) * ncol2 + int64(d2j[m])

        order = np.argsort(flat_unsorted)
        flat = flat_unsorted[order]
        pos_of_sorted = order.astype(np.int64)

        max_iters = int64(nrow2 * ncol2)

        # ---- first pass: trace each path coarse cell in dir2 -------------
        landed_pos    = np.empty(cc, dtype=np.int64)
        steps_off_path = np.empty(cc, dtype=np.int64)
        for m in range(cc):
            landed_pos[m]     = int64(-1)
            steps_off_path[m] = int64(0)

        for ic in range(cc - 1):
            ci = int64(d2i[ic])
            cj = int64(d2j[ic])
            off_steps = int64(0)

            while True:
                dv = dir2[ci, cj]
                if dv != 255 and dv > 0 and dv <= 128:
                    d_offset = idx[dv]
                    if d_offset == 0 and dv != 1:
                        break
                    ci = ci + di[d_offset]
                    cj = cj + dj[d_offset]
                    if ci < 0 or ci >= nrow2 or cj < 0 or cj >= ncol2:
                        break
                    fv = ci * ncol2 + cj
                    kp = np.searchsorted(flat, fv)
                    if kp < cc and flat[kp] == fv:
                        best = int64(-1)
                        kk = kp
                        while kk < cc and flat[kk] == fv:
                            cand = pos_of_sorted[kk]
                            if use_strict:
                                if cand > ic:
                                    if best == int64(-1) or cand < best:
                                        best = cand
                            else:
                                if cand != ic:
                                    if best == int64(-1):
                                        best = cand
                            kk += 1
                        if best != int64(-1):
                            landed_pos[ic]     = best
                            steps_off_path[ic] = off_steps
                        break
                    off_steps += 1
                    if off_steps > max_iters:
                        break
                else:
                    break

        # ---- second pass: per-cell correctness ---------------------------
        correct = np.zeros(cc, dtype=np.bool_)
        correct[cc - 1] = True

        if cascade:
            for ic in range(cc - 2, -1, -1):
                if landed_pos[ic] != int64(-1):
                    correct[ic] = correct[landed_pos[ic]]
        else:
            for ic in range(cc - 1):
                correct[ic] = landed_pos[ic] != int64(-1)

        # ---- third pass: weighted score ----------------------------------
        inter = 0.0
        total = int64(0)

        for ic in range(cc - 1):
            total += int64(d2n[ic])
            if correct[ic]:
                weight = 1.0 / (1.0 + float(steps_off_path[ic]) * alpha)
                inter += float(d2n[ic]) * weight

        if total > 0:
            res[pk] = np.float32(inter / float(total))
        else:
            res[pk] = np.float32(0.0)

    return res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_flowdir(
    flowdir_fine: Grid,
    flowdir_coarse: Grid,
    seeds: np.ndarray,
    *,
    shuffle: bool = True,
    alpha: float = 0.0,
    strict_upstream: bool = False,
    cascade: bool = False,
) -> np.ndarray:
    """Compare two D8 flow direction grids using per-seed river accuracy.

    The fine path from each seed's headwater to its mouth is traced along
    *flowdir_fine*, mapped to coarse cells, and then each coarse cell is
    scored by whether its direction in *flowdir_coarse* leads back onto the
    fine path.

    The fine→coarse cell mapping is derived from ``check_aligned`` offsets
    ``(kx, ky, off_r, off_c)``, fixing the silent correctness bug present in
    the raw ``d8comp`` script (which used ``shape[0] // nrow2`` without
    accounting for grid registration offsets).

    Parameters
    ----------
    flowdir_fine, flowdir_coarse :
        ``GridType.FlowDir`` grids.  Both must share a common CRS.
        *flowdir_coarse* must be aligned with *flowdir_fine* (integer pixel
        ratio, co-registered origin).
    seeds :
        Structured numpy array as returned by :func:`fdup.utils.river_tree`.
        Required fields:
        ``(mouth_row, mouth_col, headwater_row, headwater_col, ncells, acc,
        length_m)``.
    shuffle :
        Randomise seed processing order for better parallel load balancing.
    alpha :
        Harmonic decay coefficient for off-path deviation penalty.
        ``weight = 1 / (1 + steps_off_path * alpha)``; 0 (default) disables
        the penalty, reproducing the original binary scoring.
    strict_upstream :
        When True, a coarse cell whose dir2 trace lands on a path cell that
        is upstream of the current position is counted as incorrect.
    cascade :
        When True correctness is propagated upstream-to-downstream; forces
        strict-upstream semantics internally.

    Returns
    -------
    np.ndarray float32, shape (len(seeds),)
        Per-seed river accuracy score.  Values in [0, 1]; -1 for degenerate
        seeds whose entire fine path falls within a single coarse cell.

    Raises
    ------
    ValueError
        On type/CRS/alignment mismatch.
    TypeError
        When *seeds* is not a structured array with the expected field names
        (see :func:`fdup.utils.river_tree`).
    """
    check_type(flowdir_fine, GridType.FlowDir)
    check_type(flowdir_coarse, GridType.FlowDir)
    check_crs_match(flowdir_fine, flowdir_coarse)
    kx, ky, off_r, off_c = check_aligned(coarse=flowdir_coarse, fine=flowdir_fine)

    if (
        not hasattr(seeds, "dtype")
        or seeds.dtype.names is None
        or tuple(seeds.dtype.names) != _SEEDS_FIELDS
    ):
        raise TypeError(
            f"seeds.dtype.names must be {_SEEDS_FIELDS!r}, "
            f"got {getattr(seeds, 'dtype', type(seeds))!r}.\n{_SEEDS_HINT}"
        )

    if len(seeds) == 0:
        return np.empty(0, dtype=np.float32)

    # Convert structured seeds to a plain uint32 array expected by the kernel.
    # Kernel layout: [headwater_row, headwater_col, mouth_row, mouth_col, ncells]
    # (raw d8tree col-0 = headwater = trace start; col-2 = mouth = trace end)
    seeds_geom = np.empty((len(seeds), 5), dtype=np.uint32)
    seeds_geom[:, 0] = seeds["headwater_row"]
    seeds_geom[:, 1] = seeds["headwater_col"]
    seeds_geom[:, 2] = seeds["mouth_row"]
    seeds_geom[:, 3] = seeds["mouth_col"]
    seeds_geom[:, 4] = seeds["ncells"]

    return _d8comp(
        flowdir_fine.array,
        flowdir_coarse.array,
        seeds_geom,
        int64(kx),
        int64(ky),
        int64(off_r),
        int64(off_c),
        shuffle,
        alpha,
        strict_upstream,
        cascade,
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup(dtype: object) -> None:  # noqa: ARG001
    """Pre-compile ``_d8comp`` for the warmup registry.

    The *dtype* argument is accepted for registry uniformity; the kernel
    signature does not depend on the FlowAcc dtype because accumulation
    values are not used during scoring.
    """
    # fine 4x4, coarse 2x2, kx=ky=2, off_r=off_c=0
    _dir1 = np.array(
        [
            [255, 255, 255, 255],
            [255,   1,   1, 255],
            [255,   1,   1, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    _dir2 = np.array(
        [
            [255, 255],
            [255,   1],
        ],
        dtype=np.uint8,
    )
    # seeds_geom: [hw_row, hw_col, mouth_row, mouth_col, ncells]
    _seeds_geom = np.array([[1, 1, 1, 2, 2]], dtype=np.uint32)
    _d8comp(
        _dir1, _dir2, _seeds_geom,
        int64(2), int64(2), int64(0), int64(0),
    )
