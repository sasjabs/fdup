import numpy as np

_FLOWACC_DTYPES = (np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)


def warmup() -> None:
    """Pre-compile all numba kernels for every supported FlowAcc dtype."""
    from fdup.upscalers import dmm, nsa, cotat
    # Import _warmup directly from each submodule to avoid the naming conflict
    # between fdup.utils.d8 (module) and d8 (function) in the utils namespace.
    from fdup.utils.d8 import _warmup as _d8_warmup
    from fdup.utils.flowacc import _warmup as _flowacc_warmup
    from fdup.utils.pour import _warmup as _pour_warmup
    from fdup.utils.watershed import _warmup as _watershed_warmup
    from fdup.utils.tree import _warmup as _tree_warmup
    from fdup.evals.fdcomp import _warmup as _fdcomp_warmup
    from fdup.evals.huac import _warmup as _huac_warmup
    from fdup.evals.wscomp import _warmup as _wscomp_warmup
    from fdup.evals.windrose import _warmup as _windrose_warmup

    for dtype in _FLOWACC_DTYPES:
        dmm._warmup(dtype)
        nsa._warmup(dtype)
        cotat._warmup(dtype)
        _flowacc_warmup(dtype)
        _pour_warmup(dtype)
        _tree_warmup(dtype)
        _fdcomp_warmup(dtype)
        _huac_warmup(dtype)

    _d8_warmup(None)        # DEM-dtype parametric; full matrix compiled internally
    _watershed_warmup(None) # BFS kernel is dtype-agnostic
    _wscomp_warmup(None)    # no numba kernel; no-op stub
    _windrose_warmup(None)  # FlowDir is always uint8; kernel does not specialise
