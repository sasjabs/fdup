"""Flow-direction upscalers (DMM, NSA, COTAT).

Each public function accepts ``k`` as a positive integer (isotropic,
equivalent to ``(k, k)``) or a ``(kx, ky)`` tuple.  ``kx`` is the X-axis
/ column scale (``transform.a``); ``ky`` is the Y-axis / row scale
(``transform.e``).  A coarse cell covers ``ky`` fine rows by ``kx`` fine
columns.  DMM additionally requires both axes to be even.
"""

from fdup.upscalers.cotat import COTAT
from fdup.upscalers.dmm import DMM
from fdup.upscalers.nsa import NSA

__all__ = ["COTAT", "DMM", "NSA"]
