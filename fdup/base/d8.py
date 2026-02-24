"""D8 flow-direction encoding constants.

Standard D8 power-of-two convention:
1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE.

Provides iteration arrays, decode LUTs (byte -> offset), and an
encode LUT (offset -> byte) used by all upscaling algorithms.
"""

import numpy as np

_D8 = (
    (1,    0,  1),   # E
    (2,    1,  1),   # SE
    (4,    1,  0),   # S
    (8,    1, -1),   # SW
    (16,   0, -1),   # W
    (32,  -1, -1),   # NW
    (64,  -1,  0),   # N
    (128, -1,  1),   # NE
)

# 8-element arrays for iterating over all neighbours
DIR_CODES = np.array([c for c, _, _ in _D8], dtype=np.uint8)
DIR_DROW  = np.array([r for _, r, _ in _D8], dtype=np.int8)
DIR_DCOL  = np.array([c for _, _, c in _D8], dtype=np.int8)
DIR_DIST = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2),
                       1.0, np.sqrt(2), 1.0, np.sqrt(2)], dtype=np.float64)

# Decode LUTs indexed by D8 byte (0..128)
DECODE_DR    = np.zeros(129, dtype=np.int8)
DECODE_DC    = np.zeros(129, dtype=np.int8)
DECODE_VALID = np.zeros(129, dtype=np.bool_)

for _code, _dr, _dc in _D8:
    DECODE_DR[_code]    = _dr
    DECODE_DC[_code]    = _dc
    DECODE_VALID[_code] = True

# Encode LUT: (dr, dc) -> D8 byte via ENCODE_DIR[dr + 1, dc + 1]
ENCODE_DIR = np.zeros((3, 3), dtype=np.uint8)
for _code, _dr, _dc in _D8:
    ENCODE_DIR[_dr + 1, _dc + 1] = _code

del _code, _dr, _dc
