"""Programmatic API example for fdupr.

Edit the input/output paths below before running:
    python examples/api_demo.py
"""

import time

from fdup.upscalers import COTAT, DMM, NSA


def _print_elapsed(start_time):
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"Processed in {h}h {m}m {s:.2f}s")


def main():
    input_flowacc = r"C:\path\to\flowacc.tif"
    input_flowdir = r"C:\path\to\flowdir.tif"

    # DMM example
    dmm = DMM()
    dmm.load_flowacc(input_flowacc)
    start_time = time.time()
    dmm.upscale(k=20)
    _print_elapsed(start_time)
    dmm.save(r"C:\path\to\DMM_output.tif")

    # NSA example
    nsa = NSA()
    nsa.load_flowacc(input_flowacc)
    start_time = time.time()
    nsa.upscale(k=20)
    _print_elapsed(start_time)
    nsa.save(r"C:\path\to\NSA_output.tif")

    # COTAT example
    cotat = COTAT()
    cotat.load_flowdir(input_flowdir)
    cotat.load_flowacc(input_flowacc)
    start_time = time.time()
    cotat.upscale(k=20, area_threshold=5000)
    _print_elapsed(start_time)
    cotat.save(r"C:\path\to\COTAT_output.tif")


if __name__ == "__main__":
    main()
