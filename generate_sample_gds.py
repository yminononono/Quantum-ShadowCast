"""
generate_sample_gds.py
======================
Generates sample GDS files for testing (no external dependencies required).

Output files:
  - dolan_bridge_sample.gds  : Dolan bridge pattern (Layer 1)
  - manhattan_sample.gds     : Manhattan junction pattern (Layer 1)
"""

import struct
import numpy as np
import os


def write_gds_minimal(filename: str, polygons: list, layer: int = 1):
    """
    Writes a minimal GDS file without any external libraries.
    polygons: list of np.ndarray (N, 2), units in nm (DBU = 1 nm)
    """
    def record(rec_type, dtype, data=b""):
        length = 4 + len(data)
        return struct.pack(">HBB", length, rec_type, dtype) + data

    def int2(val):  return struct.pack(">h", int(val))
    def int4(val):  return struct.pack(">i", int(val))
    def real8(val): return struct.pack(">d", float(val))

    buf = bytearray()
    buf += record(0x00, 0x02, int2(600))           # HEADER
    buf += record(0x01, 0x02, int2(0) * 12)        # BGNLIB
    buf += record(0x02, 0x06, b"SAMPLE\x00")       # LIBNAME
    buf += record(0x03, 0x05, real8(0.001) + real8(1e-9))  # UNITS (DBU=1nm)
    buf += record(0x05, 0x02, int2(0) * 12)        # BGNSTR
    buf += record(0x06, 0x06, b"TOP\x00")          # STRNAME

    for poly in polygons:
        buf += record(0x08, 0x00)                       # BOUNDARY
        buf += record(0x0D, 0x02, int2(layer))          # LAYER
        buf += record(0x0E, 0x02, int2(0))              # DATATYPE
        pts_nm = poly.astype(int)
        xy_data = b""
        for pt in pts_nm:
            xy_data += int4(pt[0]) + int4(pt[1])
        xy_data += int4(pts_nm[0][0]) + int4(pts_nm[0][1])  # close polygon
        buf += record(0x10, 0x03, xy_data)              # XY
        buf += record(0x11, 0x00)                       # ENDEL

    buf += record(0x07, 0x00)   # ENDSTR
    buf += record(0x04, 0x00)   # ENDLIB

    with open(filename, "wb") as f:
        f.write(buf)

    print(f"Written: {filename}  ({len(polygons)} polygons, layer {layer})")


def dolan_bridge_polygons():
    """
    Dolan bridge pattern (units: nm).

    Components:
      - Left electrode (large rectangle)
      - Right electrode (large rectangle)
      - Bridge (suspended thin bar)
      - Left anchor (connects bridge to left electrode)
      - Right anchor (connects bridge to right electrode)
    """
    return [
        # Left electrode: x=-2000..-300, y=-500..500
        np.array([[-2000, -500], [-300, -500], [-300,  500], [-2000,  500]]),
        # Right electrode: x=300..2000, y=-500..500
        np.array([[ 300, -500], [2000, -500], [2000,  500], [  300,  500]]),
        # Bridge body: x=-200..200, y=-80..80
        np.array([[ -200,  -80], [ 200,  -80], [ 200,   80], [ -200,   80]]),
        # Left anchor: x=-300..-200, y=-200..200
        np.array([[ -300, -200], [-200, -200], [-200,  200], [ -300,  200]]),
        # Right anchor: x=200..300, y=-200..200
        np.array([[  200, -200], [ 300, -200], [ 300,  200], [  200,  200]]),
    ]


def manhattan_polygons():
    """
    Manhattan junction pattern (units: nm).

    Components:
      - Left electrode
      - Right electrode
      (The gap between them forms the junction.)
    """
    gap = 150  # electrode gap [nm]
    return [
        # Left electrode
        np.array([[-2000, -400], [-gap//2, -400], [-gap//2, 400], [-2000,  400]]),
        # Right electrode
        np.array([[ gap//2, -400], [2000, -400], [2000,  400], [ gap//2,  400]]),
    ]


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "sample_gds")
    os.makedirs(out_dir, exist_ok=True)

    write_gds_minimal(
        os.path.join(out_dir, "dolan_bridge_sample.gds"),
        dolan_bridge_polygons(), layer=1
    )
    write_gds_minimal(
        os.path.join(out_dir, "manhattan_sample.gds"),
        manhattan_polygons(), layer=1
    )

    print("\nSample GDS files generated:")
    print(f"  {out_dir}/dolan_bridge_sample.gds")
    print(f"  {out_dir}/manhattan_sample.gds")
