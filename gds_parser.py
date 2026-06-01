"""
gds_parser.py
=============
GDS file loader.
Priority: gdstk → gdspy → minimal built-in binary parser (no dependencies).
"""

import numpy as np
import struct
import os


def _try_gdstk(path: str):
    import gdstk
    return gdstk.read_gds(path)


def _try_gdspy(path: str):
    import gdspy
    return gdspy.GdsLibrary(infile=path)


def list_layers(path: str) -> list:
    """Returns a sorted list of layer numbers found in the GDS file."""
    try:
        lib = _try_gdstk(path)
        layers = {poly.layer for cell in lib.cells for poly in cell.polygons}
        return sorted(layers)
    except ImportError:
        pass

    try:
        lib = _try_gdspy(path)
        layers = {poly.layer for cell in lib.cells.values() for poly in cell.polygons}
        return sorted(layers)
    except ImportError:
        pass

    return _minimal_list_layers(path)


def load_gds_polygons(path: str, layer: int = None) -> list:
    """
    Returns polygons on the specified layer as a list of np.ndarray (N, 2).
    Units are nm (converted from GDS DBU).
    If layer=None, returns all layers.
    """
    try:
        lib = _try_gdstk(path)
        polys = []
        for cell in lib.cells:
            for poly in cell.polygons:
                if layer is None or poly.layer == layer:
                    polys.append(np.array(poly.points) * 1e3)  # µm → nm
        return polys
    except ImportError:
        pass

    try:
        lib = _try_gdspy(path)
        polys = []
        for cell in lib.cells.values():
            for poly in cell.polygons:
                if layer is None or poly.layer == layer:
                    polys.append(np.array(poly.points) * 1e3)
        return polys
    except ImportError:
        pass

    return _minimal_load_polygons(path, layer)


# ─── Minimal built-in GDS binary parser ───────────────────────────
# GDS format: records of [length(2B), type(1B), datatype(1B), data...]
# BOUNDARY records (0x08) define polygons.

def _read_records(path: str):
    records = []
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i < len(data) - 3:
        length = struct.unpack(">H", data[i:i+2])[0]
        if length < 4:
            break
        rec_type = data[i+2]
        dtype    = data[i+3]
        body     = data[i+4:i+length]
        records.append((rec_type, dtype, body))
        i += length
    return records


def _minimal_list_layers(path: str) -> list:
    records = _read_records(path)
    layers = set()
    for rec_type, dtype, body in records:
        if rec_type == 0x0D and len(body) >= 2:  # LAYER record
            layer = struct.unpack(">H", body[:2])[0]
            layers.add(layer)
    return sorted(layers)


def _minimal_load_polygons(path: str, target_layer=None) -> list:
    """
    Parses GDS binary directly to extract BOUNDARY polygons.
    DBU is read from the UNITS record (defaults to 1 nm if absent).
    """
    records = _read_records(path)

    # Read DBU from UNITS record
    dbu_nm = 1.0
    for rec_type, dtype, body in records:
        if rec_type == 0x03 and len(body) >= 16:  # UNITS record
            dbu_per_meter = struct.unpack(">d", body[8:16])[0]
            dbu_nm = dbu_per_meter * 1e9 if dbu_per_meter else 1.0
            break

    polys = []
    in_boundary  = False
    current_layer = None
    current_xy   = []

    for rec_type, dtype, body in records:
        if rec_type == 0x08:    # BOUNDARY
            in_boundary   = True
            current_layer = None
            current_xy    = []
        elif rec_type == 0x11:  # ENDEL
            if in_boundary and current_xy:
                pts = np.array(current_xy, dtype=float) * dbu_nm
                if target_layer is None or current_layer == target_layer:
                    polys.append(pts)
            in_boundary = False
        elif in_boundary:
            if rec_type == 0x0D and len(body) >= 2:  # LAYER
                current_layer = struct.unpack(">H", body[:2])[0]
            elif rec_type == 0x10:  # XY
                n = len(body) // 8
                for j in range(n - 1):  # last point == first point, skip
                    x = struct.unpack(">i", body[j*8:j*8+4])[0]
                    y = struct.unpack(">i", body[j*8+4:j*8+8])[0]
                    current_xy.append([x, y])

    return polys
