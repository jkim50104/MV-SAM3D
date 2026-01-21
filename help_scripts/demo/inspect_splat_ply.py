#!/usr/bin/env python3
"""
inspect_splat_ply.py

Usage:
  python inspect_splat_ply.py splat.ply
  python inspect_splat_ply.py splat.ply --head 5
  python inspect_splat_ply.py splat.ply --stats-only

Works for:
  - ASCII PLY
  - binary_little_endian PLY
  - binary_big_endian PLY

Focuses on 'element vertex' properties (3D Gaussian splats often store one splat per vertex).
"""

from __future__ import annotations

import argparse
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


PLY_SCALAR_TYPES: Dict[str, Tuple[str, np.dtype]] = {
    # PLY name -> (struct format char without endian, numpy dtype)
    "char": ("b", np.int8),
    "int8": ("b", np.int8),
    "uchar": ("B", np.uint8),
    "uint8": ("B", np.uint8),
    "short": ("h", np.int16),
    "int16": ("h", np.int16),
    "ushort": ("H", np.uint16),
    "uint16": ("H", np.uint16),
    "int": ("i", np.int32),
    "int32": ("i", np.int32),
    "uint": ("I", np.uint32),
    "uint32": ("I", np.uint32),
    "float": ("f", np.float32),
    "float32": ("f", np.float32),
    "double": ("d", np.float64),
    "float64": ("d", np.float64),
}

# list properties exist in PLY, but gaussian-splat PLYs are almost always scalar-only on vertices.
@dataclass
class PlyProperty:
    name: str
    kind: str  # "scalar" or "list"
    scalar_type: Optional[str] = None
    count_type: Optional[str] = None
    item_type: Optional[str] = None


@dataclass
class PlyElement:
    name: str
    count: int
    properties: List[PlyProperty]


@dataclass
class PlyHeader:
    format: str
    version: str
    elements: List[PlyElement]
    header_bytes: int  # file offset where data begins
    comments: List[str]


def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def parse_header(fp) -> PlyHeader:
    comments: List[str] = []
    elements: List[PlyElement] = []

    first = fp.readline()
    if not first:
        raise ValueError("Empty file")
    if first.strip() != b"ply":
        raise ValueError("Not a PLY file (missing 'ply' magic)")

    fmt = None
    ver = None
    current_element: Optional[PlyElement] = None

    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")

        s = line.decode("utf-8", errors="replace").strip()
        if s == "end_header":
            header_bytes = fp.tell()
            break

        if not s:
            continue

        parts = s.split()
        tag = parts[0].lower()

        if tag == "format":
            # format ascii 1.0
            if len(parts) != 3:
                raise ValueError(f"Malformed format line: {s}")
            fmt, ver = parts[1], parts[2]

        elif tag == "comment":
            comments.append(s[len("comment"):].strip())

        elif tag == "element":
            # element vertex 12345
            if len(parts) != 3:
                raise ValueError(f"Malformed element line: {s}")
            if current_element is not None:
                elements.append(current_element)
            current_element = PlyElement(name=parts[1], count=int(parts[2]), properties=[])

        elif tag == "property":
            if current_element is None:
                raise ValueError("Property declared before any element")

            # property float x
            # property list uchar int vertex_indices
            if len(parts) < 3:
                raise ValueError(f"Malformed property line: {s}")

            if parts[1].lower() == "list":
                if len(parts) != 5:
                    raise ValueError(f"Malformed list property line: {s}")
                count_type = parts[2].lower()
                item_type = parts[3].lower()
                name = parts[4]
                current_element.properties.append(
                    PlyProperty(
                        name=name,
                        kind="list",
                        count_type=count_type,
                        item_type=item_type,
                    )
                )
            else:
                scalar_type = parts[1].lower()
                name = parts[2]
                current_element.properties.append(
                    PlyProperty(
                        name=name,
                        kind="scalar",
                        scalar_type=scalar_type,
                    )
                )
        else:
            # ignore other header directives (obj_info, etc.)
            pass

    if current_element is not None:
        elements.append(current_element)

    if fmt is None or ver is None:
        raise ValueError("PLY header missing format/version")

    return PlyHeader(format=fmt, version=ver, elements=elements, header_bytes=header_bytes, comments=comments)


def numpy_dtype_for_vertex(props: List[PlyProperty], endian: str) -> np.dtype:
    fields = []
    for p in props:
        if p.kind != "scalar":
            raise NotImplementedError(
                f"Vertex has list property '{p.name}'. This script focuses on scalar vertex properties."
            )
        if p.scalar_type not in PLY_SCALAR_TYPES:
            raise ValueError(f"Unsupported PLY scalar type: {p.scalar_type}")
        _, npdt = PLY_SCALAR_TYPES[p.scalar_type]
        fields.append((p.name, np.dtype(npdt).newbyteorder(endian)))
    return np.dtype(fields)


def load_vertices(path: str, header: PlyHeader) -> Tuple[np.ndarray, PlyElement]:
    vertex_el = next((e for e in header.elements if e.name == "vertex"), None)
    if vertex_el is None:
        raise ValueError("No 'element vertex' in PLY")

    fmt = header.format.lower()
    if fmt == "ascii":
        # Read ASCII vertex lines
        # Warning: slow for huge files, but okay for inspection.
        vertices = []
        with open(path, "rb") as fp:
            fp.seek(header.header_bytes)
            for _ in range(vertex_el.count):
                line = fp.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading ASCII vertex data")
                vals = line.decode("utf-8", errors="replace").strip().split()
                if len(vals) != len(vertex_el.properties):
                    raise ValueError(
                        f"ASCII vertex line has {len(vals)} values but expected {len(vertex_el.properties)}"
                    )
                vertices.append(vals)

        # Build dtype and cast
        dtype = numpy_dtype_for_vertex(vertex_el.properties, endian="=")  # native
        arr = np.empty(vertex_el.count, dtype=dtype)

        for j, prop in enumerate(vertex_el.properties):
            if prop.scalar_type is None:
                raise ValueError("Missing scalar_type")
            _, npdt = PLY_SCALAR_TYPES[prop.scalar_type]
            col = np.array([v[j] for v in vertices], dtype=npdt)
            arr[prop.name] = col

        return arr, vertex_el

    elif fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = numpy_dtype_for_vertex(vertex_el.properties, endian=endian)
        with open(path, "rb") as fp:
            fp.seek(header.header_bytes)
            data = fp.read(dtype.itemsize * vertex_el.count)
            if len(data) != dtype.itemsize * vertex_el.count:
                raise ValueError("Unexpected EOF while reading binary vertex block")
            arr = np.frombuffer(data, dtype=dtype, count=vertex_el.count)
        return arr, vertex_el

    else:
        raise ValueError(f"Unsupported PLY format: {header.format}")


def describe_field(name: str, a: np.ndarray) -> str:
    if a.size == 0:
        return f"{name}: empty"
    # Handle non-finite
    finite = np.isfinite(a) if np.issubdtype(a.dtype, np.floating) else np.ones_like(a, dtype=bool)
    af = a[finite] if finite.any() else a

    if np.issubdtype(a.dtype, np.floating):
        mn = float(np.nanmin(af)) if af.size else float("nan")
        mx = float(np.nanmax(af)) if af.size else float("nan")
        mean = float(np.nanmean(af)) if af.size else float("nan")
        p1 = float(np.nanpercentile(af, 1)) if af.size else float("nan")
        p50 = float(np.nanpercentile(af, 50)) if af.size else float("nan")
        p99 = float(np.nanpercentile(af, 99)) if af.size else float("nan")
        n_nan = int(np.isnan(a).sum())
        n_inf = int(np.isinf(a).sum())
        return (f"{name}: dtype={a.dtype}, min={mn:.6g}, max={mx:.6g}, mean={mean:.6g}, "
                f"p1={p1:.6g}, p50={p50:.6g}, p99={p99:.6g}, nan={n_nan}, inf={n_inf}")
    else:
        mn = int(af.min()) if af.size else 0
        mx = int(af.max()) if af.size else 0
        # unique count can be expensive; cap
        uniq = None
        if a.size <= 2_000_000:
            uniq = int(np.unique(a).size)
        return f"{name}: dtype={a.dtype}, min={mn}, max={mx}" + (f", unique={uniq}" if uniq is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path", help="Path to splat.ply (or any PLY)")
    ap.add_argument("--head", type=int, default=0, help="Print first N rows (default 0)")
    ap.add_argument("--stats-only", action="store_true", help="Skip printing first rows even if --head is set")
    args = ap.parse_args()

    path = args.ply_path
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    with open(path, "rb") as fp:
        header = parse_header(fp)

    print("=== PLY HEADER ===")
    print(f"Format: {header.format} {header.version}")
    print(f"Data starts at byte offset: {header.header_bytes}")
    if header.comments:
        print("Comments:")
        for c in header.comments[:20]:
            print(f"  - {c}")
        if len(header.comments) > 20:
            print(f"  ... ({len(header.comments)-20} more)")

    print("\nElements:")
    for e in header.elements:
        print(f"  - {e.name}: count={e.count}, properties={len(e.properties)}")
        for p in e.properties[:40]:
            if p.kind == "scalar":
                print(f"      scalar {p.scalar_type} {p.name}")
            else:
                print(f"      list {p.count_type} {p.item_type} {p.name}")
        if len(e.properties) > 40:
            print(f"      ... ({len(e.properties)-40} more properties)")

    # Load vertices
    verts, v_el = load_vertices(path, header)
    print("\n=== VERTEX TABLE ===")
    print(f"Vertex count: {verts.shape[0]}")
    print(f"Vertex properties ({len(verts.dtype.names)}):")
    print("  " + ", ".join(verts.dtype.names))

    # Print head
    if args.head > 0 and not args.stats_only:
        n = min(args.head, verts.shape[0])
        print(f"\nFirst {n} vertices:")
        # pretty print subset
        names = verts.dtype.names
        for i in range(n):
            row = {k: verts[k][i].item() for k in names}
            print(f"[{i}] {row}")

    # Stats on likely-important fields
    names = set(verts.dtype.names)

    print("\n=== QUICK STATS (common Gaussian-splat fields) ===")

    # Positions
    for k in ("x", "y", "z"):
        if k in names:
            print(describe_field(k, verts[k].astype(np.float64, copy=False)))

    # Opacity: often logit
    if "opacity" in names:
        op = verts["opacity"].astype(np.float64, copy=False)
        print(describe_field("opacity(raw)", op))
        # decode if it looks like logits: wide range (e.g. [-10, 10]) instead of [0,1]
        if np.nanmin(op) < -1.5 or np.nanmax(op) > 1.5:
            alpha = sigmoid(op)
            print(describe_field("opacity(sigmoid)", alpha))
        else:
            print("opacity seems already in [0,1]-ish range (not obviously logits).")

    # Scales: often log-scales
    scale_keys = [k for k in ("scale_0", "scale_1", "scale_2") if k in names]
    if scale_keys:
        for k in scale_keys:
            sc = verts[k].astype(np.float64, copy=False)
            print(describe_field(f"{k}(raw)", sc))
        raw = np.stack([verts[k].astype(np.float64, copy=False) for k in scale_keys], axis=1)
        if np.nanmin(raw) < -1.0 or np.nanmax(raw) > 1.0:
            exp_sc = np.exp(np.clip(raw, -60, 60))
            for i, k in enumerate(scale_keys):
                print(describe_field(f"{k}(exp)", exp_sc[:, i]))
        else:
            print("scale_* seems already small-range; might not be log-scale, but exp() stats are still shown above if needed.")

    # Rotation quaternion
    rot_keys = [k for k in ("rot_0", "rot_1", "rot_2", "rot_3") if k in names]
    if rot_keys:
        for k in rot_keys:
            print(describe_field(k, verts[k].astype(np.float64, copy=False)))
        if len(rot_keys) == 4:
            q = np.stack([verts[k].astype(np.float64, copy=False) for k in rot_keys], axis=1)
            qnorm = np.linalg.norm(q, axis=1)
            print(describe_field("rot_quat_norm", qnorm))

    # SH base color (DC)
    dc_keys = [k for k in ("f_dc_0", "f_dc_1", "f_dc_2") if k in names]
    if dc_keys:
        for k in dc_keys:
            print(describe_field(k, verts[k].astype(np.float64, copy=False)))
        dc = np.stack([verts[k].astype(np.float64, copy=False) for k in dc_keys], axis=1)
        # Many pipelines store SH DC in a way that maps to RGB after some transform.
        # For a rough sanity check, show a naive clamp and also show mean.
        dc_clamped = np.clip(dc, 0.0, 1.0)
        print(describe_field("f_dc_rgb_naive_clamped_mean", dc_clamped.mean(axis=1)))

    # Any obvious IDs
    for k in ("object_id", "mask_id", "instance_id", "label", "class_id"):
        if k in names:
            print(describe_field(k, verts[k]))

    # List "other" fields that look like SH rest
    rest = sorted([k for k in names if k.startswith("f_rest")])
    if rest:
        print(f"\nFound {len(rest)} f_rest_* fields (higher-order spherical harmonics).")
        # show a few
        for k in rest[:10]:
            print(describe_field(k, verts[k].astype(np.float64, copy=False)))
        if len(rest) > 10:
            print(f"... (showing 10/{len(rest)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
