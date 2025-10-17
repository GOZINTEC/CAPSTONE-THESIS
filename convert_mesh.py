#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np
import trimesh


def load_as_mesh(input_path: str) -> trimesh.Trimesh:
    """Load a 3D asset as a single Trimesh mesh.

    Handles files that may contain multiple geometries by concatenating them.
    """
    loaded = trimesh.load(input_path, force=None, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    # If a Scene, concatenate geometries
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("Loaded scene contains no geometry")
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No mesh geometries found in scene")
        return trimesh.util.concatenate(meshes)

    # If it's a Path2D or other type, try to convert
    try:
        return loaded.to_mesh()  # type: ignore[attr-defined]
    except Exception as exc:
        raise TypeError(f"Unsupported geometry type: {type(loaded)}") from exc


def apply_transforms(
    mesh: trimesh.Trimesh,
    center: bool,
    scale: Optional[float],
    z_up_to_y_up: bool,
) -> None:
    if center:
        centroid = mesh.centroid
        mesh.apply_translation(-centroid)
    if scale is not None and scale > 0:
        mesh.apply_scale(scale)
    if z_up_to_y_up:
        # Rotate -90 deg about X to convert Z-up to Y-up (glTF convention)
        angle = -np.pi / 2.0
        rot = trimesh.transformations.rotation_matrix(angle, [1.0, 0.0, 0.0])
        mesh.apply_transform(rot)


def maybe_repair(mesh: trimesh.Trimesh, do_repair: bool) -> None:
    if not do_repair:
        return
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.broken_faces(mesh)
    except Exception:
        # Best-effort; proceed even if repair fails
        pass


def infer_output_path(input_path: str, fmt: str) -> str:
    root, _ = os.path.splitext(input_path)
    ext = f".{fmt.lower()}"
    return root + ext


def export_mesh(mesh: trimesh.Trimesh, output_path: str, fmt: str, stl_ascii: bool) -> None:
    fmt = fmt.lower()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "stl":
        encoding = "ascii" if stl_ascii else "binary"
        mesh.export(output_path, file_type="stl", encoding=encoding)
    elif fmt in {"glb", "gltf", "obj", "ply"}:
        mesh.export(output_path, file_type=fmt)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a mesh to viewer-friendly formats (GLB/GLTF/OBJ/STL) and optionally "
            "center/scale it so it appears in common 3D viewers."
        )
    )
    parser.add_argument("input", help="Input 3D file (e.g., .stl, .obj, .ply, .glb, .gltf)")
    parser.add_argument(
        "--to",
        dest="to_format",
        default="glb",
        choices=["glb", "gltf", "obj", "stl", "ply"],
        help="Output format (default: glb)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        help="Output file path (default: input name with new extension)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Uniform scale to apply before export (e.g., 0.001 if input is in mm "
            "and you want meters)."
        ),
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the model at the origin before export",
    )
    parser.add_argument(
        "--z-up-to-y-up",
        action="store_true",
        help="Rotate -90° about X (Z-up -> Y-up) for glTF/GLB viewers",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Attempt to repair normals/holes/inversions before export",
    )
    parser.add_argument(
        "--stl-ascii",
        action="store_true",
        help="Export STL as ASCII (default STL is binary)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_fmt = args.to_format.lower()
    output_path = args.output or infer_output_path(input_path, output_fmt)

    try:
        mesh = load_as_mesh(input_path)
        apply_transforms(
            mesh,
            center=args.center,
            scale=args.scale,
            z_up_to_y_up=args.z_up_to_y_up,
        )
        maybe_repair(mesh, args.repair)
        export_mesh(mesh, output_path, output_fmt, stl_ascii=args.stl_ascii)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {output_fmt.upper()} to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
