"""Isolated executor for LLM-generated trimesh sculpt scripts (organic / imagine path).

Usage: organic_runner.py <code.py> <out.stl>
Prints one JSON line: {ok, error?, stats?, warnings?}

Builds stylized printable figurines (cats, people, chickens, …) as solid meshes.
Not photoreal — solid toy/sculpture geometry that actually prints.
"""

from __future__ import annotations

import json
import math
import resource
import sys
import traceback

import numpy as np


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    mem = 8 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 ** 2, 256 * 1024 ** 2))


def _fail(msg: str):
    print(json.dumps({"ok": False, "error": msg[-3000:]}))
    sys.exit(0)


def _capsule(p0, p1, radius, sections=12):
    """Capsule = cylinder + hemisphere caps between two points (mm)."""
    import trimesh

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    height = float(np.linalg.norm(vec))
    if height < 1e-6:
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)

    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # align +Z to vec
    z = np.array([0.0, 0.0, 1.0])
    v = vec / height
    axis = np.cross(z, v)
    axn = np.linalg.norm(axis)
    if axn > 1e-8:
        axis = axis / axn
        angle = math.acos(float(np.clip(np.dot(z, v), -1, 1)))
        rot = trimesh.transformations.rotation_matrix(angle, axis)
        cyl.apply_transform(rot)
    elif np.dot(z, v) < 0:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))

    mid = (p0 + p1) / 2.0
    cyl.apply_translation(mid)

    s0 = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    s0.apply_translation(p0)
    s1 = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    s1.apply_translation(p1)
    return trimesh.util.concatenate([cyl, s0, s1])


def _ellipsoid(radii, center=(0, 0, 0), subdivisions=2):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    s.apply_scale(radii)
    s.apply_translation(center)
    return s


def _cone_part(radius, height, center=(0, 0, 0)):
    import trimesh

    c = trimesh.creation.cone(radius=radius, height=height, sections=16)
    # cone is built with base on XY and apex at +Z height in trimesh
    c.apply_translation(center)
    return c


def _box_part(extents, center=(0, 0, 0)):
    import trimesh

    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def _sphere_part(radius, center=(0, 0, 0), subdivisions=2):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    s.apply_translation(center)
    return s


def _cyl(radius, height, center=(0, 0, 0), sections=16):
    import trimesh

    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_translation(center)
    return m


def _union_all(meshes):
    """Union mesh list into one solid; fallback to concatenate + voxel merge."""
    import trimesh

    meshes = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not meshes:
        raise ValueError("no meshes to union")
    if len(meshes) == 1:
        return meshes[0]

    # Try boolean union chain (manifold3d backend via trimesh)
    acc = meshes[0]
    for m in meshes[1:]:
        try:
            acc = acc.union(m, engine="manifold")
        except Exception:
            try:
                acc = acc.union(m)
            except Exception:
                acc = trimesh.util.concatenate([acc, m])

    if isinstance(acc, list):
        acc = trimesh.util.concatenate(acc)

    # If still multi-body non-watertight, voxel-merge for printability
    if not getattr(acc, "is_watertight", False):
        try:
            pitch = max(float(max(acc.extents)) / 80.0, 0.6)
            vox = acc.voxelized(pitch)
            filled = vox.marching_cubes
            if filled is not None and len(filled.faces) > 0:
                acc = filled
        except Exception:
            pass

    return acc


def _make_printable(mesh, target_height_mm=100.0, add_base=True):
    """Scale, sit on bed (Z=0), optional thin base disc for adhesion."""
    import trimesh

    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("empty mesh")

    # center XY, lift to Z=0
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation([-center[0], -center[1], -bounds[0][2]])

    h = float(mesh.extents[2]) if mesh.extents[2] > 1e-6 else 1.0
    scale = target_height_mm / h
    # clamp ridiculous scales
    scale = float(np.clip(scale, 0.05, 50.0))
    mesh.apply_scale(scale)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    warnings = []
    if add_base:
        # base disc under footprint for adhesion / standing
        r = max(float(max(mesh.extents[0], mesh.extents[1])) * 0.45, 8.0)
        base = trimesh.creation.cylinder(radius=r, height=2.0, sections=32)
        base.apply_translation([0, 0, 1.0])  # sits 0..2
        mesh.apply_translation([0, 0, 2.0])
        try:
            mesh = mesh.union(base, engine="manifold")
        except Exception:
            try:
                mesh = mesh.union(base)
            except Exception:
                mesh = trimesh.util.concatenate([mesh, base])
                warnings.append("base attached by merge (boolean failed)")
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    # final cleanup
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
    except Exception:
        pass

    try:
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    return mesh, warnings


def main():
    code_path, stl_path = sys.argv[1], sys.argv[2]
    _limit_resources()

    with open(code_path) as f:
        code = f.read()

    import trimesh

    ns = {
        "trimesh": trimesh,
        "np": np,
        "numpy": np,
        "math": math,
        "sphere": _sphere_part,
        "ellipsoid": _ellipsoid,
        "capsule": _capsule,
        "cone": _cone_part,
        "box": _box_part,
        "union": _union_all,
        "cylinder": lambda r, h, center=(0, 0, 0), sections=16: _cyl(r, h, center, sections),
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "round": round, "range": range,
            "len": len, "float": float, "int": int, "list": list, "dict": dict,
            "tuple": tuple, "True": True, "False": False, "None": None,
            "enumerate": enumerate, "zip": zip, "sum": sum, "print": print,
        },
    }

    try:
        exec(compile(code, "<organic>", "exec"), ns)
    except MemoryError:
        _fail("MemoryError: organic sculpt exceeded memory limit")
    except BaseException:
        _fail(traceback.format_exc())

    parts = ns.get("parts")
    result = ns.get("result")
    target_h = float(ns.get("target_height_mm", 100.0) or 100.0)
    add_base = bool(ns.get("add_base", True))

    try:
        if parts is not None:
            if not isinstance(parts, (list, tuple)) or not parts:
                _fail("`parts` must be a non-empty list of meshes")
            mesh = _union_all(list(parts))
        elif result is not None:
            mesh = result
            if isinstance(mesh, list):
                mesh = _union_all(mesh)
        else:
            _fail("Script must set `parts` (list of meshes) or `result` (mesh)")
    except BaseException:
        _fail("Assemble failed:\n" + traceback.format_exc())

    warnings = []
    try:
        mesh, w2 = _make_printable(mesh, target_height_mm=target_h, add_base=add_base)
        warnings.extend(w2)
    except BaseException:
        _fail("Printable post-process failed:\n" + traceback.format_exc())

    try:
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(stl_path)) or ".", exist_ok=True)
        mesh.export(stl_path)
    except BaseException:
        _fail("STL export failed:\n" + traceback.format_exc())

    stats = {
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "volume_cm3": round(float(abs(mesh.volume)) / 1000.0, 2) if mesh.is_watertight else None,
        "bbox_mm": [round(float(x), 1) for x in mesh.extents],
    }
    if not mesh.is_watertight:
        warnings.append("Mesh not fully watertight — run a repair in the slicer if needed")
    if min(mesh.extents) < 1.5:
        warnings.append(f"Very thin axis ({min(mesh.extents):.1f} mm) — fragile when printing")
    if max(mesh.extents) > 250:
        warnings.append(f"Part is {max(mesh.extents):.0f} mm on longest axis — check bed size")

    print(json.dumps({"ok": True, "stats": stats, "warnings": warnings, "kind": "organic"}))


if __name__ == "__main__":
    main()
