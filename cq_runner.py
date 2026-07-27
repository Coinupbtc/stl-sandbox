"""Isolated executor for LLM-generated CadQuery scripts.

Usage: cq_runner.py <code.py> <out.stl> <out.step>
Prints exactly one JSON line to stdout as its final output:
  {ok, error?, stats?{verts,faces,watertight,volume_cm3,bbox_mm}, warnings?[]}

Runs as its own process so the FastAPI server survives crashes/hangs/OOM in
generated code; resource limits below bound the blast radius.
"""

import json
import resource
import sys
import traceback


def _limit_resources():
    # NOTE: no RLIMIT_NPROC — it counts every process this *user* owns machine-wide,
    # and this box runs dozens of services; a low cap breaks OpenBLAS thread startup.
    # Thread counts are tamed via OPENBLAS/OMP_NUM_THREADS set by the caller instead.
    resource.setrlimit(resource.RLIMIT_CPU, (75, 75))
    mem = 16 * 1024 ** 3  # virtual address space; OCP+VTK imports alone map several GB
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 ** 2, 256 * 1024 ** 2))


def _fail(msg):
    print(json.dumps({"ok": False, "error": msg[-3000:]}))
    sys.exit(0)


def main():
    code_path, stl_path, step_path = sys.argv[1], sys.argv[2], sys.argv[3]
    _limit_resources()

    with open(code_path) as f:
        code = f.read()

    import math

    import cadquery as cq

    ns = {"cq": cq, "cadquery": cq, "math": math, "__builtins__": __builtins__}
    try:
        exec(compile(code, "<generated>", "exec"), ns)
    except MemoryError:
        _fail("MemoryError: script exceeded the 6GB memory limit")
    except BaseException:
        _fail(traceback.format_exc())

    result = ns.get("result")
    if result is None:
        _fail("Script did not assign anything to the variable `result`")

    # Normalize to a single Shape
    try:
        if isinstance(result, cq.Workplane):
            vals = result.vals()
            solids = [v for v in vals if isinstance(v, cq.Shape)]
            if not solids:
                _fail("`result` Workplane contains no solid geometry")
            shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
        elif isinstance(result, cq.Shape):
            shape = result
        else:
            _fail(f"`result` is {type(result).__name__}, expected cq.Workplane or cq.Shape")
    except BaseException:
        _fail(traceback.format_exc())

    warnings = []
    try:
        if not shape.isValid():
            warnings.append("OpenCascade reports the solid as not fully valid")
    except BaseException:
        pass

    # Precision export: linear/angular deflection suitable for engineering inspection
    # (ISO-style metrology needs fine tessellation; was 0.05/0.2 → sparse junk)
    import os as _os
    lin_tol = float(_os.environ.get("STL_CQ_LINEAR_TOL", "0.01"))
    ang_tol = float(_os.environ.get("STL_CQ_ANGULAR_TOL", "0.05"))
    try:
        cq.exporters.export(shape, stl_path, tolerance=lin_tol, angularTolerance=ang_tol)
        cq.exporters.export(shape, step_path)
    except BaseException:
        _fail("Export failed:\n" + traceback.format_exc())

    # True B-rep bounding box (more accurate than mesh for dimensional certs)
    brep_bbox = None
    try:
        bb = shape.BoundingBox()
        brep_bbox = [
            round(float(bb.xlen), 4),
            round(float(bb.ylen), 4),
            round(float(bb.zlen), 4),
        ]
    except BaseException:
        try:
            # CadQuery Shape API variant
            bb = shape.val().BoundingBox() if hasattr(shape, "val") else None
            if bb:
                brep_bbox = [
                    round(float(bb.xlen), 4),
                    round(float(bb.ylen), 4),
                    round(float(bb.zlen), 4),
                ]
        except BaseException:
            pass

    stats = {}
    try:
        import numpy as np
        import trimesh

        mesh = trimesh.load(stl_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        repaired = False
        if not bool(mesh.is_watertight):
            # Best-effort repair so borderline CAD tessellations still print
            try:
                mesh.merge_vertices()
                mesh.update_faces(mesh.unique_faces())
                mesh.remove_unreferenced_vertices()
                trimesh.repair.fix_holes(mesh)
                trimesh.repair.fix_winding(mesh)
                trimesh.repair.fix_normals(mesh)
                repaired = True
            except BaseException:
                pass
            if not bool(mesh.is_watertight) and float(max(mesh.extents)) < 80:
                # Small parts only: coarse voxel remesh (large-part remesh is slow/huge)
                try:
                    pitch = max(0.6, float(max(mesh.extents)) / 60.0)
                    vox = mesh.voxelized(pitch)
                    mesh = vox.marching_cubes
                    mesh.merge_vertices()
                    repaired = True
                    warnings.append(f"voxel-remeshed @ {pitch:.2f} mm for watertight shell")
                except BaseException as e:
                    warnings.append(f"repair failed: {e}"[:200])
            elif not bool(mesh.is_watertight):
                warnings.append(
                    "mesh not watertight — skipped heavy remesh on large part"
                )

        if repaired and bool(mesh.is_watertight):
            mesh.export(stl_path)
            warnings.append("mesh auto-repaired to watertight")

        watertight = bool(mesh.is_watertight)
        vol = None
        if watertight:
            try:
                vol = round(float(abs(mesh.volume)) / 1000.0, 2)
            except BaseException:
                vol = None
        stats = {
            "verts": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": watertight,
            "volume_cm3": vol,
            "bbox_mm": [round(float(x), 3) for x in mesh.extents],
            "repaired": repaired,
            "tessellation": {"linear_mm": lin_tol, "angular_rad": ang_tol},
        }
        if brep_bbox:
            stats["brep_bbox_mm"] = brep_bbox
            # Prefer B-rep envelope for inspection when available
            stats["bbox_mm"] = [round(float(x), 3) for x in brep_bbox]
        if not watertight:
            warnings.append("Exported mesh is not watertight after repair — not safe to print as-is")
        bbox = np.array(mesh.extents, dtype=float)
        if max(bbox) > 250:
            warnings.append(f"Part is {max(bbox):.0f} mm on its longest axis — check your printer volume")
        if min(bbox) < 1:
            warnings.append(f"Part is only {min(bbox):.2f} mm on its thinnest axis — may be too thin to print")
        if watertight and float(np.prod(bbox)) > 0 and vol is not None:
            fill = float(abs(mesh.volume)) / float(np.prod(bbox))
            stats["fill_ratio"] = round(fill, 3)
            if fill < 0.02:
                warnings.append(
                    "Solid fills under 2% of its bounding box — geometry may be degenerate; "
                    "check the preview matches what you asked for"
                )
            if fill > 0.95 and max(bbox) > 80:
                warnings.append(
                    "Solid fills >95% of its bounding box — if you asked for a case/enclosure, "
                    "this may be a solid brick instead of a hollow shell"
                )
        if stats.get("faces", 0) < 200 and max(bbox) > 40:
            warnings.append(
                f"Only {stats.get('faces')} faces on a large part — geometry may be oversimplified"
            )
    except BaseException:
        warnings.append("Mesh stats unavailable: " + traceback.format_exc(limit=1).strip()[-200:])

    print(json.dumps({"ok": True, "stats": stats, "warnings": warnings}))


if __name__ == "__main__":
    main()
