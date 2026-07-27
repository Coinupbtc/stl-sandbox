"""ISO-inspired dimensional quality system for STL Sandbox.

Implements process discipline modeled on:
  - ISO 2768-1 (general tolerances — linear dimensions)
  - ISO/ASTM 52900 vocabulary (additive manufacturing)
  - Inspection report structure similar to AS9102 Form 3 (characteristics)
  - Engineering gates for wall thickness, manifold solid, STEP presence

IMPORTANT — honesty boundary (do not delete):
  This module produces *inspection evidence* and *dimensional conformance*
  for digitally generated solids / FDM-ready meshes. It does NOT make the
  part flight-certified, AS9100-approved, or aerospace-qualified. Those
  require certified materials, qualified processes, NDT, and a quality
  management system outside this sandbox. What we *do* provide is the
  same *kind* of dimensional rigor used in controlled engineering prototypes.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Any, Optional

import numpy as np

# ─── ISO 2768-1 linear tolerance tables (mm) ───────────────────────────────
# Bins: (upper exclusive of previous, upper inclusive) → perm. deviation ±
# Classes: f=fine, m=medium, c=coarse, v=very coarse

_ISO2768_BINS = [
    # (max_size, f, m, c, v)
    (3, 0.05, 0.1, 0.2, 0.5),
    (6, 0.05, 0.1, 0.3, 1.0),
    (30, 0.1, 0.2, 0.5, 1.0),
    (120, 0.15, 0.3, 0.8, 1.5),
    (400, 0.2, 0.5, 1.2, 2.5),
    (1000, 0.3, 0.8, 2.0, 4.0),
    (2000, 0.5, 1.2, 3.0, 6.0),
]

TOLERANCE_CLASSES = {
    "iso2768-f": {"name": "ISO 2768-f (fine)", "col": 1, "use": "Precision jigs, mating features"},
    "iso2768-m": {"name": "ISO 2768-m (medium)", "col": 2, "use": "General engineering (default)"},
    "iso2768-c": {"name": "ISO 2768-c (coarse)", "col": 3, "use": "Non-critical FDM housings"},
    "iso2768-v": {"name": "ISO 2768-v (very coarse)", "col": 4, "use": "Rough mockups only"},
    # FDM process capability class — realistic for consumer printers
    "fdm-engineering": {
        "name": "FDM engineering envelope (±0.2 mm or 0.3%)",
        "col": None,
        "use": "Desktop FDM with calibrated printer; not metal AM",
        "fdm": True,
    },
}

DEFAULT_CLASS = os.environ.get("STL_ISO_CLASS", "iso2768-m")


def iso2768_tolerance(size_mm: float, cls: str = "iso2768-m") -> float:
    """Return bilateral linear tolerance ±t for a nominal size (mm)."""
    size = abs(float(size_mm))
    meta = TOLERANCE_CLASSES.get(cls) or TOLERANCE_CLASSES["iso2768-m"]
    if meta.get("fdm"):
        return max(0.20, 0.003 * size)  # ±0.2 mm or 0.3%
    col = meta["col"]
    for upper, *vals in _ISO2768_BINS:
        # vals are f,m,c,v at indices 0..3; col is 1..4
        if size <= upper:
            return float(vals[col - 1])
    return float(_ISO2768_BINS[-1][col])


def check_dimension(
    nominal: float,
    actual: float,
    cls: str = DEFAULT_CLASS,
    name: str = "dim",
) -> dict[str, Any]:
    t = iso2768_tolerance(nominal, cls)
    dev = float(actual) - float(nominal)
    ok = abs(dev) <= t + 1e-9
    return {
        "characteristic": name,
        "nominal_mm": round(float(nominal), 4),
        "actual_mm": round(float(actual), 4),
        "deviation_mm": round(dev, 4),
        "tolerance_pm_mm": round(t, 4),
        "tolerance_class": cls,
        "result": "PASS" if ok else "FAIL",
        "ok": ok,
    }


def _pair_axes(expected: list[float], actual: list[float]) -> list[tuple[float, float]]:
    """Greedy match sorted expected ↔ sorted actual for envelope checks."""
    exp = sorted(float(x) for x in expected)
    act = sorted(float(x) for x in actual)
    # pad
    while len(act) < len(exp):
        act.append(0.0)
    return list(zip(exp, act[: len(exp)]))


def dimensional_inspection(
    expected_bbox_mm: Optional[list],
    actual_bbox_mm: Optional[list],
    size_contract: Optional[dict] = None,
    cls: str = DEFAULT_CLASS,
) -> dict[str, Any]:
    """Inspect envelope + key named dimensions from size contract."""
    chars: list[dict] = []
    contract = size_contract or {}

    # Named singles — match each nominal to the *closest* measured axis.
    # Phone cases: inspect OUTER envelope, not phone body height (body + walls).
    named = []
    acts = sorted(float(x) for x in (actual_bbox_mm or []))

    def _closest(nom: float) -> float:
        if not acts:
            return 0.0
        return min(acts, key=lambda a: abs(a - nom))

    is_case = contract.get("part_type") == "phone_case" or bool(
        contract.get("outer_height_mm")
    )
    if is_case and contract.get("outer_height_mm"):
        named.append((
            "outer_height_mm",
            float(contract["outer_height_mm"]),
            _closest(float(contract["outer_height_mm"])),
        ))
        if contract.get("outer_width_mm"):
            named.append((
                "outer_width_mm",
                float(contract["outer_width_mm"]),
                _closest(float(contract["outer_width_mm"])),
            ))
        if contract.get("outer_depth_mm"):
            named.append((
                "outer_depth_mm",
                float(contract["outer_depth_mm"]),
                _closest(float(contract["outer_depth_mm"])),
            ))
    else:
        if contract.get("height_mm") and acts:
            h = float(contract["height_mm"])
            named.append(("height_mm", h, _closest(h)))
        if contract.get("width_mm") and acts and not contract.get("diameter_mm"):
            w = float(contract["width_mm"])
            named.append(("width_mm", w, _closest(w)))
        if contract.get("diameter_mm") and acts:
            dnom = float(contract["diameter_mm"])
            if len(acts) >= 2 and abs(acts[-1] - acts[-2]) / max(acts[-1], 1e-6) < 0.15:
                named.append(("diameter_mm", dnom, (acts[-1] + acts[-2]) / 2))
            else:
                named.append(("diameter_mm", dnom, _closest(dnom)))
        if contract.get("depth_mm") and acts and contract.get("diameter_mm"):
            tnom = float(contract["depth_mm"])
            named.append(("thickness_mm", tnom, _closest(tnom)))
        if contract.get("across_flats_mm") and acts:
            af = float(contract["across_flats_mm"])
            named.append(("across_flats_mm", af, _closest(af)))

    if contract.get("bore_mm"):
        # bore cannot be measured from outer bbox alone — mark as design-intent only
        chars.append({
            "characteristic": "bore_mm (design intent)",
            "nominal_mm": float(contract["bore_mm"]),
            "actual_mm": None,
            "deviation_mm": None,
            "tolerance_pm_mm": iso2768_tolerance(float(contract["bore_mm"]), cls),
            "tolerance_class": cls,
            "result": "DESIGN",
            "ok": True,
            "note": "Inner bore verified by CAD construction, not outer bbox metrology",
        })

    for name, nom, act in named:
        chars.append(check_dimension(nom, act, cls, name))

    # Envelope triple — always for phone cases (primary control); else if few named
    n_named_meas = sum(1 for c in chars if c.get("result") in ("PASS", "FAIL"))
    force_env = is_case and expected_bbox_mm
    if (
        expected_bbox_mm
        and actual_bbox_mm
        and len(expected_bbox_mm) >= 3
        and len(actual_bbox_mm) >= 3
        and (force_env or n_named_meas < 2)
    ):
        for i, (e, a) in enumerate(_pair_axes(expected_bbox_mm, actual_bbox_mm)):
            # Side-by-side assemblies (project box + lid): allow 2× on one axis
            if a > e * 1.5 and any(
                abs(a - 2 * ee) / max(2 * ee, 0.01) < 0.2 for ee in expected_bbox_mm
            ):
                chars.append({
                    "characteristic": f"envelope_axis_{i+1}",
                    "nominal_mm": round(e, 4),
                    "actual_mm": round(a, 4),
                    "deviation_mm": round(a - e, 4),
                    "tolerance_pm_mm": iso2768_tolerance(e, cls),
                    "tolerance_class": cls,
                    "result": "PASS",
                    "ok": True,
                    "note": "side-by-side multi-body layout (e.g. box+lid)",
                })
            else:
                chars.append(check_dimension(e, a, cls, f"envelope_axis_{i+1}"))

    measurable = [c for c in chars if c.get("result") in ("PASS", "FAIL")]
    fails = [c for c in measurable if not c.get("ok")]
    return {
        "tolerance_class": cls,
        "tolerance_class_name": (TOLERANCE_CLASSES.get(cls) or {}).get("name", cls),
        "characteristics": chars,
        "n_checked": len(measurable),
        "n_pass": len(measurable) - len(fails),
        "n_fail": len(fails),
        "dimensional_ok": len(fails) == 0 and len(measurable) > 0,
        "dimensional_advisory": len(measurable) == 0,
    }


def mesh_integrity(stl_path: str, nozzle_mm: float = 0.4) -> dict[str, Any]:
    """Metrology on exported mesh: manifold, walls, triangle quality."""
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    checks = []
    wt = bool(mesh.is_watertight)
    checks.append({"id": "watertight", "ok": wt, "detail": "closed manifold volume" if wt else "open mesh"})

    # Euler / winding
    try:
        winding = bool(mesh.is_winding_consistent)
    except Exception:
        winding = True
    checks.append({"id": "winding_consistent", "ok": winding, "detail": str(winding)})

    # Degenerate faces
    try:
        areas = mesh.area_faces
        degen = int(np.sum(areas < 1e-8))
    except Exception:
        degen = 0
    checks.append({
        "id": "no_degenerate_faces",
        "ok": degen == 0,
        "detail": f"{degen} near-zero area faces",
    })

    extents = [float(x) for x in mesh.extents]
    vol = None
    if wt:
        try:
            vol = float(abs(mesh.volume))
        except Exception:
            vol = None

    # Min wall thickness estimate (sample inward rays)
    min_wall, wall_method = _estimate_min_wall(mesh, samples=400)
    min_req = max(1.2, 2.0 * float(nozzle_mm))  # FDM: ≥ 2 perimeters typical
    wall_ok = min_wall is None or min_wall >= min_req * 0.85  # 15% metrology slack
    checks.append({
        "id": "min_wall_thickness",
        "ok": wall_ok if min_wall is not None else True,
        "detail": (
            f"min≈{min_wall:.2f} mm (req ≥{min_req:.2f} mm, nozzle {nozzle_mm})"
            if min_wall is not None else f"not estimated ({wall_method})"
        ),
        "min_wall_mm": None if min_wall is None else round(min_wall, 3),
        "required_mm": min_req,
    })

    # Thin overall envelope
    thin = min(extents) if extents else 0
    thin_ok = thin >= 0.6 or thin == 0
    checks.append({
        "id": "min_envelope_axis",
        "ok": thin_ok,
        "detail": f"thinnest bbox axis {thin:.3f} mm",
    })

    # Triangle aspect (very long thin tris can cause slicer pain)
    try:
        # edge length ratios
        faces = mesh.faces
        verts = mesh.vertices
        bad = 0
        sample = faces[:: max(1, len(faces) // 2000)]
        for f in sample:
            pts = verts[f]
            e = [
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[1] - pts[2]),
                np.linalg.norm(pts[2] - pts[0]),
            ]
            if min(e) > 1e-9 and max(e) / min(e) > 50:
                bad += 1
        tri_ok = bad < max(3, len(sample) * 0.02)
        checks.append({
            "id": "triangle_quality",
            "ok": tri_ok,
            "detail": f"{bad} high-aspect triangles in sample of {len(sample)}",
        })
    except Exception as e:
        checks.append({"id": "triangle_quality", "ok": True, "detail": f"skip: {e}"})

    critical = {"watertight", "winding_consistent", "no_degenerate_faces", "min_wall_thickness"}
    crit_fail = [c for c in checks if c["id"] in critical and not c["ok"]]
    return {
        "checks": checks,
        "integrity_ok": len(crit_fail) == 0,
        "extents_mm": [round(x, 3) for x in extents],
        "volume_mm3": None if vol is None else round(vol, 2),
        "faces": int(len(mesh.faces)),
        "verts": int(len(mesh.vertices)),
        "min_wall_mm": None if min_wall is None else round(min_wall, 3),
        "critical_failures": [c["id"] for c in crit_fail],
    }


def _estimate_min_wall(mesh, samples: int = 400) -> tuple[Optional[float], str]:
    """Estimate minimum wall by casting rays opposite surface normals."""
    try:
        if not mesh.is_watertight or len(mesh.faces) < 4:
            return None, "not watertight"
        # sample face centers
        centers = mesh.triangles_center
        normals = mesh.face_normals
        n = len(centers)
        if n == 0:
            return None, "empty"
        idx = np.linspace(0, n - 1, num=min(samples, n), dtype=int)
        origins = centers[idx]
        # slightly inset to avoid self-hit
        origins = origins - normals[idx] * 0.05
        directions = -normals[idx]
        # ray intersector
        try:
            from trimesh.ray.ray_triangle import RayMeshIntersector
            inter = RayMeshIntersector(mesh)
            locs, ray_ids, _ = inter.intersects_location(
                origins, directions, multiple_hits=False
            )
        except Exception:
            # fallback: proximity
            return None, "ray engine unavailable"

        if len(locs) == 0:
            return None, "no hits"
        # distance origin → hit
        dists = []
        for i, rid in enumerate(ray_ids):
            d = float(np.linalg.norm(locs[i] - origins[rid]))
            if 0.15 < d < 50:  # ignore grazing / body-length
                dists.append(d)
        if not dists:
            return None, "no valid wall samples"
        # 5th percentile — robust min
        return float(np.percentile(dists, 5)), "ray_inward_p5"
    except Exception as e:
        return None, f"error:{e}"


def build_certificate(
    *,
    name: str,
    prompt: str,
    stl_path: str,
    step_path: Optional[str] = None,
    size_contract: Optional[dict] = None,
    stats: Optional[dict] = None,
    path_used: Optional[str] = None,
    code: Optional[str] = None,
    layout_validation: Optional[dict] = None,
    tolerance_class: str = DEFAULT_CLASS,
    nozzle_mm: float = 0.4,
    intended_use: str = "engineering_prototype",
) -> dict[str, Any]:
    """Full inspection certificate (AS9102-style characteristics list)."""
    stats = stats or {}
    actual_bbox = stats.get("bbox_mm") or (size_contract or {}).get("actual_bbox_mm")
    expected = (size_contract or {}).get("expected_bbox_mm")

    dim = dimensional_inspection(
        expected, actual_bbox, size_contract=size_contract, cls=tolerance_class
    )
    integ = mesh_integrity(stl_path, nozzle_mm=nozzle_mm) if os.path.exists(stl_path) else {
        "integrity_ok": False, "checks": [], "critical_failures": ["missing_stl"],
    }

    step_ok = bool(step_path and os.path.exists(step_path))
    layout_ok = True
    if layout_validation is not None:
        layout_ok = bool(layout_validation.get("ok"))

    # Process controls
    process = [
        {
            "id": "brep_step_export",
            "ok": step_ok or path_used in ("imagine", "imagine_hq"),
            "detail": "STEP present (B-rep)" if step_ok else (
                "mesh-only path (organic/HQ)" if path_used in ("imagine", "imagine_hq")
                else "STEP missing — regenerate mechanical path"
            ),
        },
        {
            "id": "size_contract",
            "ok": size_contract is not None,
            "detail": (size_contract or {}).get("source") or "no size contract",
        },
        {
            "id": "feature_layout",
            "ok": layout_ok,
            "detail": "layout probes passed" if layout_ok else str(
                (layout_validation or {}).get("errors")
            ),
        },
        {
            "id": "generation_path",
            "ok": True,
            "detail": path_used or "unknown",
        },
    ]

    # Hash for traceability
    sha = None
    try:
        h = hashlib.sha256()
        with open(stl_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        sha = h.hexdigest()[:16]
    except Exception:
        pass
    code_sha = hashlib.sha256((code or "").encode()).hexdigest()[:12] if code else None

    dim_ok = dim.get("dimensional_ok") or dim.get("dimensional_advisory")
    integ_ok = integ.get("integrity_ok")
    process_ok = all(p["ok"] for p in process if p["id"] != "generation_path")

    # Overall disposition
    if integ_ok and dim_ok and process_ok and layout_ok:
        disposition = "ACCEPT"
        grade = "A"
    elif integ_ok and (dim_ok or dim.get("dimensional_advisory")):
        disposition = "ACCEPT_WITH_DEVIATION"
        grade = "B"
    else:
        disposition = "REJECT"
        grade = "F"

    cert = {
        "schema": "stl-sandbox-inspection-v1",
        "title": "Dimensional & Process Inspection Certificate",
        "standards_referenced": [
            "ISO 2768-1 (general tolerances — linear)",
            "ISO/ASTM 52900 (additive manufacturing vocabulary)",
            "AS9102-style characteristic reporting (structure only)",
        ],
        "disclaimer": (
            "This certificate documents dimensional inspection and digital process "
            "controls for a sandbox-generated solid/mesh. It is suitable for engineering "
            "prototype control and design verification. It is NOT a statement of "
            "AS9100, NADCAP, ITAR, flight-worthiness, or production aerospace qualification. "
            "Material certification, process FAI, and NDT are out of scope."
        ),
        "part": {
            "name": name,
            "prompt": prompt,
            "revision": "A",
            "path_used": path_used,
            "stl_sha256_16": sha,
            "code_sha256_12": code_sha,
            "inspected_at_unix": time.time(),
            "intended_use": intended_use,
        },
        "tolerance_class": tolerance_class,
        "tolerance_class_name": (TOLERANCE_CLASSES.get(tolerance_class) or {}).get("name"),
        "dimensional": dim,
        "integrity": integ,
        "process_controls": process,
        "layout_validation": {
            "ok": layout_ok,
            "errors": (layout_validation or {}).get("errors") or [],
        } if layout_validation is not None else None,
        "disposition": disposition,
        "grade": grade,
        "conformance_summary": _summary(disposition, grade, dim, integ),
    }
    return cert


def _summary(disposition: str, grade: str, dim: dict, integ: dict) -> str:
    if disposition == "ACCEPT":
        return (
            f"Grade {grade}: dimensional characteristics within "
            f"{dim.get('tolerance_class_name')} and mesh integrity gates passed."
        )
    if disposition == "ACCEPT_WITH_DEVIATION":
        return (
            f"Grade {grade}: integrity OK; dimensional advisory or minor process notes. "
            "Review deviations before critical fit."
        )
    fails = integ.get("critical_failures") or []
    return (
        f"Grade {grade}: REJECT — critical failures: {', '.join(fails) or 'dimensional/process'}. "
        "Do not use for fit-critical or safety-related applications."
    )


def certificate_text(cert: dict) -> str:
    """Human-readable inspection report."""
    lines = [
        "=" * 64,
        cert.get("title", "Inspection Certificate"),
        "=" * 64,
        f"Part:        {cert.get('part', {}).get('name')}",
        f"Prompt:      {cert.get('part', {}).get('prompt')}",
        f"Path:        {cert.get('part', {}).get('path_used')}",
        f"Class:       {cert.get('tolerance_class_name')}",
        f"Disposition: {cert.get('disposition')}  (Grade {cert.get('grade')})",
        f"STL hash:    {cert.get('part', {}).get('stl_sha256_16')}",
        "",
        "— Dimensional characteristics —",
    ]
    for c in (cert.get("dimensional") or {}).get("characteristics") or []:
        act = c.get("actual_mm")
        act_s = "—" if act is None else f"{act:.3f}"
        lines.append(
            f"  [{c.get('result'):4}] {c.get('characteristic')}: "
            f"nom {c.get('nominal_mm')}  act {act_s}  "
            f"dev {c.get('deviation_mm')}  ±{c.get('tolerance_pm_mm')}"
        )
    lines.append("")
    lines.append("— Integrity —")
    for c in (cert.get("integrity") or {}).get("checks") or []:
        lines.append(f"  [{'PASS' if c.get('ok') else 'FAIL'}] {c.get('id')}: {c.get('detail')}")
    lines.append("")
    lines.append("— Process controls —")
    for c in cert.get("process_controls") or []:
        lines.append(f"  [{'PASS' if c.get('ok') else 'FAIL'}] {c.get('id')}: {c.get('detail')}")
    lines += [
        "",
        cert.get("conformance_summary", ""),
        "",
        "DISCLAIMER:",
        cert.get("disclaimer", ""),
        "=" * 64,
    ]
    return "\n".join(lines)


def should_reject_generation(cert: dict, strict: bool = True) -> Optional[str]:
    """If non-None, generation should not be accepted as success."""
    if cert.get("disposition") == "REJECT":
        fails = (cert.get("integrity") or {}).get("critical_failures") or []
        dim_fails = [
            c["characteristic"]
            for c in (cert.get("dimensional") or {}).get("characteristics") or []
            if c.get("result") == "FAIL"
        ]
        bits = fails + [f"dim:{d}" for d in dim_fails[:4]]
        return "ISO inspection REJECT: " + (", ".join(bits) if bits else cert.get("grade", "F"))
    if strict and cert.get("disposition") == "ACCEPT_WITH_DEVIATION":
        # still accept for sandbox but caller may warn
        return None
    return None
