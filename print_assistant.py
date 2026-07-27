"""Print Assistant — inspect STL, check bed fit, recommend slicer settings,
predict failures, and produce a demo print walkthrough.

All sizing is grounded in measured mesh bounds vs the printer build volume.
Nothing is marked printable unless it fits X, Y, and Z.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import trimesh

# ── Defaults when info is missing ──────────────────────────────────────────
DEFAULTS = {
    "printer_type": "FDM",
    "nozzle_mm": 0.4,
    "layer_height_mm": 0.2,
    "filament": "PLA",
    "slicer": "standard",
    "demo_speed": 10,  # 2 | 10 | 60
    "clearance_mm": 5.0,  # edge clearance
    "barely_margin_mm": 8.0,  # "barely fits" band
    "support_angle_deg": 45.0,
    "intended_use": "prototype",
}

# Common printer profiles (X, Y, Z mm). Unknown bed = missing critical info.
PRINTER_PROFILES: dict[str, dict[str, Any]] = {
    "custom": {
        "label": "Custom / unknown",
        "bed_x": None,
        "bed_y": None,
        "bed_z": None,
        "nozzle_mm": 0.4,
    },
    "ender3": {
        "label": "Creality Ender 3 / V2 / Neo",
        "bed_x": 220,
        "bed_y": 220,
        "bed_z": 250,
        "nozzle_mm": 0.4,
    },
    "ender3_s1": {
        "label": "Creality Ender 3 S1",
        "bed_x": 220,
        "bed_y": 220,
        "bed_z": 270,
        "nozzle_mm": 0.4,
    },
    "prusa_mk4": {
        "label": "Prusa MK3S+ / MK4",
        "bed_x": 250,
        "bed_y": 210,
        "bed_z": 220,
        "nozzle_mm": 0.4,
    },
    "prusa_mini": {
        "label": "Prusa MINI+",
        "bed_x": 180,
        "bed_y": 180,
        "bed_z": 180,
        "nozzle_mm": 0.4,
    },
    "bambu_a1_mini": {
        "label": "Bambu Lab A1 mini",
        "bed_x": 180,
        "bed_y": 180,
        "bed_z": 180,
        "nozzle_mm": 0.4,
    },
    "bambu_p1s": {
        "label": "Bambu Lab P1S / X1C",
        "bed_x": 256,
        "bed_y": 256,
        "bed_z": 256,
        "nozzle_mm": 0.4,
    },
    "voron_0": {
        "label": "Voron 0.2",
        "bed_x": 120,
        "bed_y": 120,
        "bed_z": 120,
        "nozzle_mm": 0.4,
    },
    "voron_2.4_300": {
        "label": "Voron 2.4 300",
        "bed_x": 300,
        "bed_y": 300,
        "bed_z": 300,
        "nozzle_mm": 0.4,
    },
    "anycubic_kobra2": {
        "label": "Anycubic Kobra 2",
        "bed_x": 220,
        "bed_y": 220,
        "bed_z": 250,
        "nozzle_mm": 0.4,
    },
    "elegoo_neptune4": {
        "label": "Elegoo Neptune 4",
        "bed_x": 225,
        "bed_y": 225,
        "bed_z": 265,
        "nozzle_mm": 0.4,
    },
}

# Material → temp / cooling / warp risk
FILAMENT_PRESETS: dict[str, dict[str, Any]] = {
    "PLA": {
        "nozzle_c": 210,
        "bed_c": 60,
        "cooling_pct": 100,
        "speed_mm_s": 60,
        "warp": "low",
        "brim_default": False,
    },
    "PETG": {
        "nozzle_c": 240,
        "bed_c": 80,
        "cooling_pct": 40,
        "speed_mm_s": 45,
        "warp": "medium",
        "brim_default": True,
    },
    "ABS": {
        "nozzle_c": 250,
        "bed_c": 100,
        "cooling_pct": 20,
        "speed_mm_s": 50,
        "warp": "high",
        "brim_default": True,
    },
    "ASA": {
        "nozzle_c": 250,
        "bed_c": 100,
        "cooling_pct": 20,
        "speed_mm_s": 50,
        "warp": "high",
        "brim_default": True,
    },
    "TPU": {
        "nozzle_c": 220,
        "bed_c": 50,
        "cooling_pct": 50,
        "speed_mm_s": 25,
        "warp": "low",
        "brim_default": False,
    },
    "Nylon": {
        "nozzle_c": 260,
        "bed_c": 80,
        "cooling_pct": 30,
        "speed_mm_s": 40,
        "warp": "high",
        "brim_default": True,
    },
}

USE_PROFILES: dict[str, dict[str, Any]] = {
    "display": {"walls": 2, "infill": 10, "layer_mult": 1.0, "strength_note": "appearance first"},
    "prototype": {"walls": 2, "infill": 15, "layer_mult": 1.0, "strength_note": "fast iteration"},
    "functional": {"walls": 3, "infill": 30, "layer_mult": 0.9, "strength_note": "load-bearing walls + denser infill"},
    "tool": {"walls": 4, "infill": 40, "layer_mult": 0.8, "strength_note": "max walls, gyroid/cubic infill"},
    "toy": {"walls": 2, "infill": 15, "layer_mult": 1.0, "strength_note": "smooth outer walls, low speed for details"},
    "mechanical": {"walls": 4, "infill": 35, "layer_mult": 0.8, "strength_note": "orient load across layers carefully"},
}


@dataclass
class PrinterContext:
    printer_model: str = "custom"
    bed_x: Optional[float] = None
    bed_y: Optional[float] = None
    bed_z: Optional[float] = None
    nozzle_mm: float = 0.4
    layer_height_mm: float = 0.2
    filament: str = "PLA"
    slicer: str = "standard"
    intended_use: str = "prototype"
    demo_speed: int = 10
    clearance_mm: float = 5.0
    support_angle_deg: float = 45.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "PrinterContext":
        d = dict(d or {})
        profile_id = (d.get("printer_model") or d.get("printer") or "custom").lower().replace(" ", "_")
        # alias common names
        aliases = {
            "ender_3": "ender3",
            "ender3v2": "ender3",
            "mk4": "prusa_mk4",
            "mk3": "prusa_mk4",
            "prusa_mk3": "prusa_mk4",
            "x1c": "bambu_p1s",
            "p1s": "bambu_p1s",
            "a1_mini": "bambu_a1_mini",
            "a1mini": "bambu_a1_mini",
        }
        profile_id = aliases.get(profile_id, profile_id)
        base = PRINTER_PROFILES.get(profile_id, PRINTER_PROFILES["custom"]).copy()

        def _first(*vals):
            for v in vals:
                if v is not None and v != "":
                    return v
            return None

        bed_x = _first(d.get("bed_x"), d.get("bed_size_x"), base.get("bed_x"))
        bed_y = _first(d.get("bed_y"), d.get("bed_size_y"), base.get("bed_y"))
        bed_z = _first(d.get("bed_z"), d.get("bed_size_z"), d.get("max_z"), base.get("bed_z"))
        # allow list/tuple bed_size
        if d.get("bed_size") and isinstance(d["bed_size"], (list, tuple)) and len(d["bed_size"]) >= 2:
            bed_x = _first(bed_x, d["bed_size"][0])
            bed_y = _first(bed_y, d["bed_size"][1])
            if len(d["bed_size"]) >= 3:
                bed_z = _first(bed_z, d["bed_size"][2])

        def _f(key, default):
            v = d.get(key)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _i(key, default):
            v = d.get(key)
            if v is None or v == "":
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        filament = str(d.get("filament") or DEFAULTS["filament"]).upper()
        if filament not in FILAMENT_PRESETS:
            # try title-case keys like "Nylon"
            for k in FILAMENT_PRESETS:
                if k.upper() == filament:
                    filament = k
                    break
            else:
                filament = "PLA"

        use = str(d.get("intended_use") or d.get("use") or DEFAULTS["intended_use"]).lower()
        if use not in USE_PROFILES:
            use = "prototype"

        speed = _i("demo_speed", DEFAULTS["demo_speed"])
        if speed not in (2, 10, 60):
            speed = 10

        return cls(
            printer_model=profile_id,
            bed_x=float(bed_x) if bed_x is not None else None,
            bed_y=float(bed_y) if bed_y is not None else None,
            bed_z=float(bed_z) if bed_z is not None else None,
            nozzle_mm=_f("nozzle_mm", base.get("nozzle_mm") or DEFAULTS["nozzle_mm"]),
            layer_height_mm=_f("layer_height_mm", DEFAULTS["layer_height_mm"]),
            filament=filament if filament in FILAMENT_PRESETS else "PLA",
            slicer=str(d.get("slicer") or DEFAULTS["slicer"]),
            intended_use=use,
            demo_speed=speed,
            clearance_mm=_f("clearance_mm", DEFAULTS["clearance_mm"]),
            support_angle_deg=_f("support_angle_deg", DEFAULTS["support_angle_deg"]),
        )

    @property
    def bed_known(self) -> bool:
        return self.bed_x is not None and self.bed_y is not None and self.bed_z is not None

    @property
    def label(self) -> str:
        p = PRINTER_PROFILES.get(self.printer_model, {})
        return p.get("label") or self.printer_model


# ── Mesh loading & health ──────────────────────────────────────────────────

def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("STL scene contains no mesh geometry")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")
    # ensure face normals
    if mesh.faces is not None and len(mesh.faces):
        mesh.rezero()
    return mesh


def model_dims(mesh: trimesh.Trimesh) -> dict[str, float]:
    extents = mesh.extents.astype(float)
    # extents are axis-aligned size after rezero; order is X,Y,Z of mesh coords
    return {
        "x_mm": round(float(extents[0]), 2),
        "y_mm": round(float(extents[1]), 2),
        "z_mm": round(float(extents[2]), 2),
        "volume_cm3": round(float(abs(mesh.volume)) / 1000.0, 2) if mesh.is_watertight else None,
        "face_count": int(len(mesh.faces)),
        "vert_count": int(len(mesh.vertices)),
    }


def inspect_health(mesh: trimesh.Trimesh, nozzle_mm: float = 0.4) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    ok_flags: list[str] = []

    wt = bool(mesh.is_watertight)
    if wt:
        ok_flags.append("watertight")
    else:
        issues.append({"id": "non_manifold", "severity": "high", "msg": "Non-manifold / not watertight — repair before slicing"})

    try:
        if hasattr(mesh, "is_winding_consistent") and not mesh.is_winding_consistent:
            issues.append({"id": "flipped_normals", "severity": "medium", "msg": "Inconsistent winding / possible flipped normals"})
        else:
            ok_flags.append("consistent winding")
    except Exception:
        pass

    # open edges (boundary edges)
    try:
        if mesh.edges_unique is not None and mesh.faces is not None:
            # boundary edges appear once in face-edge incidence
            edges = mesh.edges_sorted.reshape((-1, 2))
            # count occurrences
            if len(edges):
                # use trimesh's outline
                outline = mesh.outline()
                if outline is not None and hasattr(outline, "entities") and len(outline.entities) > 0:
                    issues.append({"id": "open_edges", "severity": "high", "msg": "Open edges / holes detected on the mesh surface"})
                elif wt:
                    ok_flags.append("no open edges")
    except Exception:
        pass

    # units sanity: if largest dim < 2mm, likely inches or wrong units; if > 1000mm, meters?
    extents = mesh.extents
    max_dim = float(max(extents))
    min_dim = float(min(extents))
    if max_dim < 2.0:
        issues.append({
            "id": "bad_scale",
            "severity": "high",
            "msg": f"Model is only {max_dim:.2f} mm on longest axis — likely wrong units (inches or meters exported as mm)",
        })
    elif max_dim > 800:
        issues.append({
            "id": "bad_scale",
            "severity": "medium",
            "msg": f"Model is {max_dim:.0f} mm long — confirm units are millimeters",
        })
    else:
        ok_flags.append("scale looks like mm")

    if min_dim < nozzle_mm:
        issues.append({
            "id": "thin_walls",
            "severity": "high",
            "msg": f"Thinnest axis {min_dim:.2f} mm is below nozzle width ({nozzle_mm} mm) — will not print solid",
        })
    elif min_dim < nozzle_mm * 2:
        issues.append({
            "id": "thin_walls",
            "severity": "medium",
            "msg": f"Thinnest axis {min_dim:.2f} mm is under 2× nozzle — fragile single-wall regions likely",
        })

    # triangle count
    nfaces = len(mesh.faces)
    if nfaces > 500_000:
        issues.append({"id": "excessive_tris", "severity": "medium", "msg": f"{nfaces:,} triangles — simplify for faster slicing"})
    elif nfaces > 1_000_000:
        issues.append({"id": "excessive_tris", "severity": "high", "msg": f"{nfaces:,} triangles — slicer may choke; decimate first"})
    else:
        ok_flags.append(f"{nfaces:,} triangles (ok)")

    # bodies / floating islands
    ncomp = 1
    try:
        components = mesh.split(only_watertight=False)
        ncomp = len(components) if isinstance(components, list) else 1
        if ncomp > 1:
            issues.append({
                "id": "floating_islands",
                "severity": "medium",
                "msg": f"{ncomp} disconnected shells — floating islands may need supports or separate prints",
            })
        else:
            ok_flags.append("single body")
    except Exception:
        pass

    # tiny fragile features: very small connected components or thin spikes
    try:
        if mesh.is_watertight and mesh.volume is not None:
            fill = float(abs(mesh.volume)) / float(np.prod(mesh.extents) + 1e-9)
            if fill < 0.02:
                issues.append({
                    "id": "degenerate",
                    "severity": "high",
                    "msg": "Solid fills <2% of bounding box — geometry may be degenerate or mostly empty shell",
                })
    except Exception:
        pass

    # intersecting shells heuristic: non-watertight + multi body
    if not wt and ncomp > 1:
        issues.append({
            "id": "intersecting_shells",
            "severity": "medium",
            "msg": "Multiple non-watertight shells — possible intersecting geometry",
        })

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    worst = max((severity_rank.get(i["severity"], 0) for i in issues), default=0)
    health = "good" if worst == 0 else ("fair" if worst <= 2 else "poor")

    return {
        "health": health,
        "watertight": wt,
        "issues": issues,
        "ok": ok_flags,
        "components": ncomp,
    }


# ── Build volume / fit ─────────────────────────────────────────────────────

def _fits(dims: tuple[float, float, float], bed: tuple[float, float, float], clearance: float) -> bool:
    usable = (bed[0] - 2 * clearance, bed[1] - 2 * clearance, bed[2] - clearance)
    return dims[0] <= usable[0] + 1e-6 and dims[1] <= usable[1] + 1e-6 and dims[2] <= usable[2] + 1e-6


def rotation_candidates(x: float, y: float, z: float) -> list[dict]:
    """Axis-aligned rotations (6 permutations of dimensions)."""
    perms = [
        ((x, y, z), "as-is (no rotation)"),
        ((x, z, y), "rotate 90° around X (swap Y/Z)"),
        ((y, x, z), "rotate 90° around Z (swap X/Y)"),
        ((y, z, x), "rotate: Y→X, Z→Y, X→Z"),
        ((z, x, y), "rotate: Z→X, X→Y, Y→Z"),
        ((z, y, x), "rotate 90° around Y (swap X/Z)"),
    ]
    seen = set()
    out = []
    for dims, label in perms:
        key = tuple(round(d, 3) for d in dims)
        if key in seen:
            continue
        seen.add(key)
        out.append({"dims": dims, "label": label, "x": dims[0], "y": dims[1], "z": dims[2]})
    return out


def check_bed_fit(dims: dict, ctx: PrinterContext) -> dict[str, Any]:
    x, y, z = dims["x_mm"], dims["y_mm"], dims["z_mm"]
    if not ctx.bed_known:
        return {
            "fits": None,
            "fits_label": "Unknown",
            "bed_known": False,
            "critical": True,
            "message": "Bed size / build volume is missing — cannot recommend print-as-is. Set printer bed X, Y, Z first.",
            "model_mm": [x, y, z],
            "printer_mm": None,
            "required_change": "provide bed size",
            "options": [],
            "best_rotation": None,
            "safe_scale_pct": None,
            "split_recommended": False,
        }

    bed = (float(ctx.bed_x), float(ctx.bed_y), float(ctx.bed_z))
    clearance = ctx.clearance_mm
    model = (x, y, z)

    candidates = rotation_candidates(x, y, z)
    scored = []
    for c in candidates:
        d = c["dims"]
        fits_strict = _fits(d, bed, clearance)
        fits_raw = d[0] <= bed[0] and d[1] <= bed[1] and d[2] <= bed[2]
        barely = fits_raw and not fits_strict
        # score: prefer fitting with least rotation change and largest leftover margin
        margin = min(bed[0] - d[0], bed[1] - d[1], bed[2] - d[2])
        score = (2 if fits_strict else (1 if fits_raw else 0), margin, -abs(d[2] - z))  # prefer original Z if tie
        scored.append({**c, "fits_strict": fits_strict, "fits_raw": fits_raw, "barely": barely, "margin_mm": round(margin, 1), "score": score})

    scored.sort(key=lambda s: s["score"], reverse=True)
    best = scored[0]

    as_is_strict = _fits(model, bed, clearance)
    as_is_raw = model[0] <= bed[0] and model[1] <= bed[1] and model[2] <= bed[2]
    as_is_barely = as_is_raw and not as_is_strict

    # scale needed for best orientation
    usable = (bed[0] - 2 * clearance, bed[1] - 2 * clearance, bed[2] - clearance)
    bx, by, bz = best["dims"]
    scale_factors = [usable[0] / bx if bx > 0 else 1, usable[1] / by if by > 0 else 1, usable[2] / bz if bz > 0 else 1]
    safe_scale = min(1.0, min(scale_factors))
    safe_scale_pct = math.floor(safe_scale * 1000) / 10  # one decimal, always floor

    options = []
    if as_is_strict:
        fits_label = "Yes"
        required = "none"
        message = "Model fits the build volume with edge clearance."
    elif as_is_barely:
        fits_label = "Maybe"
        required = "watch edges / use brim"
        message = (
            f"Model barely fits ({x:.1f}×{y:.1f}×{z:.1f} mm on {bed[0]:.0f}×{bed[1]:.0f}×{bed[2]:.0f} mm bed). "
            f"Edge clearance under {clearance:.0f} mm — adhesion risk at corners."
        )
        options.append("Add brim (5–8 mm) and center carefully")
        options.append("Slow first layer; clean bed")
    else:
        fits_label = "No"
        required = "rotate / scale / split"
        message = f"Does not fit as-is: model {x:.1f}×{y:.1f}×{z:.1f} mm vs bed {bed[0]:.0f}×{bed[1]:.0f}×{bed[2]:.0f} mm."
        if best["fits_strict"]:
            required = "rotate"
            options.append(f"Rotate: {best['label']} → {best['x']:.1f}×{best['y']:.1f}×{best['z']:.1f} mm (fits with clearance)")
        elif best["fits_raw"]:
            required = "rotate + careful placement"
            options.append(f"Rotate: {best['label']} → barely fits; use brim and center")
        if safe_scale_pct < 100:
            scaled = (round(x * safe_scale_pct / 100, 1), round(y * safe_scale_pct / 100, 1), round(z * safe_scale_pct / 100, 1))
            options.append(f"Scale to ≤{safe_scale_pct}% → ~{scaled[0]}×{scaled[1]}×{scaled[2]} mm")
            options.append(f"Safe scale percentage: {safe_scale_pct}% (includes {clearance:.0f} mm edge clearance)")
        # split recommendation
        if max(x, y) > max(bed[0], bed[1]) or z > bed[2]:
            options.append("Split into 2+ parts along longest overflow axis")
            cut_axis = "Z (height)" if z > bed[2] else ("X" if x >= y else "Y")
            options.append(f"Preferred cut plane: mid-{cut_axis} — add pegs/dovetails or glue joint")
            options.append("Connectors: pegs (⌀ nozzle×3), magnets, or CA glue pads")
        options.append("Or use a larger printer")

    split_recommended = fits_label == "No" and safe_scale_pct < 70

    # excess per axis
    excess = {
        "x": round(max(0, x - bed[0]), 2),
        "y": round(max(0, y - bed[1]), 2),
        "z": round(max(0, z - bed[2]), 2),
    }

    return {
        "fits": fits_label == "Yes",
        "fits_label": fits_label,  # Yes / Maybe / No / Unknown
        "bed_known": True,
        "critical": fits_label == "No",
        "message": message,
        "model_mm": [round(x, 2), round(y, 2), round(z, 2)],
        "printer_mm": [bed[0], bed[1], bed[2]],
        "usable_mm": [round(usable[0], 1), round(usable[1], 1), round(usable[2], 1)],
        "clearance_mm": clearance,
        "excess_mm": excess,
        "required_change": required,
        "options": options,
        "best_rotation": {
            "label": best["label"],
            "dims_mm": [round(best["x"], 2), round(best["y"], 2), round(best["z"], 2)],
            "fits": best["fits_strict"],
            "barely": best["barely"],
        },
        "safe_scale_pct": safe_scale_pct if safe_scale_pct < 100 else 100.0,
        "scaled_dims_mm": [
            round(x * min(safe_scale_pct, 100) / 100, 1),
            round(y * min(safe_scale_pct, 100) / 100, 1),
            round(z * min(safe_scale_pct, 100) / 100, 1),
        ],
        "split_recommended": split_recommended,
        "barely_fits": as_is_barely,
    }


# ── Printability ───────────────────────────────────────────────────────────

def analyze_printability(mesh: trimesh.Trimesh, ctx: PrinterContext) -> dict[str, Any]:
    """Heuristic printability from mesh normals + footprint."""
    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals
    zmin = float(verts[:, 2].min())
    zmax = float(verts[:, 2].max())
    height = max(zmax - zmin, 1e-6)

    # face areas
    try:
        areas = mesh.area_faces
    except Exception:
        areas = np.ones(len(faces))

    # bed contact first — bottom faces are not overhangs
    tol = max(ctx.layer_height_mm * 1.5, 0.3)
    face_z = verts[faces][:, :, 2].min(axis=1)
    on_bed = face_z <= zmin + tol
    bottom_mask = on_bed & (normals[:, 2] < -0.3)
    bed_contact_area = float(areas[bottom_mask].sum()) if bottom_mask.any() else 0.0
    # footprint from axis-aligned bbox of low vertices (no scipy)
    low_verts = verts[verts[:, 2] <= zmin + tol]
    if len(low_verts) >= 2:
        span = low_verts[:, :2].max(axis=0) - low_verts[:, :2].min(axis=0)
        footprint = float(max(span[0], 0.1) * max(span[1], 0.1))
    else:
        footprint = float(mesh.extents[0] * mesh.extents[1] * 0.2)
    bed_contact_area = max(bed_contact_area, footprint * 0.5)

    # overhang: downward-facing faces NOT on the bed (ceilings / steep undersides)
    ang = math.radians(ctx.support_angle_deg)
    thr = -math.sin(ang)
    overhang_mask = (normals[:, 2] < thr) & (~on_bed)
    overhang_area = float(areas[overhang_mask].sum()) if overhang_mask.any() else 0.0
    total_area = float(areas.sum()) + 1e-9
    overhang_pct = 100.0 * overhang_area / total_area

    # bridges / ceilings: near-horizontal faces facing down, not on bed
    bridge_mask = (normals[:, 2] < -0.85) & (~on_bed)
    bridge_area = float(areas[bridge_mask].sum()) if bridge_mask.any() else 0.0

    # orientation suggestion: prefer max bed contact / min overhang
    orientations = []
    # evaluate 3 primary axis-as-up orientations by transforming extents/normals heuristically
    # Simplified: recommend largest face-down by comparing projected areas on each axis pair
    ex = mesh.extents
    # score current
    orientations.append({
        "up": "+Z (current)",
        "overhang_pct": round(overhang_pct, 1),
        "bed_contact_mm2": round(bed_contact_area, 1),
        "height_mm": round(height, 1),
    })

    # crude: if overhang high and height is longest axis, lay flat
    longest_axis = int(np.argmax(ex))
    axis_names = ["X", "Y", "Z"]
    orient_note = "current orientation OK"
    if overhang_pct > 15 and longest_axis != 2:
        orient_note = f"Consider laying the {axis_names[longest_axis]} axis flat to reduce overhangs"
    elif bed_contact_area < 50 and height > 30:
        orient_note = "Small footprint vs height — risk of tip-over; add brim or reorient for larger base"

    supports_needed = overhang_pct > 3 or bridge_area > 20
    support_type = "none"
    if supports_needed:
        if overhang_pct > 20 or bridge_area > 100:
            support_type = "tree" if overhang_pct > 25 else "normal"
        else:
            support_type = "normal"

    # warping risk from height + material + footprint
    mat = FILAMENT_PRESETS.get(ctx.filament, FILAMENT_PRESETS["PLA"])
    warp_base = mat["warp"]
    if bed_contact_area < 100 and height > 50:
        warp_risk = "high" if warp_base != "low" else "medium"
    elif warp_base == "high":
        warp_risk = "high"
    else:
        warp_risk = warp_base

    # thin feature risk
    min_axis = float(min(ex))
    thin_risk = min_axis < ctx.nozzle_mm * 3

    # strength direction
    strength = "Layer lines are weakest in Z — orient so loads run in XY walls when possible"

    return {
        "bed_contact_mm2": round(bed_contact_area, 1),
        "overhang_pct": round(overhang_pct, 1),
        "bridge_area_mm2": round(bridge_area, 1),
        "supports_needed": supports_needed,
        "support_type": support_type,
        "support_angle_deg": ctx.support_angle_deg,
        "orientation_note": orient_note,
        "orientations": orientations,
        "warp_risk": warp_risk,
        "thin_feature_risk": thin_risk,
        "strength_direction": strength,
        "height_mm": round(height, 2),
        "layer_count_est": int(math.ceil(height / max(ctx.layer_height_mm, 0.05))),
    }


# ── Slicer settings ────────────────────────────────────────────────────────

def recommend_settings(ctx: PrinterContext, fit: dict, printability: dict, health: dict) -> dict[str, Any]:
    mat = FILAMENT_PRESETS.get(ctx.filament, FILAMENT_PRESETS["PLA"])
    use = USE_PROFILES.get(ctx.intended_use, USE_PROFILES["prototype"])
    lh = ctx.layer_height_mm
    # layer height vs nozzle
    if lh > ctx.nozzle_mm * 0.8:
        lh = round(ctx.nozzle_mm * 0.5, 2)

    walls = use["walls"]
    infill = use["infill"]
    if printability.get("thin_feature_risk"):
        walls = max(walls, 3)

    support = printability.get("support_type", "none")
    support_angle = ctx.support_angle_deg

    # adhesion
    brim = mat["brim_default"] or printability.get("bed_contact_mm2", 999) < 150 or fit.get("barely_fits")
    raft = False
    if ctx.filament in ("ABS", "ASA", "Nylon") and printability.get("bed_contact_mm2", 999) < 200:
        raft = True
        brim = False
    skirt = not brim and not raft

    adhesion = "raft" if raft else ("brim 5–8 mm" if brim else "skirt 2 loops")

    scale_pct = 100.0
    orientation = "as-is"
    if fit.get("fits_label") == "No":
        if fit.get("best_rotation") and fit["best_rotation"].get("fits"):
            orientation = fit["best_rotation"]["label"]
            scale_pct = 100.0
        else:
            scale_pct = fit.get("safe_scale_pct") or 100.0
            if fit.get("best_rotation"):
                orientation = fit["best_rotation"]["label"] + " then scale"
    elif fit.get("best_rotation") and fit.get("fits_label") != "Yes":
        orientation = fit["best_rotation"]["label"]

    speed = mat["speed_mm_s"]
    if ctx.intended_use in ("display", "toy"):
        speed = int(speed * 0.75)

    return {
        "layer_height_mm": lh,
        "wall_count": walls,
        "infill_pct": infill,
        "infill_pattern": "gyroid" if ctx.intended_use in ("functional", "tool", "mechanical") else "grid",
        "support_type": support,
        "support_angle_deg": support_angle,
        "adhesion": adhesion,
        "brim": brim,
        "raft": raft,
        "skirt": skirt,
        "nozzle_temp_c": mat["nozzle_c"],
        "bed_temp_c": mat["bed_c"],
        "speed_mm_s": speed,
        "cooling_pct": mat["cooling_pct"],
        "nozzle_mm": ctx.nozzle_mm,
        "filament": ctx.filament,
        "scaling_pct": scale_pct,
        "orientation": orientation,
        "notes": use["strength_note"],
    }


# ── Failure prediction ─────────────────────────────────────────────────────

def predict_failures(fit: dict, health: dict, printability: dict, ctx: PrinterContext, settings: dict) -> list[dict]:
    fails: list[dict] = []

    if not fit.get("bed_known"):
        fails.append({"id": "unknown_bed", "title": "Unknown bed size", "detail": "Cannot verify fit — set build volume before printing"})
    elif fit.get("fits_label") == "No":
        excess = fit.get("excess_mm") or {}
        if excess.get("z", 0) > 0:
            fails.append({"id": "too_tall", "title": "Model too tall for Z height", "detail": f"Exceeds max Z by {excess['z']} mm"})
        if excess.get("x", 0) > 0 or excess.get("y", 0) > 0:
            fails.append({"id": "too_large", "title": "Model too large for bed", "detail": f"Overflow X={excess.get('x',0)} Y={excess.get('y',0)} mm"})
    elif fit.get("barely_fits"):
        fails.append({"id": "edge_adhesion", "title": "Bed adhesion failure (edge)", "detail": "Barely fits — corners near bed edge lift easily"})

    if printability.get("bed_contact_mm2", 999) < 80:
        fails.append({"id": "adhesion", "title": "Bed adhesion failure", "detail": f"Small contact area (~{printability.get('bed_contact_mm2')} mm²) — use brim/raft"})

    if printability.get("supports_needed") and printability.get("overhang_pct", 0) > 12:
        fails.append({"id": "support", "title": "Support failure / sagging overhang", "detail": f"~{printability.get('overhang_pct')}% overhang area needs solid supports"})

    if printability.get("warp_risk") == "high":
        fails.append({"id": "warping", "title": "Warping", "detail": f"{ctx.filament} + geometry prone to corner lift — enclosure/brim/raft"})

    if printability.get("bridge_area_mm2", 0) > 150:
        fails.append({"id": "overhang", "title": "Bad overhang / bridge", "detail": "Large downward faces may sag without cooling + supports"})

    if printability.get("thin_feature_risk"):
        fails.append({"id": "detail", "title": "Broken small detail", "detail": "Features near nozzle width — print slow, more walls, or thicken model"})

    for iss in health.get("issues") or []:
        if iss["id"] in ("non_manifold", "open_edges", "degenerate"):
            fails.append({"id": "mesh", "title": "Mesh integrity failure", "detail": iss["msg"]})
            break

    if ctx.filament in ("PETG", "TPU") and settings.get("cooling_pct", 0) > 60:
        fails.append({"id": "stringing", "title": "Stringing", "detail": "Tune retraction; PETG/TPU string easily at high cooling"})

    if ctx.intended_use in ("functional", "tool", "mechanical") and settings.get("infill_pct", 0) < 25:
        fails.append({"id": "weak", "title": "Weak part", "detail": "Functional use needs denser infill + more walls"})

    # dedupe by id, keep top 3
    seen = set()
    top = []
    for f in fails:
        if f["id"] in seen:
            continue
        seen.add(f["id"])
        top.append(f)
        if len(top) >= 3:
            break

    if not top:
        top.append({"id": "none", "title": "No major failures predicted", "detail": "Still watch first layer carefully"})

    return top


# ── Demo print simulation ──────────────────────────────────────────────────

def demo_walkthrough(dims: dict, fit: dict, printability: dict, ctx: PrinterContext, settings: dict) -> dict[str, Any]:
    speed = ctx.demo_speed
    layers = max(1, printability.get("layer_count_est") or int(dims["z_mm"] / max(ctx.layer_height_mm, 0.05)))
    supports = printability.get("supports_needed")
    fits = fit.get("fits_label")

    # timeline scales with speed: higher speed = shorter timestamps
    # 10x is baseline (~60s narrative)
    scale = {2: 3.0, 10: 1.0, 60: 0.25}.get(speed, 1.0)

    def t(sec_at_10x: float) -> str:
        s = sec_at_10x * scale
        m = int(s // 60)
        rem = int(s % 60)
        return f"{m:02d}:{rem:02d}"

    stages = []

    stages.append({
        "time": t(0),
        "stage": "Bed fit check",
        "layer_range": "—",
        "nozzle": "Idle — measuring model vs build volume",
        "supports": "none yet",
        "risks": [] if fits == "Yes" else [
            fit.get("message") or "Size problem",
            "Model may need rotation, scaling, or splitting",
        ],
        "watch_for": "Confirm X/Y/Z fit before heating",
        "within_volume": fits == "Yes" or fits == "Maybe",
        "detail": (
            f"Model {dims['x_mm']}×{dims['y_mm']}×{dims['z_mm']} mm vs "
            f"printer {fit.get('printer_mm') or 'UNKNOWN'}. Fits: {fits}."
        ),
    })

    if fits == "No" or not fit.get("bed_known"):
        stages.append({
            "time": t(5),
            "stage": "STOP — resize required",
            "layer_range": "—",
            "nozzle": "Do not start print",
            "supports": "—",
            "risks": ["Printing as-is will hit bed limits or nozzle crash risk on some firmwares"],
            "watch_for": "Apply scale/rotate/split first",
            "within_volume": False,
            "detail": "; ".join(fit.get("options") or [fit.get("message", "")]),
        })
        return {
            "demo_speed": f"{speed}x",
            "total_layers_est": layers,
            "aborted": True,
            "stages": stages,
            "within_volume_throughout": False,
        }

    l_first = max(1, int(0.02 * layers))
    l_lower = max(l_first + 1, int(0.25 * layers))
    l_mid = max(l_lower + 1, int(0.65 * layers))
    l_upper = max(l_mid + 1, int(0.90 * layers))

    stages.append({
        "time": t(5),
        "stage": "First layer",
        "layer_range": f"1–{l_first}",
        "nozzle": "Lays base outline + solid bottom; slow speed, max adhesion",
        "supports": "interface pads if enabled",
        "risks": [
            "Corners may lift if contact area is small",
            "Elephant foot if first layer too squished",
        ],
        "watch_for": "Even extrusion, no gaps, full bed stick",
        "within_volume": True,
        "detail": f"Bed contact ~{printability.get('bed_contact_mm2')} mm²; adhesion: {settings.get('adhesion')}",
    })

    stages.append({
        "time": t(15),
        "stage": "Lower layers",
        "layer_range": f"{l_first + 1}–{l_lower}",
        "nozzle": "Walls build upward; infill begins",
        "supports": "support bases start if needed" if supports else "none",
        "risks": [
            "Thin sections may wobble",
            "Warping risk highest in first 5–10 mm of height",
        ],
        "watch_for": "Layer bonding, no corner lift",
        "within_volume": True,
        "detail": f"Infill {settings.get('infill_pct')}%, walls ×{settings.get('wall_count')}",
    })

    stages.append({
        "time": t(30),
        "stage": "Middle layers",
        "layer_range": f"{l_lower + 1}–{l_mid}",
        "nozzle": "Main body perimeter + infill; steady speed",
        "supports": "supports under overhangs" if supports else "none",
        "risks": [
            "Steep overhangs may sag" if supports else "Low overhang risk",
            "Support-to-model scarring if Z-gap wrong" if supports else "—",
        ],
        "watch_for": "Overhang quality, support stability",
        "within_volume": True,
        "detail": f"Overhang area ~{printability.get('overhang_pct')}%; support={settings.get('support_type')}",
    })

    stages.append({
        "time": t(45),
        "stage": "Upper layers",
        "layer_range": f"{l_mid + 1}–{l_upper}",
        "nozzle": "Bridges and small details; cooling critical",
        "supports": "upper support tips" if supports else "none",
        "risks": [
            "Tiny details may fail or string",
            "Bridges droop without fan" if printability.get("bridge_area_mm2", 0) > 20 else "—",
        ],
        "watch_for": "Stringing, detail loss, fan spinning",
        "within_volume": True,
        "detail": f"Cooling {settings.get('cooling_pct')}%, bridges ~{printability.get('bridge_area_mm2')} mm²",
    })

    stages.append({
        "time": t(60),
        "stage": "Final layers",
        "layer_range": f"{l_upper + 1}–{layers}",
        "nozzle": "Top surfaces close (iron/solid top)",
        "supports": "ready for removal" if supports else "none",
        "risks": [
            "Top surface gaps if too few top layers",
            "Support tear-out on fragile bits" if supports else "—",
        ],
        "watch_for": "Clean top skin; cool before yanking off bed",
        "within_volume": True,
        "detail": "Remove supports carefully; check dimensional accuracy",
    })

    return {
        "demo_speed": f"{speed}x",
        "total_layers_est": layers,
        "aborted": False,
        "stages": stages,
        "within_volume_throughout": fits in ("Yes", "Maybe"),
    }


# ── Visual print map ───────────────────────────────────────────────────────

def visual_print_map(dims: dict, fit: dict, printability: dict, settings: dict) -> str:
    fits = fit.get("fits_label", "?")
    change = fit.get("required_change", "?")
    printer = fit.get("printer_mm")
    printer_s = "×".join(str(int(v)) for v in printer) + " mm" if printer else "UNKNOWN — set bed size"

    lines = [
        "[Build volume check]",
        f"  • Model size:        {dims['x_mm']} × {dims['y_mm']} × {dims['z_mm']} mm",
        f"  • Printer max size:  {printer_s}",
        f"  • Fits as-is:        {fits}",
        f"  • Required change:   {change}",
        "",
        "[Top layers]",
        "  • Small details / top skin",
        f"  • Bridges: ~{printability.get('bridge_area_mm2', 0)} mm²",
        f"  • Overhang risk: {printability.get('overhang_pct', 0)}%",
        "",
        "[Middle layers]",
        "  • Main walls + infill",
        f"  • Infill: {settings.get('infill_pct')}% {settings.get('infill_pattern')}",
        f"  • Supports: {settings.get('support_type')}",
        "",
        "[Bottom layers]",
        "  • First layer + solid bottoms",
        f"  • Bed contact: ~{printability.get('bed_contact_mm2', 0)} mm²",
        f"  • Adhesion: {settings.get('adhesion')}",
    ]
    return "\n".join(lines)


# ── Slicer-style preview text ──────────────────────────────────────────────

def slicer_preview(dims: dict, fit: dict, printability: dict, settings: dict, ctx: PrinterContext) -> str:
    layers = printability.get("layer_count_est", "?")
    lines = [
        f"Slicer-style preview ({ctx.slicer}, {ctx.filament}, {ctx.nozzle_mm} mm nozzle)",
        f"  Layer order: bottom → top · ~{layers} layers @ {settings['layer_height_mm']} mm",
        f"  Orientation: {settings.get('orientation')}",
        f"  Scale: {settings.get('scaling_pct')}%",
        f"  Supports appear: {'under overhangs/bridges' if settings.get('support_type') != 'none' else 'not required'}",
        f"  Infill matters: core strength for {ctx.intended_use} use ({settings.get('infill_pct')}%)",
        f"  Travel/stringing: watch long travels on sparse tops; retraction for {ctx.filament}",
        f"  Failure most likely: first layer adhesion" + (
            " + overhangs" if printability.get("supports_needed") else ""
        ) + (
            " + bed overflow" if fit.get("fits_label") == "No" else ""
        ),
        f"  Physically realistic for printer: "
        + ("YES" if fit.get("fits_label") == "Yes" else ("MARGINAL" if fit.get("fits_label") == "Maybe" else "NO — resize first")),
    ]
    return "\n".join(lines)


# ── Report assembly ────────────────────────────────────────────────────────

def risk_level(fit: dict, health: dict, printability: dict, failures: list) -> str:
    if not fit.get("bed_known") or fit.get("fits_label") == "No":
        return "High"
    if health.get("health") == "poor":
        return "High"
    if fit.get("fits_label") == "Maybe" or printability.get("warp_risk") == "high":
        return "Medium"
    if any(f["id"] not in ("none",) for f in failures if f["id"] != "none"):
        if printability.get("overhang_pct", 0) > 15 or printability.get("thin_feature_risk"):
            return "Medium"
    if health.get("health") == "fair" or printability.get("supports_needed"):
        return "Medium"
    return "Low"


def printable_as_is(fit: dict, health: dict) -> str:
    if not fit.get("bed_known"):
        return "No"
    if fit.get("fits_label") == "No":
        return "No"
    if health.get("health") == "poor":
        return "No"
    if fit.get("fits_label") == "Maybe" or health.get("health") == "fair":
        return "Maybe"
    return "Yes"


def format_report(report: dict) -> str:
    """Human-readable report in the required section order."""
    r = report
    fit = r["fit"]
    lines = []
    lines.append("## Summary")
    lines.append(r["summary"])
    lines.append("")
    lines.append("## Printer / build volume")
    p = r["printer"]
    bed = f"{p['bed_x']}×{p['bed_y']}×{p['bed_z']} mm" if p.get("bed_x") else "UNKNOWN (critical)"
    lines.append(f"  Model: {p.get('label')} ({p.get('printer_model')})")
    lines.append(f"  Build volume: {bed}")
    lines.append(f"  Nozzle: {p.get('nozzle_mm')} mm · Layer: {p.get('layer_height_mm')} mm · {p.get('filament')} · {p.get('slicer')}")
    lines.append(f"  Use: {p.get('intended_use')}")
    lines.append("")
    lines.append("## Model dimensions")
    d = r["model"]
    lines.append(f"  {d['x_mm']} × {d['y_mm']} × {d['z_mm']} mm")
    if d.get("volume_cm3") is not None:
        lines.append(f"  Volume: {d['volume_cm3']} cm³ · {d['face_count']:,} triangles")
    lines.append("")
    lines.append(f"## Fits bed: {r['fits_bed']}")
    lines.append(f"## Required size/config change: {r['required_change']}")
    lines.append(f"## Printable as-is: {r['printable_as_is']}")
    lines.append(f"## Risk level: {r['risk_level']}")
    lines.append(f"## Demo speed: {r['demo']['demo_speed']}")
    lines.append("")
    lines.append("## Demo print walkthrough")
    for s in r["demo"]["stages"]:
        lines.append(f"  {s['time']} — {s['stage']}")
        lines.append(f"    Layers: {s['layer_range']}")
        lines.append(f"    Nozzle: {s['nozzle']}")
        lines.append(f"    Supports: {s['supports']}")
        risks = [x for x in (s.get("risks") or []) if x and x != "—"]
        if risks:
            lines.append(f"    Risk: {'; '.join(risks)}")
        lines.append(f"    Watch: {s['watch_for']}")
        lines.append(f"    Within volume: {'yes' if s.get('within_volume') else 'NO'}")
    lines.append("")
    lines.append("## Visual print map")
    lines.append(r["visual_map"])
    lines.append("")
    lines.append("## Slicer-style preview")
    lines.append(r["slicer_preview"])
    lines.append("")
    lines.append("## Main problems")
    for pbm in r["problems"]:
        lines.append(f"  • {pbm}")
    lines.append("")
    lines.append("## Fixes")
    for fx in r["fixes"]:
        lines.append(f"  • {fx}")
    lines.append("")
    lines.append("## Recommended slicer settings")
    st = r["settings"]
    lines.append(f"  Layer height:     {st['layer_height_mm']} mm")
    lines.append(f"  Wall count:       {st['wall_count']}")
    lines.append(f"  Infill:           {st['infill_pct']}% ({st['infill_pattern']})")
    lines.append(f"  Support:          {st['support_type']} @ {st['support_angle_deg']}°")
    lines.append(f"  Adhesion:         {st['adhesion']}")
    lines.append(f"  Nozzle / bed:     {st['nozzle_temp_c']}°C / {st['bed_temp_c']}°C")
    lines.append(f"  Speed / cooling:  {st['speed_mm_s']} mm/s · {st['cooling_pct']}% fan")
    lines.append(f"  Scaling:          {st['scaling_pct']}%")
    lines.append(f"  Orientation:      {st['orientation']}")
    lines.append("")
    lines.append("## Top failure predictions")
    for i, f in enumerate(r["failures"], 1):
        lines.append(f"  {i}. {f['title']} — {f['detail']}")
    lines.append("")
    lines.append("## Final print recommendation")
    lines.append(r["final_recommendation"])
    return "\n".join(lines)


def analyze_stl(path: str, printer: Optional[dict] = None) -> dict[str, Any]:
    """Full analysis pipeline. Returns structured report + markdown text."""
    ctx = PrinterContext.from_dict(printer)
    mesh = load_mesh(path)
    dims = model_dims(mesh)
    health = inspect_health(mesh, nozzle_mm=ctx.nozzle_mm)
    fit = check_bed_fit(dims, ctx)
    printability = analyze_printability(mesh, ctx)
    settings = recommend_settings(ctx, fit, printability, health)
    failures = predict_failures(fit, health, printability, ctx, settings)
    demo = demo_walkthrough(dims, fit, printability, ctx, settings)
    vmap = visual_print_map(dims, fit, printability, settings)
    sprev = slicer_preview(dims, fit, printability, settings, ctx)

    # problems & fixes
    problems = []
    fixes = []
    if not fit.get("bed_known"):
        problems.append("Build volume unknown — critical info missing")
        fixes.append("Set printer model or enter bed X/Y/Z mm")
    if fit.get("fits_label") == "No":
        problems.append(fit.get("message") or "Does not fit bed")
        fixes.extend(fit.get("options") or [])
    elif fit.get("barely_fits"):
        problems.append("Barely fits — edge clearance / adhesion risk")
        fixes.append("Center on bed; add brim; slow first layer")
    for iss in health.get("issues") or []:
        problems.append(iss["msg"])
        if iss["id"] in ("non_manifold", "open_edges", "flipped_normals"):
            fixes.append("Repair mesh (PrusaSlicer repair, meshnet, or Blender 3D-Print toolbox)")
        elif iss["id"] == "thin_walls":
            fixes.append("Thicken walls to ≥2× nozzle or print with 0.2–0.25 mm nozzle carefully")
        elif iss["id"] == "bad_scale":
            fixes.append("Re-export in mm or scale in slicer (25.4× if inches→mm)")
        elif iss["id"] == "floating_islands":
            fixes.append("Split bodies and print separately, or add supports")
        elif iss["id"] == "excessive_tris":
            fixes.append("Decimate mesh to <200k faces")
    if printability.get("supports_needed"):
        problems.append(f"Overhangs ~{printability['overhang_pct']}% — supports recommended")
        fixes.append(f"Enable {settings['support_type']} supports @ {settings['support_angle_deg']}°")
    if printability.get("bed_contact_mm2", 999) < 100:
        problems.append("Small bed contact area")
        fixes.append(f"Use {settings['adhesion']}")
    if not problems:
        problems.append("No critical issues found")
    if not fixes:
        fixes.append("Print with recommended settings; babysit first layer")

    printable = printable_as_is(fit, health)
    risk = risk_level(fit, health, printability, failures)

    # summary one-liner
    if not fit.get("bed_known"):
        summary = "Blocked: printer bed/build volume unknown. Set bed size before print recommendations."
    elif fit.get("fits_label") == "No":
        summary = (
            f"Does NOT fit {ctx.label} bed "
            f"({dims['x_mm']}×{dims['y_mm']}×{dims['z_mm']} vs "
            f"{ctx.bed_x:.0f}×{ctx.bed_y:.0f}×{ctx.bed_z:.0f} mm). "
            f"Scale ≤{fit.get('safe_scale_pct')}% or rotate/split — do not print as-is."
        )
    elif printable == "Yes":
        summary = (
            f"Fits {ctx.label}; mesh health {health['health']}; "
            f"risk {risk}. Demo @ {ctx.demo_speed}x looks printable with recommended settings."
        )
    else:
        summary = (
            f"Fits with caveats on {ctx.label} (printable: {printable}, risk {risk}). "
            f"See demo for layer risks before printing."
        )

    if printable == "No":
        final = "DO NOT print as-is. Fix bed fit and/or mesh issues first, then re-analyze."
    elif printable == "Maybe":
        final = "Printable with care: apply fixes (brim/supports/repair), babysit first layer, accept medium risk."
    else:
        final = (
            f"OK to print on {ctx.label} with listed settings "
            f"({ctx.filament}, {settings['layer_height_mm']} mm layers, "
            f"{settings['infill_pct']}% infill). Watch first layer."
        )

    report = {
        "summary": summary,
        "printer": {
            "printer_model": ctx.printer_model,
            "label": ctx.label,
            "bed_x": ctx.bed_x,
            "bed_y": ctx.bed_y,
            "bed_z": ctx.bed_z,
            "nozzle_mm": ctx.nozzle_mm,
            "layer_height_mm": ctx.layer_height_mm,
            "filament": ctx.filament,
            "slicer": ctx.slicer,
            "intended_use": ctx.intended_use,
            "demo_speed": ctx.demo_speed,
            "bed_known": ctx.bed_known,
        },
        "model": dims,
        "fits_bed": fit.get("fits_label"),
        "required_change": fit.get("required_change"),
        "printable_as_is": printable,
        "risk_level": risk,
        "fit": fit,
        "health": health,
        "printability": printability,
        "settings": settings,
        "failures": failures,
        "demo": demo,
        "visual_map": vmap,
        "slicer_preview": sprev,
        "problems": problems,
        "fixes": list(dict.fromkeys(fixes)),  # dedupe preserve order
        "final_recommendation": final,
        "source_file": os.path.basename(path),
    }
    report["text"] = format_report(report)
    return report


def list_printers() -> list[dict]:
    out = []
    for pid, p in PRINTER_PROFILES.items():
        out.append({
            "id": pid,
            "label": p["label"],
            "bed_x": p["bed_x"],
            "bed_y": p["bed_y"],
            "bed_z": p["bed_z"],
            "nozzle_mm": p.get("nozzle_mm", 0.4),
        })
    return out


def list_filaments() -> list[str]:
    return list(FILAMENT_PRESETS.keys())


def list_uses() -> list[str]:
    return list(USE_PROFILES.keys())
