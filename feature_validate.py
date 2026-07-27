"""Semantic feature-layout validation for product-fit parts.

Catches logic bugs the mesh stats miss — e.g. USB-C and camera both on the
top edge because a CadQuery workplane normal was inverted.

Convention for phone cases (world mm, case centered on X/Y, back on Z=0):
  X = width   (−X left, +X right) when phone face-up, top away from you
  Y = height  (−Y bottom/charging, +Y top/earpiece)
  Z = thickness (−Z back plate at 0, +Z screen/lip)

Probe points declare where air (hole) vs material (wall) MUST be. Validation
runs on the CadQuery solid via Shape.isInside — no mesh/ray deps.
"""

from __future__ import annotations

from typing import Any, Optional


# ─── Phone-case layout builder ───

def phone_case_layout(dims: dict) -> dict:
    """Return feature centers + probe points derived from researched dims."""
    wall = float(dims.get("wall_mm", 1.6))
    clr = float(dims.get("clearance_mm", 0.4))
    lip = float(dims.get("lip_mm", 1.2))
    ph = float(dims["height_mm"])
    pw = float(dims["width_mm"])
    pd = float(dims["depth_mm"])

    cav_h = ph + 2 * clr
    cav_w = pw + 2 * clr
    cav_d = pd + clr
    out_h = cav_h + 2 * wall
    out_w = cav_w + 2 * wall
    out_d = wall + cav_d + lip

    cam_s = 40.0 if pw >= 76 else 34.0
    cam_m = wall + 2.5
    # Camera island: BACK face, TOP-LEFT (face-up, top = +Y)
    cam_cx = -out_w / 2 + cam_m + cam_s / 2
    cam_cy = out_h / 2 - cam_m - cam_s / 2
    cam_cz = wall * 0.5  # mid back wall

    port_w, port_h = 12.5, 7.0
    # USB-C: BOTTOM edge (−Y), mid phone thickness
    port_cx = 0.0
    port_cy = -out_h / 2 + wall * 0.5
    port_cz = wall + min(cav_d * 0.5, pd * 0.55)

    # Side buttons: volume on LEFT (−X), power on RIGHT (+X), upper half
    vol_cy = out_h * 0.12
    pwr_cy = out_h * 0.06
    btn_cz = wall + cav_d * 0.45

    features = {
        "camera": {
            "face": "back",
            "end": "top",
            "side": "left",
            "center": (cam_cx, cam_cy, cam_cz),
            "size": (cam_s, cam_s, wall + 0.6),
        },
        "usb_c": {
            "face": "bottom",
            "end": "bottom",
            "side": "center",
            "center": (port_cx, port_cy, port_cz),
            "size": (port_w, wall + 3.0, port_h),
        },
        "volume": {
            "face": "left",
            "end": "upper",
            "center": (-out_w / 2 + wall * 0.3, vol_cy, btn_cz),
        },
        "power": {
            "face": "right",
            "end": "upper",
            "center": (out_w / 2 - wall * 0.3, pwr_cy, btn_cz),
        },
        "screen": {
            "face": "front",
            "center": (0.0, 0.0, out_d - lip * 0.3),
        },
    }

    # Probes: (x,y,z, expect_material: bool, label)
    # expect_material True  → point must be INSIDE the solid (wall)
    # expect_material False → point must be OUTSIDE (hole / cavity / air)
    probes = [
        # Back wall present at center
        (0.0, 0.0, wall * 0.5, True, "back_wall_center"),
        # Camera hole through back at top-left
        (cam_cx, cam_cy, wall * 0.5, False, "camera_hole_top_left_back"),
        # Mirror of camera at bottom-left back MUST be solid (no hole at bottom)
        (cam_cx, -cam_cy, wall * 0.5, True, "back_bottom_left_no_camera"),
        # USB-C hole at bottom edge
        (port_cx, -out_h / 2 + wall * 0.4, port_cz, False, "usb_hole_bottom_edge"),
        # Top edge mid thickness MUST be solid (USB must NOT be here)
        (0.0, out_h / 2 - wall * 0.4, port_cz, True, "top_edge_no_usb"),
        # Bottom edge away from port still solid
        (out_w * 0.3, -out_h / 2 + wall * 0.4, port_cz, True, "bottom_edge_beside_port"),
        # Cavity air
        (0.0, 0.0, wall + cav_d * 0.5, False, "cavity_air"),
        # Screen opening
        (0.0, 0.0, out_d - 0.3, False, "screen_opening"),
        # Camera and USB must be on opposite ends of long axis
        # (encoded as the probes above — also explicit separation rule below)
    ]

    return {
        "part_type": "phone_case",
        "axes": {
            "X": "width (− left, + right)",
            "Y": "height (− bottom/USB, + top/camera)",
            "Z": "thickness (0 back, + screen)",
        },
        "envelope": {"w": out_w, "h": out_h, "d": out_d},
        "features": features,
        "probes": probes,
        # Hard separation rules on feature centers (mm along long axis Y)
        "rules": [
            {
                "name": "usb_and_camera_opposite_ends",
                "a": "camera",
                "b": "usb_c",
                "axis": 1,  # Y
                "min_separation_frac": 0.5,  # centers must be > 50% of height apart
                "require_opposite_sign": True,  # one +Y, one −Y
            },
            {
                "name": "camera_on_back_not_front",
                "feature": "camera",
                "z_max": wall + 1.0,  # camera center Z must be in back wall band
            },
            {
                "name": "usb_on_bottom_edge",
                "feature": "usb_c",
                "y_max": -out_h * 0.35,  # USB center Y must be in bottom third
            },
            {
                "name": "camera_on_top_half",
                "feature": "camera",
                "y_min": out_h * 0.15,
            },
        ],
    }


def _is_inside(shape, point: tuple[float, float, float], tol: float = 0.15) -> bool:
    """True if point is in solid material."""
    try:
        return bool(shape.isInside(point, tolerance=tol))
    except TypeError:
        # older cadquery: isInside(Vector) or tuple variants
        import cadquery as cq
        return bool(shape.isInside(cq.Vector(*point)))
    except Exception:
        # Fallback OCP classifier
        from OCP.gp import gp_Pnt
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON

        solid = shape.wrapped if hasattr(shape, "wrapped") else shape
        clf = BRepClass3d_SolidClassifier(solid)
        clf.Perform(gp_Pnt(*point), 1e-3)
        st = clf.State()
        return st in (TopAbs_IN, TopAbs_ON)


def _normalize_shape(result) -> Any:
    import cadquery as cq

    if isinstance(result, cq.Workplane):
        vals = [v for v in result.vals() if isinstance(v, cq.Shape)]
        if not vals:
            raise ValueError("no solid in Workplane")
        return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if hasattr(result, "isInside"):
        return result
    raise ValueError(f"unsupported result type: {type(result)}")


def validate_layout(shape_or_wp, layout: dict) -> dict:
    """Run probe + rule checks. Returns {ok, errors[], checks[]}."""
    errors: list[str] = []
    checks: list[dict] = []
    try:
        shape = _normalize_shape(shape_or_wp)
    except Exception as e:
        return {"ok": False, "errors": [f"shape normalize failed: {e}"], "checks": []}

    # Probe points
    for item in layout.get("probes") or []:
        x, y, z, expect_mat, label = item
        try:
            inside = _is_inside(shape, (x, y, z))
        except Exception as e:
            errors.append(f"probe {label}: classification failed ({e})")
            checks.append({"label": label, "ok": False, "error": str(e)})
            continue
        ok = inside if expect_mat else (not inside)
        checks.append({
            "label": label,
            "ok": ok,
            "inside": inside,
            "expect_material": expect_mat,
            "point": [round(x, 2), round(y, 2), round(z, 2)],
        })
        if not ok:
            want = "material" if expect_mat else "air/hole"
            got = "material" if inside else "air/hole"
            errors.append(
                f"layout probe '{label}' at ({x:.1f},{y:.1f},{z:.1f}): "
                f"expected {want}, got {got}"
            )

    # Feature separation / placement rules
    feats = layout.get("features") or {}
    for rule in layout.get("rules") or []:
        name = rule["name"]
        if "a" in rule and "b" in rule:
            fa, fb = feats.get(rule["a"]), feats.get(rule["b"])
            if not fa or not fb:
                continue
            ca, cb = fa["center"], fb["center"]
            axis = rule.get("axis", 1)
            sep = abs(ca[axis] - cb[axis])
            env = layout.get("envelope") or {}
            span = env.get("h") or env.get("w") or 100.0
            min_sep = rule.get("min_separation_frac", 0.5) * span
            ok_sep = sep >= min_sep
            ok_sign = True
            if rule.get("require_opposite_sign"):
                ok_sign = (ca[axis] * cb[axis]) < 0
            ok = ok_sep and ok_sign
            checks.append({
                "label": name,
                "ok": ok,
                "separation_mm": round(sep, 2),
                "min_mm": round(min_sep, 2),
                "opposite_sign": ok_sign,
            })
            if not ok:
                errors.append(
                    f"rule '{name}': {rule['a']} center Y={ca[axis]:.1f}, "
                    f"{rule['b']} center Y={cb[axis]:.1f}, separation {sep:.1f} mm "
                    f"(need ≥{min_sep:.1f} mm on opposite ends). "
                    f"USB-C and camera must NOT share the same end of the case."
                )
        elif rule.get("y_max") is not None:
            f = feats.get(rule["feature"])
            if f:
                y = f["center"][1]
                ok = y <= rule["y_max"]
                checks.append({"label": name, "ok": ok, "y": round(y, 2), "y_max": rule["y_max"]})
                if not ok:
                    errors.append(
                        f"rule '{name}': {rule['feature']} Y={y:.1f} exceeds "
                        f"y_max={rule['y_max']:.1f} (feature on wrong end)"
                    )
        elif rule.get("y_min") is not None:
            f = feats.get(rule["feature"])
            if f:
                y = f["center"][1]
                ok = y >= rule["y_min"]
                checks.append({"label": name, "ok": ok, "y": round(y, 2), "y_min": rule["y_min"]})
                if not ok:
                    errors.append(
                        f"rule '{name}': {rule['feature']} Y={y:.1f} below "
                        f"y_min={rule['y_min']:.1f} (feature on wrong end)"
                    )
        elif rule.get("z_max") is not None:
            f = feats.get(rule["feature"])
            if f:
                z = f["center"][2]
                ok = z <= rule["z_max"]
                checks.append({"label": name, "ok": ok, "z": round(z, 2), "z_max": rule["z_max"]})
                if not ok:
                    errors.append(
                        f"rule '{name}': {rule['feature']} Z={z:.1f} exceeds "
                        f"z_max={rule['z_max']:.1f} (camera must be on BACK, not front)"
                    )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "checks": checks,
        "layout_axes": layout.get("axes"),
    }


def layout_for_dims(dims: Optional[dict]) -> Optional[dict]:
    if not dims:
        return None
    if dims.get("part_type") == "phone_case":
        return phone_case_layout(dims)
    return None


def validate_result_solid(code: str, dims: Optional[dict]) -> Optional[dict]:
    """Execute CadQuery code in-process (trusted templates only) and validate layout.

    For untrusted LLM code, prefer validate via cq_runner export + re-load, or
    call validate_layout on a shape already built in the runner.
    Returns None if no layout applies.
    """
    layout = layout_for_dims(dims)
    if not layout:
        return None

    import math
    import cadquery as cq

    ns = {"cq": cq, "cadquery": cq, "math": math, "__builtins__": __builtins__}
    exec(compile(code, "<feature_validate>", "exec"), ns)  # noqa: S102 — local CAD only
    result = ns.get("result")
    if result is None:
        return {"ok": False, "errors": ["no result variable"], "checks": []}
    return validate_layout(result, layout)
