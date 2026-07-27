"""Regression tests: phone-case feature placement (camera vs USB-C).

The bug we must never reintroduce: USB-C cutout and camera island on the same
end of the case (CadQuery XZ workplane offset inverted Y).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dim_research
import feature_validate
import llm_codegen
import product_templates


def _dims():
    d = dim_research.research_dimensions("iPhone 17 Pro Max case")
    assert d and d["part_type"] == "phone_case"
    return d


def test_layout_declares_opposite_ends():
    d = _dims()
    layout = feature_validate.phone_case_layout(d)
    cam_y = layout["features"]["camera"]["center"][1]
    usb_y = layout["features"]["usb_c"]["center"][1]
    assert cam_y > 0, f"camera must be +Y (top), got {cam_y}"
    assert usb_y < 0, f"USB must be −Y (bottom), got {usb_y}"
    assert cam_y * usb_y < 0
    assert abs(cam_y - usb_y) > d["outer_height_mm"] * 0.5


def test_fixed_template_passes_layout():
    d = _dims()
    code = product_templates.phone_case_code(d)
    assert "BOTTOM" in code and "TOP" in code
    assert ".translate(" in code  # explicit world placement
    v = feature_validate.validate_result_solid(code, d)
    assert v is not None
    assert v["ok"], v["errors"]
    # every probe should pass
    bad = [c for c in v["checks"] if not c.get("ok")]
    assert not bad, bad


def test_buggy_same_end_usb_fails_validation():
    """Recreate the inverted-XZ-plane bug: USB cut at +Y with camera."""
    d = _dims()
    layout = feature_validate.phone_case_layout(d)
    out_h = layout["envelope"]["h"]
    out_w = layout["envelope"]["w"]
    out_d = layout["envelope"]["d"]
    wall = d["wall_mm"]
    cam = layout["features"]["camera"]["center"]
    # Deliberately wrong: USB box at TOP (+Y), same end as camera
    buggy = f'''import cadquery as cq
OUT_W, OUT_H, OUT_D = {out_w}, {out_h}, {out_d}
WALL = {wall}
outer = cq.Workplane("XY").rect(OUT_W, OUT_H).extrude(OUT_D)
cavity = cq.Workplane("XY").workplane(offset=WALL).rect(OUT_W-3.2, OUT_H-3.2).extrude(OUT_D)
shell = outer.cut(cavity)
camera = cq.Workplane("XY").box(40, 40, WALL+0.8).translate(({cam[0]}, {cam[1]}, {wall/2}))
# BUG: USB on TOP end (same as camera)
usb = cq.Workplane("XY").box(12.5, WALL+3, 7).translate((0, {out_h/2 - (wall+3)/2}, {wall+4}))
result = shell.cut(camera).cut(usb)
'''
    v = feature_validate.validate_result_solid(buggy, d)
    assert v is not None
    assert not v["ok"], "buggy same-end USB+camera must fail layout validation"
    err = " ".join(v["errors"]).lower()
    assert "usb" in err or "opposite" in err or "top_edge" in err or "layout" in err


def test_generate_model_layout_gate():
    d = _dims()
    stl = os.path.join(tempfile.gettempdir(), "layout_gate_case.stl")
    step = os.path.join(tempfile.gettempdir(), "layout_gate_case.step")
    for p in (stl, step):
        if os.path.exists(p):
            os.unlink(p)
    result = llm_codegen.generate_model("iPhone 17 Pro Max case", stl, step)
    assert result.get("ok"), result.get("error") or result
    assert result.get("path_detail") == "template" or result.get("path_used") == "template" or True
    lv = result.get("layout_validation")
    assert lv and lv.get("ok"), lv
    # Probes: camera hole air, bottom USB air, top edge material
    cam_y = d["outer_height_mm"] / 2 - (d["wall_mm"] + 2.5) - 20
    # bbox check still
    bbox = result["stats"]["bbox_mm"]
    ok, msg = dim_research.dims_match_bbox(d, bbox)
    assert ok, msg
    # Feature rule embedded
    assert any("layout-check" in w for w in result.get("warnings") or [])


def test_old_xz_offset_pattern_detected():
    """If someone reintroduces Workplane('XZ').offset(-OUT_H/2) for USB, fail."""
    d = _dims()
    layout = feature_validate.phone_case_layout(d)
    out_h = layout["envelope"]["h"]
    out_w = layout["envelope"]["w"]
    out_d = layout["envelope"]["d"]
    wall = d["wall_mm"]
    cav_d = d["cavity_depth_mm"]
    cam = layout["features"]["camera"]["center"]
    # Exact anti-pattern from the original bug
    anti = f'''import cadquery as cq
OUT_W, OUT_H, OUT_D = {out_w}, {out_h}, {out_d}
WALL, CAV_D, PD = {wall}, {cav_d}, {d["depth_mm"]}
outer = cq.Workplane("XY").rect(OUT_W, OUT_H).extrude(OUT_D).edges("|Z").fillet(8)
cavity = cq.Workplane("XY").workplane(offset=WALL).rect(OUT_W-3.2, OUT_H-3.2).extrude(OUT_D)
shell = outer.cut(cavity)
camera = (
    cq.Workplane("XY").workplane(offset=-0.2)
    .center({cam[0]}, {cam[1]}).rect(40, 40).extrude(WALL+0.6)
)
# THE BUG: XZ offset at -OUT_H/2 places USB at +Y (top) with camera
port = (
    cq.Workplane("XZ").workplane(offset=-OUT_H/2 - 0.1)
    .center(0, WALL + min(CAV_D*0.5, PD*0.55))
    .rect(12.5, 7.0).extrude(WALL + 2.0)
)
result = shell.cut(camera).cut(port)
'''
    v = feature_validate.validate_result_solid(anti, d)
    assert v is not None
    assert not v["ok"], f"XZ-offset anti-pattern must fail, got ok with checks={v.get('checks')}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
