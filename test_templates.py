"""Smoke tests for household + product parametric templates."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dim_research
import feature_validate
import llm_codegen
import product_templates


def _run(code: str, tag: str) -> dict:
    stl = os.path.join(tempfile.gettempdir(), f"stl_tmpl_{tag}.stl")
    step = os.path.join(tempfile.gettempdir(), f"stl_tmpl_{tag}.step")
    for p in (stl, step):
        if os.path.exists(p):
            os.unlink(p)
    run = llm_codegen.run_cadquery_code(code, stl, step)
    assert run.get("ok"), f"{tag}: {run}"
    assert run["stats"]["watertight"], f"{tag} not watertight"
    assert run["stats"]["faces"] > 20, f"{tag} too few faces"
    return run


def test_parse_sizes_mug():
    s = product_templates.parse_sizes("a coffee mug with a handle, 85 mm tall")
    assert s.get("height_mm") == 85.0


def test_parse_sizes_pen_holder():
    s = product_templates.parse_sizes(
        "desk pen holder: cup about 80mm diameter, 100mm tall, 3mm walls, hexagonal outside"
    )
    assert s.get("height_mm") == 100.0
    assert s.get("diameter_mm") == 80.0
    assert s.get("wall_mm") == 3.0


def test_parse_sizes_box_triple():
    s = product_templates.parse_sizes("project box 100x60x40 with lid")
    assert s.get("width_mm") == 100.0
    assert s.get("depth_mm") == 60.0
    assert s.get("height_mm") == 40.0


def test_parse_sizes_hex_nut():
    s = product_templates.parse_sizes("a hex nut, 24 mm across flats, M10 bore")
    assert s.get("across_flats_mm") == 24.0
    assert s.get("bore_mm") == 10.0


def test_mug_template():
    code = product_templates.match_template("a coffee mug with a handle, 85 mm tall", None)
    assert code and "Parametric coffee mug" in code
    run = _run(code, "mug")
    bbox = run["stats"]["bbox_mm"]
    # body ~80 dia × 85 tall; handle extends ~100+ on one axis
    assert 80 in [round(x) for x in bbox] or any(78 <= x <= 82 for x in bbox)
    assert any(83 <= x <= 87 for x in bbox), bbox
    assert max(bbox) >= 95  # handle stick-out


def test_pen_holder_template():
    code = product_templates.match_template(
        "desk pen holder: cup about 80mm diameter, 100mm tall, 3mm walls, hexagonal outside",
        None,
    )
    assert code and "pen holder" in code.lower()
    run = _run(code, "pen")
    assert max(run["stats"]["bbox_mm"]) >= 95


def test_project_box_template():
    code = product_templates.match_template("project box 100x60x40 with lid", None)
    assert code and "Project" in code
    run = _run(code, "box")
    assert run["stats"]["faces"] > 50


def test_cable_clip_template():
    code = product_templates.match_template("desk cable clip for 6mm cord", None)
    assert code
    _run(code, "clip")


def test_wall_hook_template():
    code = product_templates.match_template("wall hook with screw holes", None)
    assert code
    _run(code, "hook")


def test_hex_nut_template():
    code = product_templates.match_template(
        "a hex nut, 24 mm across flats, M10 bore", None
    )
    assert code
    run = _run(code, "nut")
    bbox = sorted(run["stats"]["bbox_mm"])
    # height ~AF*0.45, AF=24, vertex span ~27.7
    assert 8 <= bbox[0] <= 15, bbox   # thickness
    assert 22 <= bbox[1] <= 30, bbox  # across flats
    assert 24 <= bbox[2] <= 32, bbox  # vertex-to-vertex


def test_desk_tray_template():
    code = product_templates.match_template(
        "a desk tray 120x80 mm with 2 compartments", None
    )
    assert code and "compartment" in code.lower()
    _run(code, "tray")


def test_phone_case_still_works_with_magsafe():
    d = dim_research.research_dimensions("iPhone 17 Pro Max case")
    code = product_templates.match_template("iPhone 17 Pro Max case", d)
    assert code and "MagSafe" in code
    run = _run(code, "case")
    ok, msg = dim_research.dims_match_bbox(d, run["stats"]["bbox_mm"])
    assert ok, msg
    layout = feature_validate.validate_result_solid(code, d)
    assert layout.get("ok"), layout.get("errors")


def test_generate_model_mug_is_template_path():
    """End-to-end: mug must hit template path, not LLM (works offline)."""
    stl = os.path.join(tempfile.gettempdir(), "stl_e2e_mug.stl")
    step = os.path.join(tempfile.gettempdir(), "stl_e2e_mug.step")
    for p in (stl, step):
        if os.path.exists(p):
            os.unlink(p)
    result = llm_codegen.generate_model(
        "a coffee mug with a comfortable handle, 85 mm tall", stl, step
    )
    assert result.get("ok"), result
    assert result.get("path_detail") == "template", result
    assert result["elapsed_s"] < 30


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
