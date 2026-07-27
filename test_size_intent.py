"""Tests for universal size contract + new templates + routing."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import organic_gen
import product_templates
import size_intent
import llm_codegen


def test_parse_cube():
    s = size_intent.parse_prompt_sizes("a 40mm cube with a 10mm hole through it")
    assert s.get("height_mm") == 40.0
    assert s.get("bore_mm") == 10.0


def test_credit_card_catalog():
    c = size_intent.resolve_size_intent("credit card sized wallet tray")
    assert c is not None
    assert abs(c["width_mm"] - 85.6) < 0.2 or abs(c.get("width_mm", 0) - 90) < 5
    assert c.get("expected_bbox_mm")


def test_soda_can_catalog():
    c = size_intent.resolve_size_intent("soda can holder")
    assert c is not None
    assert abs(float(c["diameter_mm"]) - 66.0) < 0.5


def test_m8_washer_metric():
    c = size_intent.resolve_size_intent("M8 washer 2mm thick")
    assert c is not None
    assert c.get("diameter_mm") == 16.0 or c.get("diameter_mm", 0) >= 14
    assert c.get("depth_mm") == 2.0


def test_size_brief_mentions_contract():
    c = size_intent.resolve_size_intent("coffee mug 85 mm tall")
    brief = size_intent.format_size_brief(c)
    assert "SIZE CONTRACT" in brief
    assert "85" in brief


def test_organic_does_not_steal_functional():
    assert organic_gen.is_organic_prompt("a cute cat figurine") is True
    assert organic_gen.is_organic_prompt("phone stand at 60 degrees") is False
    assert organic_gen.is_organic_prompt("soap dish with drain holes") is False
    assert organic_gen.is_organic_prompt("M8 washer") is False
    assert organic_gen.is_organic_prompt("cat phone stand") is False  # functional wins


def test_washer_template_exec():
    code = product_templates.match_template("M8 washer 2mm thick", None)
    assert code and "washer" in code.lower()
    stl = os.path.join(tempfile.gettempdir(), "si_washer.stl")
    step = os.path.join(tempfile.gettempdir(), "si_washer.step")
    run = llm_codegen.run_cadquery_code(code, stl, step)
    assert run.get("ok"), run
    assert run["stats"]["watertight"]
    bbox = sorted(run["stats"]["bbox_mm"])
    assert bbox[0] <= 3.0  # thickness
    assert 14 <= bbox[-1] <= 20  # OD ~16


def test_soap_dish_template_exec():
    code = product_templates.match_template("soap dish with drain holes", None)
    assert code
    stl = os.path.join(tempfile.gettempdir(), "si_soap.stl")
    step = os.path.join(tempfile.gettempdir(), "si_soap.step")
    run = llm_codegen.run_cadquery_code(code, stl, step)
    assert run.get("ok"), run
    assert run["stats"]["watertight"]
    bbox = sorted(run["stats"]["bbox_mm"])
    # Shallow dish: height ≤ 20 mm; footprint fits a bar of soap with margin
    assert bbox[0] <= 20, bbox
    assert bbox[-1] >= 100, bbox


def test_can_holder_template_exec():
    code = product_templates.match_template("soda can holder", None)
    assert code
    stl = os.path.join(tempfile.gettempdir(), "si_can.stl")
    step = os.path.join(tempfile.gettempdir(), "si_can.step")
    run = llm_codegen.run_cadquery_code(code, stl, step)
    assert run.get("ok"), run
    # outer OD ~72; height partial sleeve ~40–90
    bb = run["stats"]["bbox_mm"]
    assert max(bb) <= 95, bb
    assert min(x for x in bb if x > 20) >= 65  # diameter axes


def test_phone_stand_template_no_research():
    code = product_templates.match_template(
        "a phone stand at 60 degrees with a lip", None
    )
    assert code and "stand" in code.lower()
    # Must actually lean — dead ANGLE constants are not quality
    assert "rotate" in code and "60" in code
    stl = os.path.join(tempfile.gettempdir(), "si_stand.stl")
    step = os.path.join(tempfile.gettempdir(), "si_stand.step")
    run = llm_codegen.run_cadquery_code(code, stl, step)
    assert run.get("ok"), run
    assert run["stats"]["watertight"]
    assert run["stats"]["faces"] >= 12
    # Stand footprint should be desk-sized, not a phone wafer
    assert max(run["stats"]["bbox_mm"]) >= 70
    assert min(run["stats"]["bbox_mm"]) >= 6


def test_generate_model_washer_template_path():
    stl = os.path.join(tempfile.gettempdir(), "si_e2e_washer.stl")
    step = os.path.join(tempfile.gettempdir(), "si_e2e_washer.step")
    r = llm_codegen.generate_model("M8 washer 2mm thick", stl, step)
    assert r.get("ok"), r
    assert r.get("path_detail") == "template"
    assert r.get("size_intent") is not None


def test_quality_rejects_solid_brick_case_warning():
    reason = size_intent.quality_should_retry(
        {"faces": 64, "watertight": True, "bbox_mm": [81, 160, 75]},
        ["Solid fills >95% of its bounding box — if you asked for a case/enclosure, "
         "this may be a solid brick instead of a hollow shell"],
        "iphone case",
    )
    assert reason and "brick" in reason


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
