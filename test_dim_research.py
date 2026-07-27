"""Smoke tests for dimension research + product templates."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dim_research
import product_templates
import llm_codegen


def test_catalog_iphone_17_pro_max():
    d = dim_research.research_dimensions("iphone 17 pro max cell phone casing")
    assert d is not None
    assert d["height_mm"] == 163.4
    assert d["width_mm"] == 78.0
    assert abs(d["depth_mm"] - 8.75) < 0.01
    assert d["part_type"] == "phone_case"
    assert d["confidence"] == "high"
    assert d["outer_width_mm"] > d["width_mm"]
    assert d["outer_height_mm"] > d["height_mm"]


def test_extract_product_query():
    q = dim_research.extract_product_query("make me an iPhone 17 Pro Max protective case")
    assert "iphone 17 pro max" in q.lower()


def test_needs_research():
    assert dim_research.needs_dimension_research("galaxy s25 ultra case")
    assert not dim_research.needs_dimension_research("a simple cube 40mm")


def test_phone_case_template_executes():
    d = dim_research.research_dimensions("iPhone 17 Pro Max case")
    code = product_templates.match_template("iPhone 17 Pro Max case", d)
    assert code and "163.4" in code
    stl = os.path.join(tempfile.gettempdir(), "test_case.stl")
    step = os.path.join(tempfile.gettempdir(), "test_case.step")
    run = llm_codegen.run_cadquery_code(code, stl, step)
    assert run.get("ok"), run
    assert run["stats"]["watertight"]
    assert run["stats"]["faces"] > 500
    ok, msg = dim_research.dims_match_bbox(d, run["stats"]["bbox_mm"])
    assert ok, msg
    # Old broken model was ~81×160×75 with 64 faces — ensure we don't regress
    bbox = sorted(run["stats"]["bbox_mm"])
    assert bbox[0] < 20, f"case thickness should be thin, got {bbox}"
    assert bbox[-1] > 150, f"case should be phone-length, got {bbox}"


def test_format_brief_contains_must_use():
    d = dim_research.research_dimensions("iphone 16 pro case")
    brief = dim_research.format_dimension_brief(d)
    assert "RESEARCHED REAL-WORLD DIMENSIONS" in brief
    assert "149.6" in brief or str(d["height_mm"]) in brief


def test_dims_match_bbox_detects_bad():
    d = dim_research.research_dimensions("iphone 17 pro max case")
    ok, _ = dim_research.dims_match_bbox(d, [81, 160, 75])  # old garbage
    assert not ok
    ok2, _ = dim_research.dims_match_bbox(d, d["expected_bbox_mm"])
    assert ok2


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
