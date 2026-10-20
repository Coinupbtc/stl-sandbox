"""ISO inspection certificate + strict generation gates."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import iso_quality
import llm_codegen


def test_iso2768_bins():
    # 50 mm → medium class ±0.3
    assert abs(iso_quality.iso2768_tolerance(50, "iso2768-m") - 0.3) < 1e-9
    # fine tighter than medium
    assert iso_quality.iso2768_tolerance(50, "iso2768-f") < iso_quality.iso2768_tolerance(50, "iso2768-m")
    # coarse looser
    assert iso_quality.iso2768_tolerance(50, "iso2768-c") > iso_quality.iso2768_tolerance(50, "iso2768-m")


def test_check_dimension_pass_fail():
    p = iso_quality.check_dimension(100, 100.2, "iso2768-m", "L")
    assert p["ok"]  # ±0.3
    f = iso_quality.check_dimension(100, 101.0, "iso2768-m", "L")
    assert not f["ok"]


def test_washer_certificate_accept():
    stl = os.path.join(tempfile.gettempdir(), "iso_washer.stl")
    step = os.path.join(tempfile.gettempdir(), "iso_washer.step")
    r = llm_codegen.generate_model("M8 washer 2mm thick", stl, step)
    assert r.get("ok"), r
    cert = r.get("inspection")
    assert cert, "missing inspection certificate"
    assert cert["disposition"] in ("ACCEPT", "ACCEPT_WITH_DEVIATION"), cert
    assert cert["grade"] in ("A", "B")
    assert os.path.exists(stl.replace(".stl", ".quality.json"))
    assert os.path.exists(stl.replace(".stl", ".quality.txt"))
    # text report mentions ISO
    txt = open(stl.replace(".stl", ".quality.txt")).read()
    assert "ISO 2768" in txt
    assert "Disposition" in txt


def test_phone_case_certificate():
    stl = os.path.join(tempfile.gettempdir(), "iso_case.stl")
    step = os.path.join(tempfile.gettempdir(), "iso_case.step")
    r = llm_codegen.generate_model("iphone 17 pro max case", stl, step)
    assert r.get("ok"), r
    cert = r.get("inspection")
    assert cert
    assert cert["disposition"] in ("ACCEPT", "ACCEPT_WITH_DEVIATION")
    # layout + step process controls
    pcs = {p["id"]: p for p in cert.get("process_controls") or []}
    assert pcs.get("brep_step_export", {}).get("ok")
    assert (cert.get("integrity") or {}).get("integrity_ok")


def test_certificate_text_disclaimer():
    cert = {
        "title": "Test",
        "part": {"name": "x", "prompt": "y", "path_used": "template", "stl_sha256_16": "abc"},
        "tolerance_class_name": "ISO 2768-m",
        "disposition": "ACCEPT",
        "grade": "A",
        "dimensional": {"characteristics": []},
        "integrity": {"checks": []},
        "process_controls": [],
        "conformance_summary": "ok",
        "disclaimer": "NOT AS9100",
    }
    t = iso_quality.certificate_text(cert)
    assert "NOT AS9100" in t


def test_should_reject():
    cert = {"disposition": "REJECT", "integrity": {"critical_failures": ["watertight"]}, "grade": "F"}
    msg = iso_quality.should_reject_generation(cert)
    assert msg and "REJECT" in msg


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


def _cert_with_integrity(monkeypatch, checks, **extra):
    import trimesh
    stl = os.path.join(tempfile.gettempdir(), "iso_gate_box.stl")
    trimesh.creation.box(extents=[20, 20, 10]).export(stl)
    step = stl.replace(".stl", ".step")
    open(step, "w").write("ISO-10303-21;")
    crit = {"watertight", "winding_consistent", "no_degenerate_faces", "min_wall_thickness"}
    integ = {
        "checks": checks,
        "integrity_ok": all(c["ok"] for c in checks if c["id"] in crit),
        "critical_failures": [c["id"] for c in checks if c["id"] in crit and not c["ok"]],
        "advisory_failures": [c["id"] for c in checks if c["id"] not in crit and not c["ok"]],
        "not_evaluated": [c["id"] for c in checks if c.get("evaluated") is False],
        "extents_mm": [20.0, 20.0, 10.0], **extra,
    }
    monkeypatch.setattr(iso_quality, "mesh_integrity", lambda *a, **k: integ)
    return iso_quality.build_certificate(name="gate_box", prompt="box", stl_path=stl, step_path=step,
        size_contract={"height_mm": 10, "source": "test"})


def test_failed_noncritical_check_is_not_grade_a(monkeypatch):
    cert = _cert_with_integrity(monkeypatch, [
        {"id": "watertight", "ok": True, "detail": ""},
        {"id": "triangle_quality", "ok": False, "detail": "504 high-aspect triangles in sample of 1032"},
    ])
    assert cert["grade"] == "B" and cert["disposition"] == "ACCEPT_WITH_DEVIATION", cert["grade"]
    assert "triangle_quality" in cert["conformance_summary"]
    assert "integrity gates passed" not in cert["conformance_summary"]


def test_all_checks_ok_still_grade_a(monkeypatch):
    cert = _cert_with_integrity(monkeypatch, [
        {"id": "watertight", "ok": True, "detail": ""},
        {"id": "triangle_quality", "ok": True, "detail": ""},
    ])
    assert cert["grade"] == "A"


def test_unevaluated_check_is_named_in_summary(monkeypatch):
    cert = _cert_with_integrity(monkeypatch, [
        {"id": "watertight", "ok": True, "detail": ""},
        {"id": "min_wall_thickness", "ok": True, "evaluated": False, "detail": "not estimated (ray engine unavailable)"},
    ])
    assert "Not evaluated: min_wall_thickness" in cert["conformance_summary"]
