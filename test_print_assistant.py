#!/usr/bin/env python3
"""Smoke tests for Print Assistant — bed fit, oversized model, missing bed."""

import json
import os
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import print_assistant as pa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


def write_box(path, extents):
    m = trimesh.creation.box(extents=extents)
    m.export(path)


def main():
    print("=== Print Assistant smoke tests ===")

    # 1) cube_40 on Ender 3 — must fit
    cube = os.path.join(OUT, "cube_40.stl")
    if not os.path.exists(cube):
        write_box(cube, (40, 40, 40))
    r = pa.analyze_stl(cube, {"printer_model": "ender3", "filament": "PLA", "intended_use": "prototype", "demo_speed": 10})
    check("cube fits Ender3", r["fits_bed"] == "Yes", r["fits_bed"])
    check("cube printable", r["printable_as_is"] in ("Yes", "Maybe"), r["printable_as_is"])
    check("cube demo not aborted", r["demo"]["aborted"] is False)
    check("cube has settings", r["settings"]["layer_height_mm"] == 0.2)
    check("report text has Summary", "Summary" in r["text"])

    # 2) oversized model — must NOT fit, must recommend scale
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
        big_path = tf.name
    write_box(big_path, (280, 220, 180))
    try:
        r2 = pa.analyze_stl(big_path, {
            "printer_model": "ender3",  # 220x220x250
            "filament": "PLA",
            "demo_speed": 10,
        })
        check("oversized does not fit", r2["fits_bed"] == "No", r2["fits_bed"])
        check("oversized not printable as-is", r2["printable_as_is"] == "No")
        check("safe scale < 100%", r2["fit"]["safe_scale_pct"] < 100, str(r2["fit"]["safe_scale_pct"]))
        # Best-rotation scale is better than as-is (~75%); expect ≤90% with clearance
        check("safe scale ≤90%", r2["fit"]["safe_scale_pct"] <= 90, str(r2["fit"]["safe_scale_pct"]))
        check("demo aborted on no-fit", r2["demo"]["aborted"] is True)
        check("options mention scale or split", any("cale" in o or "plit" in o for o in r2["fit"]["options"]))
        check("final says DO NOT", "DO NOT" in r2["final_recommendation"])
    finally:
        os.unlink(big_path)

    # 3) unknown bed — critical missing info
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
        p = tf.name
    write_box(p, (50, 50, 50))
    try:
        r3 = pa.analyze_stl(p, {"printer_model": "custom", "bed_x": None, "bed_y": None, "bed_z": None})
        check("unknown bed → Unknown fit", r3["fits_bed"] == "Unknown", r3["fits_bed"])
        check("unknown bed not printable", r3["printable_as_is"] == "No")
        check("unknown bed high risk", r3["risk_level"] == "High")
        check("message mentions missing", "missing" in r3["fit"]["message"].lower() or "unknown" in r3["summary"].lower())
    finally:
        os.unlink(p)

    # 4) barely fits — 215x215x20 on 220 bed with 5mm clearance
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
        p = tf.name
    write_box(p, (215, 215, 20))
    try:
        r4 = pa.analyze_stl(p, {"printer_model": "ender3", "clearance_mm": 5})
        check("barely fits → Maybe or No", r4["fits_bed"] in ("Maybe", "No"), r4["fits_bed"])
    finally:
        os.unlink(p)

    # 5) rotation can save fit — 250x100x50 on 220x220x250: as-is X too big, rotate may help if we swap
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf:
        p = tf.name
    write_box(p, (250, 100, 50))
    try:
        r5 = pa.analyze_stl(p, {"printer_model": "ender3"})
        check("long part as-is No", r5["fits_bed"] == "No", r5["fits_bed"])
        # best rotation 100x250x50 still no; 250x50x100 no; 100x50x250 fits Z?
        # 100x50x250: Z=250 equals bed Z with clearance 5 → usable Z=245 → still no
        # scale needed
        check("rotation or scale offered", len(r5["fit"]["options"]) >= 1)
    finally:
        os.unlink(p)

    # 6) filament temps
    r6 = pa.analyze_stl(cube, {"printer_model": "prusa_mk4", "filament": "PETG", "intended_use": "functional"})
    check("PETG nozzle ~240", r6["settings"]["nozzle_temp_c"] == 240)
    check("functional walls ≥3", r6["settings"]["wall_count"] >= 3)

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
