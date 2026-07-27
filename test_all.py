#!/usr/bin/env python3
"""Test all 11 primitives, upload fix, and STEP export."""
import glob
import os
import time
from pathlib import Path

import requests

BASE_URL = os.environ.get("STL_SANDBOX_URL", "http://127.0.0.1:8050")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Clear output dir
for f in list(OUT.iterdir()):
    if f.is_file():
        f.unlink()
print("Cleared output directory")

# Test 1: Config endpoint
print("\n=== Test 1: /api/config ===")
r = requests.get(f"{BASE_URL}/api/config")
config = r.json()
shapes = [s["name"] for s in config["shapes"]]
print(f"Available shapes: {shapes}")
print(f"Boolean ops: {len(config['boolean_ops'])}")
print(f"Assembly ops: {len(config['assembly_ops'])}")

# Test 2: Generate all 11 primitives
print("\n=== Test 2: Generate all 11 primitives ===")
presets = [
    ("box 4x2", "box"),
    ("cube 3", "cube"),
    ("sphere r=3.0", "sphere"),
    ("cylinder 2x6", "cylinder"),
    ("cone 3x4", "cone"),
    ("torus major 2 minor 0.5", "torus"),
    ("prism 5x3 tall 2", "prism"),
    ("pentagon_prism 2x4", "pentagon_prism"),
    ("hex_prism 1.5x3", "hex_prism"),
    ("tetrahedron 2", "tetrahedron"),
    ("octahedron 2", "octahedron"),
]

results = {}
for prompt, name in presets:
    r = requests.post(f"{BASE_URL}/api/generate", json={"prompt": prompt, "name": name})
    if r.status_code == 200:
        data = r.json()
        file_path = data.get("filepath", "")
        filename = file_path.split("/")[-1] if file_path else name
        file_size = (OUT / filename).stat().st_size
        results[name] = {
            "status": "OK",
            "filename": filename,
            "size_bytes": file_size,
            "total_verts": data.get("total_verts", 0),
            "total_faces": data.get("total_faces", 0),
            "mesh_count": data.get("mesh_count", 0),
        }
        print(
            f"✓ {name:15} → {filename:20} "
            f"({file_size:>8,} bytes, {data['total_verts']:>5} verts, {data['total_faces']:>5} faces)"
        )
    else:
        results[name] = {"status": "ERROR", "error": r.text[:100]}
        print(f"✗ {name:15} → FAILED: {r.text[:80]}")

# Check STEP files
print("\n=== Test 3: Check STEP files ===")
step_files = list(OUT.glob("*.step"))
print(f"STEP files generated: {len(step_files)}")
for sf in step_files[:3]:
    print(f"  {sf.name}: {sf.stat().st_size} bytes")

# Test 4: Upload fix (test upload with valid data)
print("\n=== Test 4: Upload endpoint ===")
upload_path = Path("/tmp/test_upload.stl")
upload_path.write_bytes(b"solid test\nendsolid test\n")

with upload_path.open("rb") as f:
    r = requests.post(
        f"{BASE_URL}/api/upload",
        files={"file": ("test_upload.stl", f, "application/octet-stream")},
    )
if r.status_code == 200:
    upload_data = r.json()
    print(
        f"✓ Upload OK: {upload_data['filename']} "
        f"({upload_data['size_kb']} KB, {upload_data['size_bytes']} bytes)"
    )
    print(f"  Stats: {upload_data.get('stats', 'N/A')}")
else:
    print(f"✗ Upload FAILED: {r.status_code} - {r.text[:100]}")

# Test 5: List files
print("\n=== Test 5: /api/list ===")
r = requests.get(f"{BASE_URL}/api/list")
files = r.json()
print(f"Total files: {len(files)}")
for f in files[:5]:
    print(f"  {f['filename']:25} {f['size_kb']:>6} KB")

# Test 6: STEP export endpoint
print("\n=== Test 6: /api/export-step ===")
r = requests.post(f"{BASE_URL}/api/export-step", json={"name": "box"})
if r.status_code == 200:
    print(f"✓ STEP export OK: {r.headers.get('Content-Disposition', '')}")
else:
    print(f"✗ STEP export FAILED: {r.status_code} - {r.text[:100]}")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
ok_count = sum(1 for v in results.values() if v["status"] == "OK")
print(f"Primitives tested: {len(presets)}")
print(f"Successful: {ok_count}/{len(presets)}")
if ok_count < len(presets):
    failed = [k for k, v in results.items() if v["status"] != "OK"]
    print(f"Failed: {failed}")
print(f"STEP files generated: {len(step_files)}")
print(f"Total files in output: {len(files)}")
