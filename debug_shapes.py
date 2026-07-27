#!/usr/bin/env python3
"""Debug: test each primitive creation individually."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import GeometryEngine

engine = GeometryEngine()

tests = [
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

for prompt, name in tests:
    mesh = engine.create_primitive(prompt)
    v_count = len(mesh.vertices) if hasattr(mesh, "vertices") else "?"
    f_count = len(mesh.faces) if hasattr(mesh, "faces") else "?"
    print(f"{name:15} -> {v_count:>4} verts, {f_count:>4} faces")
