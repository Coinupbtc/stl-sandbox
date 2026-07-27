"""Deterministic fast path: exact-syntax primitive prompts -> CadQuery code, no LLM.

Matches the old prototype's shorthand ("box 40x20x10", "sphere r=30", "cylinder 20x60",
"hex_prism 25x40", ...). Returns generated CadQuery source (run through the same
cq_runner as LLM output) or None if the prompt is not a simple primitive.
Dimensions are millimeters. Single numbers like "cube 30" mean 30 mm.
"""

import re
from typing import Optional


def _nums(text: str, n: int, defaults: list) -> list:
    found = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]
    return (found + defaults[len(found):])[:n]


def match(prompt: str) -> Optional[str]:
    p = prompt.lower().strip()

    # Only claim prompts that are a shape word plus (optionally) numbers/units —
    # anything with real language goes to the LLM.
    if not re.fullmatch(
        r"(box|cube|sphere|ball|cylinder|cone|torus|donut|tetrahedron|octahedron"
        r"|hex_prism|hex prism|pentagon_prism|pentagon prism|prism)"
        r"[\sxX=r×,.\d mm]*",
        p,
    ):
        return None

    if p.startswith(("box", "prism")) and not p.startswith(("hex", "pentagon")):
        l, w, h = _nums(p, 3, [40.0, 40.0, 40.0])
        if len(re.findall(r"\d", p)) and "x" not in p and w == 40.0:
            w = h = l  # single number = cube-ish box
        return f"import cadquery as cq\nresult = cq.Workplane('XY').box({l}, {w}, {h})\n"

    if p.startswith("cube"):
        (s,) = _nums(p, 1, [40.0])
        return f"import cadquery as cq\nresult = cq.Workplane('XY').box({s}, {s}, {s})\n"

    if p.startswith(("sphere", "ball")):
        (r,) = _nums(p, 1, [25.0])
        return f"import cadquery as cq\nresult = cq.Workplane('XY').sphere({r})\n"

    if p.startswith("cylinder"):
        r, h = _nums(p, 2, [20.0, 60.0])
        return f"import cadquery as cq\nresult = cq.Workplane('XY').circle({r}).extrude({h})\n"

    if p.startswith("cone"):
        r, h = _nums(p, 2, [25.0, 50.0])
        return (
            "import cadquery as cq\n"
            f"result = cq.Solid.makeCone({r}, 0, {h})\n"
        )

    if p.startswith(("torus", "donut")):
        major, minor = _nums(p, 2, [30.0, 10.0])
        return (
            "import cadquery as cq\n"
            f"result = (cq.Workplane('XZ').center({major}, 0).circle({minor})"
            ".revolve(360, (0, 0, 0), (0, 1, 0)))\n"
        )

    if p.startswith("tetrahedron"):
        (s,) = _nums(p, 1, [40.0])
        return (
            "import cadquery as cq\n"
            "import math\n"
            f"s = {s}\n"
            "pts = [(0,0,0), (s,0,0), (s/2, s*math.sqrt(3)/2, 0), (s/2, s*math.sqrt(3)/6, s*math.sqrt(6)/3)]\n"
            "faces = [(0,2,1), (0,1,3), (1,2,3), (2,0,3)]\n"
            "shell = cq.Shell.makeShell([\n"
            "    cq.Face.makeFromWires(cq.Wire.makePolygon(\n"
            "        [cq.Vector(*pts[i]) for i in f] + [cq.Vector(*pts[f[0]])]))\n"
            "    for f in faces\n"
            "])\n"
            "result = cq.Solid.makeSolid(shell)\n"
        )

    if p.startswith("octahedron"):
        (s,) = _nums(p, 1, [40.0])
        return (
            "import cadquery as cq\n"
            f"s = {s} / 2\n"
            "pts = [(s,0,0), (-s,0,0), (0,s,0), (0,-s,0), (0,0,s), (0,0,-s)]\n"
            "faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]\n"
            "shell = cq.Shell.makeShell([\n"
            "    cq.Face.makeFromWires(cq.Wire.makePolygon(\n"
            "        [cq.Vector(*pts[i]) for i in f] + [cq.Vector(*pts[f[0]])]))\n"
            "    for f in faces\n"
            "])\n"
            "result = cq.Solid.makeSolid(shell)\n"
        )

    if p.startswith(("hex_prism", "hex prism")):
        r, h = _nums(p, 2, [25.0, 50.0])
        return (
            "import cadquery as cq\n"
            f"result = cq.Workplane('XY').polygon(6, {r * 2}).extrude({h})\n"
        )

    if p.startswith(("pentagon_prism", "pentagon prism")):
        r, h = _nums(p, 2, [25.0, 50.0])
        return (
            "import cadquery as cq\n"
            f"result = cq.Workplane('XY').polygon(5, {r * 2}).extrude({h})\n"
        )

    return None
