"""Imagine path: freeform prompts → stylized printable organic figurines.

"a cat", "build me a chicken", "3d human figure" → LLM writes a trimesh sculpt
script (spheres, ellipsoids, capsules, cones) → organic_runner unions + scales
to a printable toy with a flat base.

This is NOT photoreal mesh diffusion. It is a solid, recognizable figurine that
actually 3D-prints. Mechanical parts still use the CadQuery path.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Optional

import requests

LLM_BASE_URL = os.environ.get("STL_SANDBOX_LLM_URL", "http://127.0.0.1:8889/v1")
LLM_TIMEOUT_S = int(os.environ.get("STL_SANDBOX_LLM_TIMEOUT", "240"))
EXEC_TIMEOUT_S = int(os.environ.get("STL_SANDBOX_ORGANIC_TIMEOUT", "120"))
MAX_ATTEMPTS = int(os.environ.get("STL_SANDBOX_MAX_ATTEMPTS", "3"))

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_THIS_DIR, "organic_runner.py")

# Heuristic: freeform creatures / characters / sculptures → imagine path
_ORGANIC_RE = re.compile(
    r"\b("
    r"cat|kitten|dog|puppy|bird|chicken|rooster|hen|duck|owl|eagle|fish|"
    r"horse|cow|pig|sheep|rabbit|bunny|mouse|rat|bear|fox|wolf|deer|lion|"
    r"tiger|elephant|giraffe|monkey|ape|gorilla|dinosaur|dragon|unicorn|"
    r"snake|lizard|frog|turtle|whale|dolphin|shark|octopus|spider|bee|"
    r"human|person|man|woman|boy|girl|child|body|figure|character|hero|"
    r"warrior|knight|robot|android|golem|ghost|angel|demon|fairy|elf|"
    r"dwarf|goblin|orc|zombie|skeleton|skull|head|bust|statue|sculpture|"
    r"figurine|toy|plush|creature|monster|animal|beast|pokemon|anime|"
    r"cartoon|chibi|mage|wizard|ninja|samurai|astronaut|alien|ufo|"
    r"tree|flower|mushroom|heart|star|moon|sun|cloud|castle|house|"
    r"car\b|truck|boat|ship|plane|rocket|spaceship|sword|shield|crown|"
    r"imagine|sculpt|organic"
    r")\b",
    re.I,
)

# Mechanical signals that should stay on CadQuery even if a word overlaps
_MECHANICAL_RE = re.compile(
    r"\b("
    r"bracket|mount|enclosure|housing|gear|pulley|bearing|flange|gasket|"
    r"washer|nut|bolt|screw|spacer|standoff|plate|panel|lid|cover|"
    r"phone\s*stand|pen\s*holder|cable|clip|hook|hinge|latch|knob|"
    r"pipe|tube|adapter|coupler|thread|metric|m\d+|clearance\s*hole|"
    r"extrusion|cad|parametric|tolerance|mm\s*thick|"
    r"case|casing|stand|dock|cradle|holder|tray|dish|soap|"
    r"project\s*box|enclosure|organizer|clamp|jig|fixture|"
    r"mm\b|diameter|across\s+flats|bore|wall(?:s)?\b"
    r")\b",
    re.I,
)

# Explicit toy/art intent — only then organic wins over mechanical words
_TOY_ART_RE = re.compile(
    r"\b(figurine|toy|plush|sculpture|statue|bust|chibi|cartoon|anime|"
    r"imagine|sculpt|organic|miniature\s+figure)\b",
    re.I,
)


SYSTEM_PROMPT = """You are an expert at designing STYLIZED 3D-printable toy figurines.
You convert a plain-English idea (cat, chicken, human, dragon, anything imaginable)
into a short Python script that builds the figure from solid primitives.

You do NOT make photoreal meshes. You make cute, solid, LOW-POLY TOY sculptures
that are recognizable and 3D-printable.

## API available in your script (already imported — do NOT import anything)

- sphere(radius, center=(x,y,z), subdivisions=2) -> mesh
- ellipsoid(radii=(rx,ry,rz), center=(x,y,z), subdivisions=2) -> mesh
- capsule(p0=(x,y,z), p1=(x,y,z), radius=r) -> mesh   # bone / limb
- cone(radius, height, center=(x,y,z)) -> mesh
- box(extents=(x,y,z), center=(x,y,z)) -> mesh
- cylinder(radius, height, center=(x,y,z)) -> mesh   # height along Z, centered
- union(list_of_meshes) -> mesh   # optional; runner also unions `parts`

Coordinates: millimeters in a character-local frame.
- +Z is UP
- +Y is FORWARD (nose / face direction)
- +X is character's LEFT
- Origin near the belly / pelvis

## Rules — follow ALL
1. Output ONLY a Python code block. No prose.
2. Build a list named `parts = [ ... ]` of meshes. Do NOT assign `result` unless needed.
3. Optionally set `target_height_mm = 100` (default 100) and `add_base = True`.
4. NO imports. NO file I/O. NO loops over mesh vertices. Keep under ~60 lines.
5. Use 6–25 solid parts. Overlap parts so unions fuse (no floating islands).
6. Minimum feature radius ≥ 2.5 mm (ears, tails, fingers as chunky blobs).
7. Standing pose preferred. Feet near z=0. Body above feet.
8. Make it RECOGNIZABLE: correct proportions for the subject (cat has head+ears+tail+4 legs;
   chicken has body+small head+beak+2 legs+wings; human has torso+head+2 arms+2 legs).
9. Stylized chibi / toy look is good. Exaggerate head slightly for characters.
10. Do not use strings, dicts of meshes, or helper function definitions — straight-line code only.

## Example — "a cute cat figurine"
```python
target_height_mm = 90
add_base = True
parts = []
# body
parts.append(ellipsoid((18, 28, 16), center=(0, 0, 22)))
# head
parts.append(sphere(14, center=(0, 18, 40)))
# ears
parts.append(cone(6, 12, center=(-8, 16, 52)))
parts.append(cone(6, 12, center=(8, 16, 52)))
# snout
parts.append(sphere(5, center=(0, 28, 38)))
# legs
parts.append(capsule((10, 10, 4), (10, 12, 18), 4.5))
parts.append(capsule((-10, 10, 4), (-10, 12, 18), 4.5))
parts.append(capsule((10, -12, 4), (10, -10, 18), 4.5))
parts.append(capsule((-10, -12, 4), (-10, -10, 18), 4.5))
# tail
parts.append(capsule((0, -26, 20), (8, -40, 32), 3.5))
# eyes (raised bumps)
parts.append(sphere(2.5, center=(-5, 28, 43)))
parts.append(sphere(2.5, center=(5, 28, 43)))
```

## Example — "a simple chicken"
```python
target_height_mm = 85
add_base = True
parts = []
parts.append(ellipsoid((16, 20, 14), center=(0, 0, 28)))  # body
parts.append(sphere(10, center=(0, 16, 40)))              # head
parts.append(cone(4, 10, center=(0, 26, 38)))             # beak (will point +Z; ok as stub)
parts.append(box((3, 8, 6), center=(0, 22, 38)))          # beak block
parts.append(capsule((6, 2, 6), (8, 4, 22), 3.2))         # legs
parts.append(capsule((-6, 2, 6), (-8, 4, 22), 3.2))
parts.append(ellipsoid((10, 6, 8), center=(12, 0, 30)))   # wing
parts.append(ellipsoid((10, 6, 8), center=(-12, 0, 30)))
parts.append(cone(5, 8, center=(0, 12, 50)))              # comb
parts.append(capsule((0, -18, 26), (0, -30, 22), 3))      # tail feathers blob
```

## Example — "a stylized human figure"
```python
target_height_mm = 120
add_base = True
parts = []
parts.append(ellipsoid((12, 8, 20), center=(0, 0, 55)))   # torso
parts.append(sphere(11, center=(0, 0, 80)))               # head
parts.append(capsule((0, 0, 35), (0, 0, 55), 7))          # hips/waist
parts.append(capsule((8, 0, 60), (18, 0, 40), 4))         # right arm
parts.append(capsule((-8, 0, 60), (-18, 0, 40), 4))        # left arm
parts.append(capsule((5, 0, 35), (6, 0, 8), 4.5))         # right leg
parts.append(capsule((-5, 0, 35), (-6, 0, 8), 4.5))        # left leg
parts.append(box((8, 12, 3), center=(6, 4, 4)))           # right foot
parts.append(box((8, 12, 3), center=(-6, 4, 4)))          # left foot
```

Now build the figurine the user asks for. Be creative but keep it printable and solid.
"""

RETRY_PROMPT = """The sculpt script failed. Fix it and output the FULL corrected script
(only a Python code block). Keep using sphere/ellipsoid/capsule/cone/box/cylinder only.
If a part is wrong, simplify it.

Failed script:
```python
{code}
```

Error:
```
{error}
```
"""

_ALLOWED_CALLS = {
    "sphere", "ellipsoid", "capsule", "cone", "box", "cylinder", "union",
    "append", "extend", "range", "len", "float", "int", "abs", "min", "max",
    "round", "list", "tuple", "enumerate", "zip", "sum", "print",
}
_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "getattr", "setattr", "delattr", "vars", "memoryview",
    "os", "sys", "subprocess", "pathlib", "requests", "socket",
}


def is_organic_prompt(prompt: str) -> bool:
    """True when the user wants a creature/character/sculpture, not a mechanical part.

    Functional / product-fit language always wins over creature keywords unless
    the user also asked for a figurine/toy/sculpture (e.g. "cat phone stand" → CAD,
    "cat figurine" → imagine).
    """
    p = prompt.strip()
    if not p:
        return False
    mech = bool(_MECHANICAL_RE.search(p))
    org = bool(_ORGANIC_RE.search(p))
    toy = bool(_TOY_ART_RE.search(p))

    if mech and not toy:
        return False
    if org:
        return True
    # short freeform "build me X" / "make a X" without mechanical words → imagine
    if re.search(r"\b(build|make|create|sculpt|imagine|design)\b", p, re.I):
        if not mech:
            if re.search(r"\b(box|cube|cylinder|bracket|plate|gear|enclosure)\b", p, re.I):
                return False
            return True
    return False


def _check_code_safety(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError before execution: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed in organic scripts"
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _FORBIDDEN_NAMES:
            return f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"forbidden dunder: {node.attr}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return "function/class definitions not allowed — use straight-line code"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_CALLS:
                return f"forbidden call: {node.func.id}()"
        # method calls: only allow benign ones on lists/meshes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in ("append", "extend", "apply_translation", "apply_scale", "apply_transform"):
                return f"forbidden method: .{node.func.attr}()"
    return None


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip()


def _chat(messages: list, temperature: float = 0.4, think: bool = False) -> str:
    last_err = None
    for backoff in (0, 5, 15, 30):
        if backoff:
            time.sleep(backoff)
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 8192 if think else 4096,
                "chat_template_kwargs": {"enable_thinking": think},
            },
            timeout=LLM_TIMEOUT_S,
        )
        if resp.status_code == 503:
            last_err = requests.HTTPError("LLM busy (503)", response=resp)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise last_err


def run_organic_code(code: str, stl_path: str) -> dict:
    bad = _check_code_safety(code)
    if bad:
        return {"ok": False, "error": f"Rejected before execution: {bad}"}

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        code_path = f.name
    try:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        proc = subprocess.run(
            [sys.executable, _RUNNER, code_path, stl_path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_S,
            cwd=tempfile.gettempdir(),
            env=env,
        )
        out = proc.stdout.strip().splitlines()
        for line in reversed(out):
            if line.startswith("{"):
                return json.loads(line)
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout or "organic runner produced no result")[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Organic sculpt timed out after {EXEC_TIMEOUT_S}s"}
    finally:
        try:
            os.unlink(code_path)
        except OSError:
            pass


def generate_organic(
    prompt: str,
    stl_path: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """prompt → LLM trimesh sculpt → STL. Returns result dict compatible with app.py."""
    t0 = time.time()
    notify = on_progress or (lambda a, p: None)
    user_prompt = (
        f"Design a stylized 3D-printable toy figurine for:\n\n{prompt}\n\n"
        "Make it cute, solid, and recognizable. Standing pose with chunky features."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    attempts = []
    last_code = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        think = attempt == MAX_ATTEMPTS
        notify(attempt, "llm")
        try:
            content = _chat(messages, temperature=0.45 if attempt == 1 else 0.3, think=think)
            code = _extract_code(content)
            last_code = code
        except Exception as e:
            attempts.append({"n": attempt, "ok": False, "error": f"LLM error: {e}"})
            continue

        notify(attempt, "sculpt")
        run = run_organic_code(code, stl_path)
        attempts.append({
            "n": attempt,
            "ok": run.get("ok", False),
            "error": run.get("error"),
        })
        if run.get("ok"):
            return {
                "ok": True,
                "error": None,
                "code": code,
                "stats": run.get("stats", {}),
                "warnings": (run.get("warnings") or []) + [
                    "Imagine path: stylized toy/figurine (not photoreal mesh). Great for printing."
                ],
                "attempts": attempts,
                "path_used": "imagine",
                "elapsed_s": round(time.time() - t0, 1),
            }

        # retry with error feedback
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({
            "role": "user",
            "content": RETRY_PROMPT.format(code=code, error=run.get("error") or "unknown"),
        })

    return {
        "ok": False,
        "error": (attempts[-1].get("error") if attempts else "no attempts") or "organic generation failed",
        "code": last_code,
        "stats": {},
        "warnings": [],
        "attempts": attempts,
        "path_used": "imagine",
        "elapsed_s": round(time.time() - t0, 1),
    }
