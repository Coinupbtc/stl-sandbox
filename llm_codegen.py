"""LLM -> CadQuery code generation with sandboxed execution and self-correcting retries.

Pipeline (the pattern from CADSmith / Text-to-CadQuery), upgraded for realism:
  prompt
    -> dimension research (catalog / Wikipedia / GSMArena) when product-fit
    -> parametric template when available (phone case/stand, mug, pen holder,
       project box, cable clip, wall hook, nameplate, hex nut, desk tray)
    -> else LLM writes CadQuery with researched dimensions injected
    -> run in isolated subprocess
    -> on exception OR wrong size vs researched envelope, retry with feedback
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

import dim_research
import feature_validate
import iso_quality
import product_templates
import size_intent

# When true (default), REJECT disposition fails the generation instead of soft-pass
ISO_STRICT = os.environ.get("STL_ISO_STRICT", "1") not in ("0", "false", "False")
ISO_CLASS = os.environ.get("STL_ISO_CLASS", iso_quality.DEFAULT_CLASS)

LLM_BASE_URL = os.environ.get("STL_SANDBOX_LLM_URL", "http://127.0.0.1:8889/v1")
LLM_TIMEOUT_S = int(os.environ.get("STL_SANDBOX_LLM_TIMEOUT", "240"))
EXEC_TIMEOUT_S = int(os.environ.get("STL_SANDBOX_EXEC_TIMEOUT", "90"))
MAX_ATTEMPTS = int(os.environ.get("STL_SANDBOX_MAX_ATTEMPTS", "3"))

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_THIS_DIR, "cq_runner.py")


SYSTEM_PROMPT = """You are an expert CadQuery (version 2.x) programmer and mechanical designer.
You convert a plain-English description of a physical object into a short CadQuery Python
script that builds a single, watertight, 3D-printable solid with REALISTIC dimensions.

Rules — follow ALL of them:
1. Output ONLY a Python code block. No explanations before or after.
2. `import cadquery as cq` and optionally `import math`. No other imports.
3. Assign the finished solid to a variable named `result` (a cq.Workplane or cq.Shape).
4. Units are ALWAYS millimeters.
5. DIMENSIONS ARE SACRED:
   - If the user message includes a "RESEARCHED REAL-WORLD DIMENSIONS" block, you MUST
     use those numbers. Do not invent alternate phone/product sizes.
   - If the user gives sizes, use them exactly.
   - If no sizes are given and no research block is present, pick sensible printable
     sizes (30–120 mm) and put them in named constants at the top of the script.
6. The part must be ONE connected, watertight solid — union separate pieces together.
7. Never read/write files, never use os/sys/subprocess/open/exec/eval, no network.
8. Prefer simple robust operations: box, cylinder, extrude, revolve, cut, union, shell,
   fillet, chamfer, hole, cboreHole, cskHole, rect, circle, polygon, polarArray, rarray.
   Signatures that are easy to get wrong — read carefully:
   - polygon(nSides, diameter) — the second argument is the DIAMETER of the circumscribed
     circle (vertex to vertex), NOT a radius.
   - hole(diameter) / cboreHole / cskHole take DIAMETERS, not radii.
   - circle(radius) takes a RADIUS.
   - shell(thickness) with a NEGATIVE value shells inward, keeping outer dimensions.
   - box(length, width, height) is centered on the origin by default.
9. Fillet/chamfer radii must be clearly smaller than the faces they touch, or omit them.
   Fillets and chamfers are OPTIONAL decoration — when in doubt, leave them out entirely.
10. Keep it under ~60 lines. Straight-line code only: no functions, no try/except, no
    conditionals that inspect the model, no loops over selection results.
11. PRODUCT-FIT / CASES / ENCLOSURES (critical quality rules):
    - A "case" or "cover" for a device is a HOLLOW SHELL, not a solid brick or a 0.8 mm plate.
    - Wall thickness for FDM: 1.4–1.8 mm (never under 1.0 mm).
    - Per-side clearance for a phone/device insert: 0.3–0.5 mm.
    - Open the screen/front face; leave a thin lip (~0.8–1.2 mm) to retain the device.
    - Add the openings the real object needs (charging port, camera island, buttons).
    - Outer size ≈ device + 2×clearance + 2×wall on width/height; depth ≈ wall + device + lip.
    - Rounded corners should track the device corner radius when provided.
12. PHONE CASE ORIENTATION (logic errors here are unacceptable):
    Use world axes consistently:
      X = width (− left, + right), Y = height (− BOTTOM, + TOP), Z = thickness (0 BACK, + SCREEN).
    - Camera island cutout: through the BACK wall, at the TOP end (+Y), usually top-left.
    - USB-C / charging cutout: on the BOTTOM edge (−Y), mid-thickness — NEVER on the back face
      and NEVER on the same end as the camera.
    - Volume buttons: LEFT edge (−X); power: RIGHT edge (+X); both upper half.
    - Prefer axis-aligned boxes placed with .translate((x,y,z)) over Workplane("XZ"/"YZ")
      offsets (named-plane normals invert easily and put USB on the camera end).
13. Never produce degenerate geometry: no paper-thin solids, no 2D sheets, no parts whose
    thinnest axis is under 1.0 mm unless the user explicitly asked for a gasket/shim.

Example — "a name tag base 60 by 25 mm with rounded corners":
```python
import cadquery as cq
result = (
    cq.Workplane("XY")
    .box(60, 25, 3)
    .edges("|Z")
    .fillet(4)
)
```

Example — "a plate 80x50x5 with four M4 clearance holes near the corners":
```python
import cadquery as cq
result = (
    cq.Workplane("XY")
    .box(80, 50, 5)
    .faces(">Z")
    .workplane()
    .rect(66, 36, forConstruction=True)
    .vertices()
    .hole(4.4)
)
```

Example — "a simple coffee mug, 80 mm tall":
```python
import cadquery as cq
body = (
    cq.Workplane("XY")
    .circle(40)
    .extrude(80)
    .faces(">Z")
    .shell(-3)
)
handle = (
    cq.Workplane("XZ")
    .center(44, 40)
    .circle(14)
    .extrude(6, both=True)
    .cut(
        cq.Workplane("XZ").center(44, 40).circle(8).extrude(8, both=True)
    )
)
result = body.union(handle)
```

Example — "a hex nut, 20 mm across flats, 10 mm tall, 10 mm bore":
```python
import cadquery as cq
import math
flat = 20.0
r_hex = flat / math.cos(math.pi / 6) / 2
result = (
    cq.Workplane("XY")
    .polygon(6, r_hex * 2)
    .extrude(10)
    .faces(">Z")
    .hole(10)
)
```

Example — phone case shell using researched body dims (world-space cutters only):
```python
import cadquery as cq
# Example phone body: 163.4 x 78.0 x 8.75 mm — REPLACE with researched numbers
# Axes: X=width, Y=height (−BOTTOM/USB, +TOP/camera), Z=depth (0=BACK, +=SCREEN)
PH, PW, PD = 163.4, 78.0, 8.75
WALL, CLR, LIP = 1.6, 0.4, 1.2
CAV_H, CAV_W, CAV_D = PH + 2*CLR, PW + 2*CLR, PD + CLR
OUT_H, OUT_W, OUT_D = CAV_H + 2*WALL, CAV_W + 2*WALL, WALL + CAV_D + LIP
OUTER_R, INNER_R = 12.0, 10.5
outer = cq.Workplane("XY").rect(OUT_W, OUT_H).extrude(OUT_D).edges("|Z").fillet(OUTER_R)
cavity = (
    cq.Workplane("XY").workplane(offset=WALL).rect(CAV_W, CAV_H)
    .extrude(CAV_D + LIP + 0.5).edges("|Z").fillet(INNER_R)
)
shell = outer.cut(cavity)
screen = (
    cq.Workplane("XY").workplane(offset=WALL + CAV_D - 0.1)
    .rect(CAV_W - 1.8, CAV_H - 1.8).extrude(LIP + 1.0)
)
shell = shell.cut(screen)
# camera: BACK + TOP — world box + translate (never XZ/YZ offsets for ports)
cam = (
    cq.Workplane("XY").box(40, 40, WALL + 0.8).edges("|Z").fillet(3.5)
    .translate((-OUT_W/2 + 24, OUT_H/2 - 24, WALL/2))
)
shell = shell.cut(cam)
# USB-C: BOTTOM edge (−Y), opposite end from camera
usb = (
    cq.Workplane("XY").box(12.5, WALL + 3.0, 7.0)
    .translate((0.0, -OUT_H/2 + (WALL + 3.0)/2, WALL + CAV_D*0.45))
)
result = shell.cut(usb)
```
"""

RETRY_PROMPT = """The script you wrote failed. Fix it and output the FULL corrected script \
(only a Python code block, same rules as before). If the failing operation is cosmetic \
(a fillet or chamfer), simply REMOVE it rather than trying to fix it.

Failed script:
```python
{code}
```

Error:
```
{error}
```
"""

DIM_RETRY_PROMPT = """The script ran but the solid is the WRONG SIZE compared to researched \
real-world dimensions. Fix the script so the bounding box matches the expected envelope \
(within ~15%). Output the FULL corrected script only.

Expected outer envelope (W×H×D mm, order may differ): {expected}
Actual bbox_mm from export: {actual}
Problem: {problem}

Researched dimension brief (must obey):
{brief}

Failed script:
```python
{code}
```
"""

LAYOUT_RETRY_PROMPT = """The script built a solid, but FEATURE LAYOUT VALIDATION failed. \
These are logic errors (wrong face / wrong end), not cosmetic ones. Fix them.

Phone case axis convention (mandatory):
  X = width (− left, + right)
  Y = height (− BOTTOM / USB-C, + TOP / camera)
  Z = thickness (0 = BACK plate, + = SCREEN)
  Camera = BACK + TOP.  USB-C = BOTTOM edge.  They must be on OPPOSITE ends of Y.

Validation errors:
{errors}

Failed script:
```python
{code}
```

Output the FULL corrected script only (Python code block). Prefer .box(...).translate((x,y,z))
for cutouts so world positions are unambiguous.
"""


def _layout_check(code: str, dims: dict | None) -> dict | None:
    """Return validation dict if a layout applies; None if not applicable."""
    if not dims or not feature_validate.layout_for_dims(dims):
        return None
    try:
        return feature_validate.validate_result_solid(code, dims)
    except Exception as e:
        return {"ok": False, "errors": [f"layout validation crashed: {e}"], "checks": []}

# Best-effort guard: this runs locally for a single user, but the code still comes
# from an LLM — reject the obviously dangerous constructs before executing anything.
# AST-based so comments/strings can't false-positive (regex rejected "# across flats").
_ALLOWED_IMPORTS = {"cadquery", "math"}
_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "getattr", "setattr", "delattr", "vars", "memoryview",
}


def _check_code_safety(code: str):
    """Returns an error string, or None if the code looks safe to execute."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError before execution: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return f"forbidden import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                return f"forbidden import: from {node.module}"
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return f"forbidden builtin: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"forbidden dunder attribute: {node.attr}"
    return None


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip()


def _chat(messages: list, temperature: float = 0.2, think: bool = False) -> str:
    # think=False keeps generations fast (~10s); the final retry turns thinking on
    # so the model gets one slow, careful attempt before giving up.
    # The llama.cpp server is shared with other local agents and returns 503
    # while its slots are busy — wait it out with backoff instead of failing.
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
            last_err = requests.HTTPError("LLM busy (503) — shared with agent stack", response=resp)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise last_err


def llm_available() -> bool:
    try:
        r = requests.get(f"{LLM_BASE_URL}/models", timeout=4)
        return r.status_code == 200
    except requests.RequestException:
        return False


def run_cadquery_code(code: str, stl_path: str, step_path: str) -> dict:
    """Execute a CadQuery script in an isolated subprocess. Returns the runner's JSON."""
    bad = _check_code_safety(code)
    if bad:
        return {"ok": False, "error": f"Rejected before execution: {bad}"}

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        code_path = f.name
    try:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        proc = subprocess.run(
            [sys.executable, _RUNNER, code_path, stl_path, step_path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_S,
            cwd=tempfile.gettempdir(),
            env=env,
        )
        out = proc.stdout.strip().splitlines()
        # Runner prints exactly one JSON line last; anything before it is script noise.
        for line in reversed(out):
            if line.startswith("{"):
                return json.loads(line)
        return {
            "ok": False,
            "error": (proc.stderr or "runner produced no result").strip()[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Execution timed out after {EXEC_TIMEOUT_S}s"}
    finally:
        os.unlink(code_path)


def _build_user_message(prompt: str, dims: dict | None, size: dict | None) -> str:
    parts = [prompt, ""]
    if dims and dims.get("part_type") in ("phone_case", "holder"):
        parts.append(dim_research.format_dimension_brief(dims))
        parts.append("")
    if size:
        parts.append(size_intent.format_size_brief(size))
    else:
        parts.append(
            "No size contract available — use REALISTIC real-world millimeters "
            "(household objects 30–200 mm). Put sizes in named constants. "
            "Never invent fantasy phone dimensions; prefer published sizes."
        )
    return "\n".join(parts)


def _unlink_outputs(stl_path: str, step_path: str):
    for p in (stl_path, step_path):
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass


def _attach_iso_certificate(
    *,
    prompt: str,
    stl_path: str,
    step_path: str,
    code: str,
    run: dict,
    dims: dict | None,
    size: dict | None,
    layout,
    path_detail: str,
    warnings: list,
    attempts: list,
    t0: float,
    research_err=None,
) -> dict:
    """Build inspection certificate; hard-fail on REJECT when STL_ISO_STRICT=1."""
    contract = dict(size or {})
    if dims:
        for k, v in dims.items():
            if k not in contract and v is not None:
                contract[k] = v
    stats = run.get("stats") or {}
    cert = iso_quality.build_certificate(
        name=os.path.splitext(os.path.basename(stl_path))[0],
        prompt=prompt,
        stl_path=stl_path,
        step_path=step_path if os.path.exists(step_path) else None,
        size_contract=contract or None,
        stats=stats,
        path_used=path_detail,
        code=code,
        layout_validation=layout,
        tolerance_class=ISO_CLASS,
    )
    # Persist certificate beside the part
    cert_path = stl_path.replace(".stl", ".quality.json")
    try:
        with open(cert_path, "w") as f:
            json.dump(cert, f, indent=1)
        txt_path = stl_path.replace(".stl", ".quality.txt")
        with open(txt_path, "w") as f:
            f.write(iso_quality.certificate_text(cert))
    except OSError:
        pass

    reject = iso_quality.should_reject_generation(cert, strict=ISO_STRICT)
    if reject and ISO_STRICT:
        _unlink_outputs(stl_path, step_path)
        return {
            "ok": False,
            "error": reject,
            "code": code,
            "stats": stats,
            "warnings": warnings + [f"ISO disposition: {cert.get('disposition')}"],
            "attempts": attempts,
            "elapsed_s": round(time.time() - t0, 1),
            "dimensions": dims,
            "size_intent": size,
            "path_detail": path_detail,
            "inspection": cert,
            "research_error": research_err,
        }

    warnings = list(warnings)
    warnings.append(
        f"ISO inspection: {cert.get('disposition')} grade {cert.get('grade')} "
        f"({cert.get('tolerance_class_name')})"
    )
    if cert.get("disposition") == "ACCEPT_WITH_DEVIATION":
        warnings.append("ACCEPT_WITH_DEVIATION — review dimensional notes before critical fit")

    return {
        "ok": True,
        "code": code,
        "stats": stats,
        "warnings": warnings,
        "attempts": attempts,
        "elapsed_s": round(time.time() - t0, 1),
        "dimensions": dims,
        "size_intent": size,
        "path_detail": path_detail,
        "dimension_brief": (
            size_intent.format_size_brief(size) if size
            else (dim_research.format_dimension_brief(dims) if dims else None)
        ),
        "layout_validation": layout,
        "inspection": cert,
        "files_extra": {
            "quality_json": os.path.basename(cert_path) if os.path.exists(cert_path) else None,
        },
    }


def _accept_or_reason(
    prompt: str,
    code: str,
    run: dict,
    dims: dict | None,
    size: dict | None,
    *,
    from_template: bool = False,
) -> tuple[bool, str | None, list, dict | None]:
    """Return (ok, error_reason, warnings, layout)."""
    if not run.get("ok"):
        return False, run.get("error", "exec failed"), [], None
    stats = run.get("stats") or {}
    warnings = list(run.get("warnings") or [])
    layout = _layout_check(code, dims)
    if layout is not None and not layout.get("ok"):
        return False, "layout: " + "; ".join(layout.get("errors") or ["failed"]), warnings, layout

    q = size_intent.quality_should_retry(stats, warnings, prompt)
    if q:
        # Parametric templates are trusted for topology; only hard-fail watertight/thin
        if from_template and not (
            "watertight" in q or "too thin" in q or "brick" in q
        ):
            warnings.append(f"quality-note: {q}")
        else:
            return False, f"quality: {q}", warnings, layout

    # Size contract (prefer size_intent; fall back to product dims)
    contract = size or dims
    if contract:
        ok_dim, dim_msg = size_intent.dims_match_bbox(
            contract, stats.get("bbox_mm") or [],
            tol_frac=0.28 if from_template else 0.20,
        )
        # Product-fit cases still use the stricter research checker too
        if dims and dims.get("expected_bbox_mm") and dims.get("part_type") == "phone_case":
            ok2, msg2 = dim_research.dims_match_bbox(dims, stats.get("bbox_mm") or [])
            if not ok2:
                ok_dim, dim_msg = ok2, msg2
        if not ok_dim:
            # Phone cases must match researched envelope — never soft-accept
            if from_template and not (
                dims and dims.get("part_type") == "phone_case"
            ):
                warnings.append(f"size-note: {dim_msg} (template geometry accepted)")
            else:
                return False, f"wrong size: {dim_msg}", warnings, layout
        else:
            warnings.append(f"size-check: {dim_msg}")
    if layout is not None:
        warnings.append("layout-check: feature probes passed")
    return True, None, warnings, layout


QUALITY_RETRY_PROMPT = """The script ran but the solid failed QUALITY checks for a printable part.
Fix the FULL script (Python code block only).

Quality problem: {problem}
Actual bbox_mm: {actual}
Stats: faces={faces}, watertight={wt}

Rules when fixing:
- Must be ONE watertight solid (union pieces).
- Must match the SIZE CONTRACT if present (real mm, not a tiny toy unless asked).
- Cases/enclosures must be HOLLOW shells with openings — never a solid brick.
- Avoid degenerate thin sheets and oversimplified 12-face bricks for complex objects.
- Prefer simple robust CadQuery ops; drop failing fillets.

Failed script:
```python
{code}
```
"""


def generate_model(prompt: str, stl_path: str, step_path: str, on_progress=None) -> dict:
    """Full loop: research -> size contract -> template or LLM -> quality/size gates.

    on_progress(attempt:int, phase:str) is called at each stage
    ("research", "template", "llm", "exec").
    """
    t0 = time.time()
    notify = on_progress or (lambda a, p: None)
    attempts = []
    code = ""
    dims = None
    size = None
    path_detail = "llm"

    # ── Dimension research + universal size contract ──
    notify(0, "research")
    research_err = None
    try:
        dims = dim_research.research_dimensions(prompt)
    except Exception as e:
        dims = None
        research_err = str(e)
    try:
        size = size_intent.resolve_size_intent(prompt, researched=dims)
    except Exception as e:
        size = None
        research_err = (research_err or "") + f"; size_intent: {e}"

    # ── Parametric templates (phone cases, mugs, washers, stands…) ──
    tmpl = product_templates.match_template(prompt, dims)
    if tmpl:
        notify(1, "template")
        run = run_cadquery_code(tmpl, stl_path, step_path)
        ok, err, warnings, layout = _accept_or_reason(
            prompt, tmpl, run, dims, size, from_template=True
        )
        attempts.append({
            "n": 1,
            "ok": ok,
            "error": None if ok else (err or "?")[:800],
            "path": "template",
        })
        if ok:
            if layout is None and "path:" not in " ".join(warnings):
                warnings.append("path: parametric template (no LLM)")
            return _attach_iso_certificate(
                prompt=prompt,
                stl_path=stl_path,
                step_path=step_path,
                code=tmpl,
                run=run,
                dims=dims,
                size=size,
                layout=layout,
                path_detail="template",
                warnings=warnings,
                attempts=attempts,
                t0=t0,
                research_err=research_err,
            )
        code = tmpl
        _unlink_outputs(stl_path, step_path)

    # ── LLM path with size contract ──
    user_msg = _build_user_message(prompt, dims, size)
    if code and attempts and not attempts[-1]["ok"]:
        err = attempts[-1].get("error") or ""
        user_msg += (
            f"\n\nA parametric template failed with:\n{err}\n"
            "Write a correct CadQuery script from scratch obeying the size contract "
            "and (if a phone case) camera BACK+TOP / USB BOTTOM opposite ends."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    dim_retries_used = 0
    quality_retries_used = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        notify(attempt, "llm")
        try:
            raw = _chat(messages, think=(attempt == MAX_ATTEMPTS))
        except requests.RequestException as e:
            return {
                "ok": False,
                "error": f"LLM unreachable: {e}",
                "attempts": attempts,
                "elapsed_s": round(time.time() - t0, 1),
                "dimensions": dims,
                "size_intent": size,
                "research_error": research_err,
            }
        code = _extract_code(raw)
        notify(attempt, "exec")
        run = run_cadquery_code(code, stl_path, step_path)
        ok, err, warnings, layout = _accept_or_reason(prompt, code, run, dims, size)
        attempts.append({
            "n": attempt,
            "ok": ok,
            "error": None if ok else (err or "?")[:800],
            "path": "llm",
        })
        if ok:
            return _attach_iso_certificate(
                prompt=prompt,
                stl_path=stl_path,
                step_path=step_path,
                code=code,
                run=run,
                dims=dims,
                size=size,
                layout=layout,
                path_detail=path_detail,
                warnings=warnings,
                attempts=attempts,
                t0=t0,
                research_err=research_err,
            )

        _unlink_outputs(stl_path, step_path)
        err_s = err or "unknown"
        # Targeted retries
        if err_s.startswith("layout:") and attempt < MAX_ATTEMPTS:
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({
                "role": "user",
                "content": LAYOUT_RETRY_PROMPT.format(
                    errors=err_s[:1500],
                    code=code,
                ),
            })
            continue
        if err_s.startswith("wrong size:") and dim_retries_used < 1 and attempt < MAX_ATTEMPTS:
            dim_retries_used += 1
            brief = (
                size_intent.format_size_brief(size) if size
                else (dim_research.format_dimension_brief(dims) if dims else "")
            )
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({
                "role": "user",
                "content": DIM_RETRY_PROMPT.format(
                    expected=(size or dims or {}).get("expected_bbox_mm"),
                    actual=(run.get("stats") or {}).get("bbox_mm"),
                    problem=err_s,
                    brief=brief,
                    code=code,
                ),
            })
            continue
        if (
            (err_s.startswith("quality:") or not run.get("ok"))
            and quality_retries_used < 2
            and attempt < MAX_ATTEMPTS
        ):
            quality_retries_used += 1
            st = run.get("stats") or {}
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            if run.get("ok"):
                messages.append({
                    "role": "user",
                    "content": QUALITY_RETRY_PROMPT.format(
                        problem=err_s,
                        actual=st.get("bbox_mm"),
                        faces=st.get("faces"),
                        wt=st.get("watertight"),
                        code=code,
                    ),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": RETRY_PROMPT.format(
                        code=code, error=err_s[:1500]
                    ),
                })
            continue
        if attempt < MAX_ATTEMPTS:
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({
                "role": "user",
                "content": RETRY_PROMPT.format(code=code, error=err_s[:1500]),
            })
    return {
        "ok": False,
        "error": attempts[-1]["error"] if attempts else "no attempts ran",
        "code": code,
        "attempts": attempts,
        "elapsed_s": round(time.time() - t0, 1),
        "dimensions": dims,
        "size_intent": size,
        "research_error": research_err,
    }
