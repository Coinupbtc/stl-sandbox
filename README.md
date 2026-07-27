# STL Sandbox

Type a plain-English prompt → get a **3D-printable STL** (and a **STEP** for CAD) — fully
local, no cloud. Browser sandbox at **http://localhost:8050**.

## At a glance

| | |
|---|---|
| **What it is** | A **local prompt → 3D-printable STL/STEP** sandbox (CadQuery + optional LLM), with bed-fit checks and a print assistant in the browser. |
| **What it’s for** | Turn plain English (or `cube 40`) into printable geometry on your machine — no cloud CAD — and catch bed/overhang issues before you slice. |
| **How to use it** | `./install.sh` (or `./setup.sh`), run `app.py`, open **http://127.0.0.1:8050/**. Primitives work offline; AI gen needs `STL_SANDBOX_LLM_URL`. |

## Try it (pick one)

### One command
```bash
git clone https://github.com/Coinupbtc/stl-sandbox.git
cd stl-sandbox && ./install.sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python app.py
# open http://127.0.0.1:8050/
```

### Copy-paste
```bash
git clone https://github.com/Coinupbtc/stl-sandbox.git && cd stl-sandbox
./install.sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python app.py
```

Primitives (`cube 40`, etc.) work without an LLM. AI generation needs any OpenAI-compatible server — set `STL_SANDBOX_LLM_URL` (default `http://127.0.0.1:8889/v1`).

## How it works

```
prompt ──► fast path: "cube 40", "cylinder 20x60" → CadQuery, instant
       │
       ├─► IMAGINE path (auto for creatures / "anything I imagine")
       │     "a cat", "build me a chicken", "human figure", "dragon toy"…
       │     → local LLM writes a trimesh sculpt (spheres/capsules/cones)
       │     → organic_runner unions parts, scales ~100 mm, adds print base
       │     → STL (stylized toy/figurine — solid & printable, not photoreal)
       │
       └─► mechanical path (phone cases, stands, mugs, nuts, enclosures…)
             1. DIMENSION RESEARCH — catalog / Wikipedia / GSMArena for real
                product body sizes (e.g. iPhone 17 Pro Max = 163.4×78.0×8.75 mm)
             2. SIZE CONTRACT — every prompt gets real mm from (a) numbers in the
                prompt, (b) product research, or (c) object catalog (credit card,
                soda can, AA battery, M-series washers, generic phone, …)
             3. PRODUCT TEMPLATE (no LLM) when shape is known — phone case/stand,
                mug, pen holder, project box, cable clip, wall hook, washer,
                soap dish, can holder, nameplate, hex nut, desk tray, …
             4. else LLM CadQuery with the size contract injected
             5. Quality gates: watertight (auto light-repair), size check, layout
                probes, reject solid-brick cases / oversimplified junk → retry
             → STL + STEP
```

**Auto mode routes for you.** Creatures/characters/sculptures go to Imagine (fast toy);
brackets/gears/plates go to mechanical CAD. For **high-quality** organic models use:

| Mode | What it does | Quality |
|---|---|---|
| `imagine_hq` | **Flux Schnell → Hunyuan3D 2.1 → repaired STL** (needs ComfyUI :8188) | High |
| `imagine` | LLM sphere/capsule toy sculpt | Fast draft |
| `llm` / `auto` mechanical | Dim research → product template or CadQuery | **Realistic product-fit** |
| `fast` | `cube 40` style primitives | Instant |

**Product-fit quality:** prompts like “iPhone 17 Pro Max case” look up real body
dimensions, then build a parametric hollow case (~82×167×12 mm outer for that
phone) instead of an LLM-guessed brick. Meta JSON stores `dimensions` + source.

## ISO-style inspection (engineering prototype control)

Every mechanical generation gets an **inspection certificate** modeled on
ISO 2768-1 dimensional tolerances + AS9102-style characteristic lists:

| Artifact | Content |
|---|---|
| `*.quality.json` | Machine-readable certificate (disposition, dims, integrity) |
| `*.quality.txt` | Human inspection report |
| `/api/inspect` | Re-inspect any library model |
| UI panel | Grade A/B/F · PASS/FAIL characteristics · download cert |

**Tolerance classes:** `iso2768-f|m|c|v` (default **m**) or `fdm-engineering`.
Env: `STL_ISO_CLASS`, `STL_ISO_STRICT=1` (REJECT fails the job),
`STL_CQ_LINEAR_TOL=0.01` (B-rep tessellation).

**Honest boundary:** certificates prove *process-controlled digital geometry*
and dimensional inspection — suitable for engineering prototypes and design
verification. They are **not** AS9100 / flight / NADCAP qualification
(materials, NDT, and QMS are out of scope).
Status badges: `/api/status` reports `imagine_hq_ready`, `hunyuan_up`, `flux_ready`.
Quality notes: `docs/QUALITY_COMPARISON.md`.

## Print Assistant

Every model can be **inspected for real printability**, not just generated. The assistant:

1. Measures STL dimensions and checks them against the printer **build volume** (X/Y/Z)
2. Refuses to call something printable if it exceeds the bed or max height
3. Suggests **rotate / scale % / split** with estimated new sizes when it does not fit
4. Scores **mesh health** (watertight, open edges, thin walls, islands, triangle count)
5. Estimates **printability** (bed contact, overhangs, bridges, supports, warp risk)
6. Runs a **demo print walkthrough** at 2x / 10x / 60x (layer stages, nozzle, risks)
7. Emits a **visual print map** + slicer-style preview + top-3 failure predictions
8. Recommends practical slicer settings (temps, walls, infill, adhesion, cooling)

**Rule:** unknown bed size is treated as missing critical info — analysis is blocked until
you set printer bed X/Y/Z (or pick a known profile like Ender 3 / Prusa / Bambu).

```bash
# CLI-style via API
curl -s -X POST localhost:8050/api/analyze -H 'Content-Type: application/json' \
  -d '{"name":"cube_40","printer_model":"ender3","filament":"PLA","demo_speed":10}'
```

UI: sidebar **Print Assistant** panel — pick printer, filament, use-case, demo speed →
**Analyze**. Auto-runs after generate/upload/library open. Mesh tints green/yellow/red by fit.

Defaults when unspecified: FDM · 0.4 mm nozzle · 0.2 mm layer · PLA · standard profile · demo 10x.

## Files

| File | Role |
|---|---|
| `app.py` | FastAPI server: generate, run-code, library, upload, **analyze** |
| `dim_research.py` | **Product dimensions** — catalog + Wikipedia/GSMArena + fit clearances |
| `size_intent.py` | **Universal size contract** — catalog + parse + bbox gate + quality retry |
| `product_templates.py` | Parametric phone case/stand + household (mug, washer, soap dish, can…) |
| `test_templates.py` | Household template smoke + mug e2e path=template |
| `test_size_intent.py` | Size catalog, routing, washer/soap/can/stand templates |
| `feature_validate.py` | **Layout probes** — camera vs USB opposite ends, face checks; blocks bad STLs |
| `test_feature_layout.py` | Regression: XZ-offset anti-pattern must fail validation |
| `organic_gen.py` | **Imagine path** — freeform creatures/characters → trimesh sculpt via LLM |
| `organic_runner.py` | sandboxed mesh assembler (union, scale, print base, STL export) |
| `print_assistant.py` | bed fit, mesh health, printability, demo walkthrough, slicer recs |
| `llm_codegen.py` | research → template/LLM CadQuery (503 backoff, AST safety, size retry) |
| `cq_runner.py` | isolated CadQuery executor: rlimits, validity, fine STL+STEP export |
| `fast_path.py` | deterministic primitive-syntax → CadQuery templates |
| `www/viewer.html` | three.js viewer + Imagine chips + Print Assistant + dim badge |
| `test_print_assistant.py` | smoke tests for fit / oversize / missing bed |
| `test_dim_research.py` | product dims + phone case template smoke tests |
| `install.sh` | idempotent setup incl. the aarch64 cadquery workaround |

## aarch64 note (DGX Spark / ARM64)

PyPI has no `nlopt` wheel/sdist ≥2.9 for aarch64, which blocks `pip install cadquery`.
cadquery only touches nlopt for 2D sketch constraint solving (never used here), so
`install.sh` installs cadquery `--no-deps` + its other deps and symlinks
`nlopt_shim/nlopt.py` into site-packages to satisfy the import. If a real aarch64 nlopt
appears later, `pip install nlopt` and delete the symlink.

## Run (after clone)

```bash
./install.sh                 # once
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python app.py
```

AI generation needs an OpenAI-compatible server (`STL_SANDBOX_LLM_URL`). Primitives still work when the LLM is down (UI badge: "AI offline — primitives only").

## Env knobs

`SANDBOX_PORT` (8050) · `STL_SANDBOX_LLM_URL` (http://127.0.0.1:8889/v1) ·
`STL_SANDBOX_MAX_ATTEMPTS` (3) · `STL_SANDBOX_EXEC_TIMEOUT` (90s) · `BIND_HOST` (127.0.0.1)

**Imagine HQ quality** (shape/resolution — all optional):

| Env | Default | Effect |
|---|---:|---|
| `STL_HQ_FLUX_SIZE` | 768 | Flux reference image edge (↑ detail, ↑ VRAM) |
| `STL_HQ_FLUX_STEPS` | 4 | Schnell steps (keep 4) |
| `STL_HQ_HY_STEPS` | 40 | Hunyuan sampler steps |
| `STL_HQ_HY_OCTREE` | 384 | Mesh density (256 safe / 512 max detail) |
| `STL_HQ_HY_CHUNKS` | 12000 | Decode chunks |

User prompts are auto-expanded for 3D reconstruction (studio gray bg, ¾ view, matte, full body).
Write a clear **subject + pose + distinctive parts**; skip “chibi blob” wording if you want accurate shape.

## API (print)

| Endpoint | Purpose |
|---|---|
| `GET /api/printers` | printer profiles, filaments, use-cases, demo speeds |
| `POST /api/analyze` | full report for a library STL (`name` + printer context) |

Analyze body fields: `name`, `printer_model`, `bed_x/y/z`, `nozzle_mm`, `layer_height_mm`,
`filament`, `slicer`, `intended_use`, `demo_speed` (2|10|60), `clearance_mm`.
