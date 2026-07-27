"""STL Sandbox — prompt to 3D-printable STL/STEP, fully local.

Architecture:
  prompt -> fast path (exact primitive syntax -> templated CadQuery code, instant)
         -> product-fit path (dimension research + parametric case/stand templates)
         -> LLM path  (local llama.cpp writes CadQuery with researched dims, retry x3)
  Both CAD paths run through cq_runner.py, so every model is a real B-rep solid with
  native STL + STEP export, watertight by construction.

Generation runs as background jobs; the UI polls /api/job/{id} for live progress.
"""

import glob
import json
import os
import re
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import fast_path
import hq_comfy
import llm_codegen
import organic_gen
import print_assistant

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
WWW_DIR = os.path.join(THIS_DIR, "www")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="STL Sandbox", description="Type a prompt, get a printable STL + STEP.")
app.mount("/www", StaticFiles(directory=WWW_DIR), name="www")


class GenerateRequest(BaseModel):
    prompt: str
    name: Optional[str] = None
    mode: str = "auto"  # auto | fast | llm | imagine | imagine_hq
    # optional reference image filename already in library/output or absolute under output/
    image: Optional[str] = None


class RunCodeRequest(BaseModel):
    code: str
    name: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Analyze an existing model in the library (or absolute path for local use)."""
    name: Optional[str] = None
    path: Optional[str] = None
    # printer context — any missing bed size is treated as critical missing info
    printer_model: str = "ender3"
    bed_x: Optional[float] = None
    bed_y: Optional[float] = None
    bed_z: Optional[float] = None
    nozzle_mm: float = 0.4
    layer_height_mm: float = 0.2
    filament: str = "PLA"
    slicer: str = "standard"
    intended_use: str = "prototype"
    demo_speed: int = 10  # 2 | 10 | 60
    clearance_mm: float = 5.0
    support_angle_deg: float = 45.0


# ─── Job store (single-user box; in-memory is fine) ───

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
GENERATE_LOCK = threading.Lock()  # one CAD generation at a time


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")[:48] or "model"
    return s


def _unique_name(base: str) -> str:
    name = base
    i = 1
    while os.path.exists(os.path.join(OUTPUT_DIR, f"{name}.stl")):
        i += 1
        name = f"{base}_{i}"
    return name


def _meta_path(name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{name}.meta.json")


def _write_meta(name: str, meta: dict):
    with open(_meta_path(name), "w") as f:
        json.dump(meta, f, indent=1)


def _job_worker(job_id: str, prompt: str, name: str, mode: str, image: Optional[str] = None):
    stl = os.path.join(OUTPUT_DIR, f"{name}.stl")
    step = os.path.join(OUTPUT_DIR, f"{name}.step")

    def progress(attempt, phase):
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = {"attempt": attempt, "phase": phase}

    def progress_msg(msg):
        # map string progress from HQ into attempt/phase shape
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = {"attempt": 1, "phase": str(msg)[:80]}

    with GENERATE_LOCK:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
        try:
            # Route:
            #   fast       → exact primitive syntax only
            #   imagine    → organic figurine sculpt (stylized toy)
            #   imagine_hq → Flux/photo → Hunyuan3D high-fidelity mesh
            #   llm        → CadQuery mechanical path only
            #   auto       → fast → organic(if creature) → CadQuery
            #                (HQ is opt-in via mode=imagine_hq — heavier)
            if mode == "imagine_hq":
                img_path = None
                if image:
                    cand = image if os.path.isabs(image) else os.path.join(OUTPUT_DIR, image)
                    if os.path.exists(cand):
                        img_path = cand
                    else:
                        cand2 = os.path.join(OUTPUT_DIR, os.path.basename(image))
                        if os.path.exists(cand2):
                            img_path = cand2
                result = hq_comfy.run_imagine_hq(
                    prompt, stl, image_path=img_path, on_progress=progress_msg
                )
                if os.path.exists(step):
                    try:
                        os.unlink(step)
                    except OSError:
                        pass
            elif mode == "imagine" or (mode == "auto" and organic_gen.is_organic_prompt(prompt)):
                # organic path produces STL only (no STEP — it's a mesh sculpture)
                result = organic_gen.generate_organic(prompt, stl, on_progress=progress)
                if os.path.exists(step):
                    try:
                        os.unlink(step)
                    except OSError:
                        pass
            else:
                code = None if mode == "llm" else fast_path.match(prompt)
                if code is not None and mode != "llm":
                    progress(1, "exec")
                    run = llm_codegen.run_cadquery_code(code, stl, step)
                    result = {
                        "ok": run.get("ok", False),
                        "error": run.get("error"),
                        "code": code,
                        "stats": run.get("stats", {}),
                        "warnings": run.get("warnings", []),
                        "attempts": [{"n": 1, "ok": run.get("ok", False), "error": run.get("error")}],
                        "path_used": "fast",
                        "elapsed_s": 0.0,
                    }
                elif mode == "fast":
                    result = {
                        "ok": False,
                        "error": "Prompt is not simple primitive syntax; use auto, imagine, or llm mode.",
                        "path_used": "fast",
                    }
                else:
                    result = llm_codegen.generate_model(prompt, stl, step, on_progress=progress)
                    # template path is still mechanical CAD; surface detail in meta
                    detail = result.get("path_detail") or "llm"
                    result["path_used"] = "template" if detail == "template" else "llm"
        except Exception as e:  # never let a job vanish silently
            result = {"ok": False, "error": f"internal error: {e}", "path_used": mode}

    result["name"] = name
    result["prompt"] = prompt
    if result.get("ok"):
        result["files"] = {
            "stl": f"/output/{name}.stl",
            "step": f"/output/{name}.step" if os.path.exists(step) else None,
        }
        meta = {
            "prompt": prompt,
            "code": result.get("code"),
            "created": time.time(),
            "stats": result.get("stats", {}),
            "warnings": result.get("warnings", []),
            "path_used": result.get("path_used"),
        }
        if result.get("expanded_prompt"):
            meta["expanded_prompt"] = result["expanded_prompt"]
        if result.get("quality"):
            meta["quality"] = result["quality"]
        if result.get("source_image"):
            meta["source_image"] = result["source_image"]
        if result.get("engine"):
            meta["engine"] = result["engine"]
        if result.get("dimensions"):
            meta["dimensions"] = result["dimensions"]
        if result.get("size_intent"):
            # compact: drop nested explicit dump noise for library
            si = result["size_intent"]
            meta["size_intent"] = {
                k: si[k]
                for k in (
                    "product", "source", "confidence", "height_mm", "width_mm",
                    "depth_mm", "diameter_mm", "bore_mm", "across_flats_mm",
                    "wall_mm", "expected_bbox_mm", "sources",
                )
                if k in si and si[k] is not None
            }
        if result.get("dimension_brief"):
            meta["dimension_brief"] = result["dimension_brief"]
        if result.get("path_detail"):
            meta["path_detail"] = result["path_detail"]
        if result.get("layout_validation"):
            # Store compact pass/fail + errors only (not full probe dump)
            lv = result["layout_validation"]
            meta["layout_validation"] = {
                "ok": lv.get("ok"),
                "errors": lv.get("errors") or [],
                "checks_passed": sum(1 for c in (lv.get("checks") or []) if c.get("ok")),
                "checks_total": len(lv.get("checks") or []),
            }
        if result.get("inspection"):
            insp = result["inspection"]
            meta["inspection"] = {
                "disposition": insp.get("disposition"),
                "grade": insp.get("grade"),
                "tolerance_class": insp.get("tolerance_class"),
                "tolerance_class_name": insp.get("tolerance_class_name"),
                "conformance_summary": insp.get("conformance_summary"),
                "dimensional_ok": (insp.get("dimensional") or {}).get("dimensional_ok"),
                "integrity_ok": (insp.get("integrity") or {}).get("integrity_ok"),
                "n_dim_pass": (insp.get("dimensional") or {}).get("n_pass"),
                "n_dim_fail": (insp.get("dimensional") or {}).get("n_fail"),
                "min_wall_mm": (insp.get("integrity") or {}).get("min_wall_mm"),
                "stl_sha256_16": (insp.get("part") or {}).get("stl_sha256_16"),
            }
        _write_meta(name, meta)
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result


# ─── Endpoints ───

@app.get("/")
async def index():
    return FileResponse(os.path.join(WWW_DIR, "viewer.html"))


@app.get("/api/status")
def status():
    model = None
    try:
        import requests as rq
        r = rq.get(f"{llm_codegen.LLM_BASE_URL}/models", timeout=4)
        if r.status_code == 200:
            data = r.json().get("models") or r.json().get("data") or []
            if data:
                model = os.path.basename(data[0].get("name") or data[0].get("id") or "?")
    except Exception:
        pass
    flux = {}
    try:
        flux = hq_comfy.flux_assets_available()
    except Exception:
        flux = {"ready": False}
    comfy = hq_comfy.comfy_up()
    hunyuan = hq_comfy.hunyuan_available() if comfy else False
    flux_ready = bool(flux.get("ready"))
    return {
        "llm_up": model is not None,
        "model": model,
        "busy": GENERATE_LOCK.locked(),
        "output_count": len(glob.glob(os.path.join(OUTPUT_DIR, "*.stl"))),
        "comfy_up": comfy,
        "hunyuan_up": hunyuan,
        "flux_ready": flux_ready,
        # text→HQ needs Flux+Hunyuan; photo→HQ only needs Hunyuan (pass image=)
        "imagine_hq_ready": comfy and hunyuan and flux_ready,
        "hq_quality": hq_comfy.hq_quality_settings() if comfy else None,
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    if req.mode not in ("auto", "fast", "llm", "imagine", "imagine_hq"):
        raise HTTPException(status_code=400, detail="mode must be auto|fast|llm|imagine|imagine_hq")
    name = _unique_name(_slugify(req.name or prompt))
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "progress": None, "result": None, "created": time.time()}
    t = threading.Thread(
        target=_job_worker, args=(job_id, prompt, name, req.mode, req.image), daemon=True
    )
    t.start()
    return {"job_id": job_id, "name": name}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return dict(job)


@app.post("/api/run-code")
def run_code(req: RunCodeRequest):
    """Execute user-edited CadQuery code directly (no LLM)."""
    code = req.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Empty code")
    name = _unique_name(_slugify(req.name or "edited"))
    stl = os.path.join(OUTPUT_DIR, f"{name}.stl")
    step = os.path.join(OUTPUT_DIR, f"{name}.step")
    with GENERATE_LOCK:
        run = llm_codegen.run_cadquery_code(code, stl, step)
    result = {
        "ok": run.get("ok", False),
        "error": run.get("error"),
        "name": name,
        "code": code,
        "stats": run.get("stats", {}),
        "warnings": run.get("warnings", []),
        "path_used": "manual",
    }
    if result["ok"]:
        result["files"] = {
            "stl": f"/output/{name}.stl",
            "step": f"/output/{name}.step" if os.path.exists(step) else None,
        }
        _write_meta(name, {
            "prompt": "(manually edited code)",
            "code": code,
            "created": time.time(),
            "stats": result["stats"],
            "warnings": result["warnings"],
            "path_used": "manual",
        })
    return result


@app.get("/api/list")
def list_models():
    items = []
    for stl in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.stl")), key=os.path.getmtime, reverse=True):
        name = os.path.splitext(os.path.basename(stl))[0]
        meta = {}
        if os.path.exists(_meta_path(name)):
            try:
                with open(_meta_path(name)) as f:
                    meta = json.load(f)
            except Exception:
                pass
        items.append({
            "name": name,
            "stl": f"/output/{name}.stl",
            "step": f"/output/{name}.step" if os.path.exists(os.path.join(OUTPUT_DIR, f"{name}.step")) else None,
            "size_kb": round(os.path.getsize(stl) / 1024, 1),
            "prompt": meta.get("prompt"),
            "has_code": bool(meta.get("code")),
            "stats": meta.get("stats", {}),
            "warnings": meta.get("warnings", []),
            "path_used": meta.get("path_used"),
            "dimensions": meta.get("dimensions"),
            "created": meta.get("created") or os.path.getmtime(stl),
            "last_analysis": meta.get("last_analysis"),
        })
    return items


@app.get("/api/model/{name}")
def model_detail(name: str):
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
        raise HTTPException(status_code=400, detail="bad name")
    if not os.path.exists(_meta_path(name)):
        raise HTTPException(status_code=404, detail="No metadata for this model")
    with open(_meta_path(name)) as f:
        return json.load(f)


@app.delete("/api/model/{name}")
def delete_model(name: str):
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
        raise HTTPException(status_code=400, detail="bad name")
    removed = []
    for ext in (".stl", ".step", ".meta.json"):
        p = os.path.join(OUTPUT_DIR, name + ext)
        if os.path.exists(p):
            os.unlink(p)
            removed.append(ext)
    if not removed:
        raise HTTPException(status_code=404, detail="Not found")
    return {"removed": removed}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".stl"):
        raise HTTPException(status_code=400, detail="Only .stl files are accepted")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    name = _unique_name(_slugify(os.path.splitext(file.filename)[0]))
    path = os.path.join(OUTPUT_DIR, f"{name}.stl")
    with open(path, "wb") as f:
        f.write(content)
    _write_meta(name, {"prompt": f"(uploaded: {file.filename})", "created": time.time(), "path_used": "upload"})
    return {"name": name, "stl": f"/output/{name}.stl", "size_kb": round(len(content) / 1024, 1)}


# ─── Print Assistant ───

def _resolve_stl_path(name: Optional[str] = None, path: Optional[str] = None) -> str:
    if name:
        if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
            raise HTTPException(status_code=400, detail="bad name")
        p = os.path.join(OUTPUT_DIR, f"{name}.stl")
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"No STL named '{name}'")
        return p
    if path:
        # only allow files inside OUTPUT_DIR (no arbitrary FS read from the web API)
        ap = os.path.realpath(path)
        root = os.path.realpath(OUTPUT_DIR)
        if not ap.startswith(root + os.sep) and ap != root:
            # also allow basename-only references
            cand = os.path.join(OUTPUT_DIR, os.path.basename(path))
            if os.path.exists(cand):
                return cand
            raise HTTPException(status_code=403, detail="path must be inside the sandbox output dir")
        if not os.path.exists(ap):
            raise HTTPException(status_code=404, detail="STL not found")
        return ap
    raise HTTPException(status_code=400, detail="Provide name or path of an STL")


@app.get("/api/printers")
def printers():
    return {
        "printers": print_assistant.list_printers(),
        "filaments": print_assistant.list_filaments(),
        "uses": print_assistant.list_uses(),
        "demo_speeds": [2, 10, 60],
        "defaults": print_assistant.DEFAULTS,
    }


class InspectRequest(BaseModel):
    """ISO-style dimensional + process inspection certificate."""
    name: Optional[str] = None
    path: Optional[str] = None
    tolerance_class: str = "iso2768-m"  # iso2768-f|m|c|v | fdm-engineering
    nozzle_mm: float = 0.4


@app.post("/api/inspect")
def inspect(req: InspectRequest):
    """Produce an ISO 2768-style inspection certificate for an existing model."""
    import iso_quality

    stl_path = _resolve_stl_path(req.name, req.path)
    name = req.name or os.path.splitext(os.path.basename(stl_path))[0]
    step_path = os.path.join(OUTPUT_DIR, f"{name}.step")
    meta = {}
    mp = _meta_path(name)
    if os.path.exists(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    size = meta.get("size_intent") or meta.get("dimensions")
    stats = meta.get("stats") or {}
    if not stats.get("bbox_mm"):
        try:
            import trimesh
            m = trimesh.load(stl_path, force="mesh")
            stats = {
                "bbox_mm": [round(float(x), 3) for x in m.extents],
                "faces": int(len(m.faces)),
                "watertight": bool(m.is_watertight),
            }
        except Exception:
            pass
    cert = iso_quality.build_certificate(
        name=name,
        prompt=meta.get("prompt") or name,
        stl_path=stl_path,
        step_path=step_path if os.path.exists(step_path) else None,
        size_contract=size,
        stats=stats,
        path_used=meta.get("path_used"),
        code=meta.get("code"),
        layout_validation=meta.get("layout_validation"),
        tolerance_class=req.tolerance_class,
        nozzle_mm=req.nozzle_mm,
    )
    # persist
    try:
        qj = os.path.join(OUTPUT_DIR, f"{name}.quality.json")
        with open(qj, "w") as f:
            json.dump(cert, f, indent=1)
        with open(os.path.join(OUTPUT_DIR, f"{name}.quality.txt"), "w") as f:
            f.write(iso_quality.certificate_text(cert))
    except OSError:
        pass
    return {
        "inspection": cert,
        "report_text": iso_quality.certificate_text(cert),
        "files": {
            "quality_json": f"/output/{name}.quality.json",
            "quality_txt": f"/output/{name}.quality.txt",
        },
    }


@app.get("/api/tolerance-classes")
def tolerance_classes():
    import iso_quality
    return {"classes": iso_quality.TOLERANCE_CLASSES, "default": iso_quality.DEFAULT_CLASS}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Inspect STL: bed fit, mesh health, printability, demo walkthrough, slicer settings."""
    stl_path = _resolve_stl_path(req.name, req.path)
    printer = {
        "printer_model": req.printer_model,
        "bed_x": req.bed_x,
        "bed_y": req.bed_y,
        "bed_z": req.bed_z,
        "nozzle_mm": req.nozzle_mm,
        "layer_height_mm": req.layer_height_mm,
        "filament": req.filament,
        "slicer": req.slicer,
        "intended_use": req.intended_use,
        "demo_speed": req.demo_speed,
        "clearance_mm": req.clearance_mm,
        "support_angle_deg": req.support_angle_deg,
    }
    try:
        report = print_assistant.analyze_stl(stl_path, printer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"analysis failed: {e}") from e

    # cache last analysis on the model meta (non-fatal if meta missing)
    name = req.name or os.path.splitext(os.path.basename(stl_path))[0]
    meta_p = _meta_path(name)
    if os.path.exists(meta_p):
        try:
            with open(meta_p) as f:
                meta = json.load(f)
            meta["last_analysis"] = {
                "fits_bed": report["fits_bed"],
                "printable_as_is": report["printable_as_is"],
                "risk_level": report["risk_level"],
                "summary": report["summary"],
                "printer_model": report["printer"]["printer_model"],
                "ts": time.time(),
            }
            _write_meta(name, meta)
        except Exception:
            pass

    return report


@app.get("/output/{filename}")
def serve_output(filename: str):
    if not re.fullmatch(r"[a-zA-Z0-9_\-.]+", filename) or ".." in filename:
        raise HTTPException(status_code=403, detail="bad filename")
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    media = "application/step" if filename.endswith(".step") else "model/stl"
    return FileResponse(path, media_type=media, filename=filename)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SANDBOX_PORT", "8050"))
    host = os.environ.get("BIND_HOST", "127.0.0.1")
    print(f"STL Sandbox on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
