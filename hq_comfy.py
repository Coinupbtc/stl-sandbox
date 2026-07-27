"""ComfyUI orchestration for high-quality image→3D (Hunyuan3D) and later TRELLIS.

Used by the sandbox Imagine HQ path. Talks to local ComfyUI HTTP API on :8188.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
# Paths are env-overridable; defaults use ~/comfy/ComfyUI (no hardcoded username).
_COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", str(Path.home() / "comfy" / "ComfyUI")))
COMFY_INPUT = Path(os.environ.get("COMFY_INPUT", str(_COMFY_ROOT / "input")))
COMFY_OUTPUT = Path(os.environ.get("COMFY_OUTPUT", str(_COMFY_ROOT / "output")))
HUNYUAN_CKPT = os.environ.get("HUNYUAN_CKPT", "hunyuan_3d_v2.1.safetensors")


def _req(method: str, path: str, data: Optional[dict] = None, timeout: float = 60):
    url = f"{COMFY_URL.rstrip('/')}{path}"
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw)


def comfy_up() -> bool:
    try:
        _req("GET", "/system_stats", timeout=4)
        return True
    except Exception:
        return False


def hunyuan_available() -> bool:
    try:
        info = _req("GET", "/object_info/ImageOnlyCheckpointLoader", timeout=8)
        names = info["ImageOnlyCheckpointLoader"]["input"]["required"]["ckpt_name"][0]
        return HUNYUAN_CKPT in names
    except Exception:
        return False


def stage_image(src_path: str, name: str = "hq_ref.png") -> str:
    """Copy image into Comfy input dir; return the filename Comfy LoadImage expects."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    # Keep extension
    ext = src.suffix.lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    dest_name = name if name.endswith(ext) else f"{Path(name).stem}{ext}"
    dest = COMFY_INPUT / dest_name
    shutil.copy2(src, dest)
    return dest_name


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# HQ quality defaults (override via env). Higher octree/size = sharper mesh, more RAM.
# Safe baseline on this box: Flux 768² + Hunyuan octree 384. Drop to 512/256 if OOM.
HQ_FLUX_SIZE = _env_int("STL_HQ_FLUX_SIZE", 768)
HQ_FLUX_STEPS = _env_int("STL_HQ_FLUX_STEPS", 4)
HQ_FLUX_GUIDANCE = _env_float("STL_HQ_FLUX_GUIDANCE", 3.5)
HQ_HY_STEPS = _env_int("STL_HQ_HY_STEPS", 40)
HQ_HY_CFG = _env_float("STL_HQ_HY_CFG", 5.0)
HQ_HY_RESOLUTION = _env_int("STL_HQ_HY_RESOLUTION", 3072)
HQ_HY_OCTREE = _env_int("STL_HQ_HY_OCTREE", 384)
HQ_HY_CHUNKS = _env_int("STL_HQ_HY_CHUNKS", 12000)
HQ_HY_THRESHOLD = _env_float("STL_HQ_HY_THRESHOLD", 0.6)

# Optional Flux unet paths outside Comfy's models tree (override with STL_FLUX_UNET_DIR)
_FLUX_HOME = Path(os.environ.get("STL_FLUX_UNET_DIR", str(Path.home() / "models" / "flux")))
_FLUX_UNET_FALLBACKS = (
    _FLUX_HOME / "flux-schnell-fp8" / "flux1-schnell-fp8.safetensors",
    _FLUX_HOME / "flux-dev-fp8" / "flux1-dev-fp8.safetensors",
)


def expand_hq_image_prompt(user_prompt: str) -> str:
    """Expand a short user idea into a Flux prompt optimized for image→3D.

    Hunyuan3D reconstructs from a single image, so the reference must maximize
    geometric clarity: full subject, clear silhouette, neutral bg, even light,
    no props/text/transparency. User subject text is kept verbatim up front.
    """
    subject = (user_prompt or "").strip()
    if not subject:
        subject = "a detailed 3D figurine"

    # Avoid double-wrapping if caller already sent a full reconstruction prompt
    already = (
        "plain" in subject.lower()
        and "background" in subject.lower()
        and ("studio" in subject.lower() or "product photo" in subject.lower())
    )
    if already and len(subject) > 120:
        return subject

    # Framing + materials tuned for single-view mesh recovery
    suffix = (
        "highly detailed solid 3D object, accurate proportions, clear silhouette, "
        "single subject only, fully visible head-to-toe with small margin, "
        "centered in frame, three-quarter front view, slight turn to the right, "
        "standing on an invisible ground plane, no cropping, "
        "plain seamless light gray studio background, no floor grid, no props, "
        "no hands, no pedestal, no baseplate, no text, no watermark, no logo, "
        "even soft studio lighting, gentle fill light, minimal hard shadows, "
        "matte opaque surface, no glass, no transparency, no mirror reflections, "
        "sharp focus, high geometric detail, product photography for 3D scanning, "
        "photoreal sculpted form suitable for 3D reconstruction"
    )
    return f"{subject}, {suffix}"


def build_hunyuan_workflow(
    image_name: str,
    *,
    seed: int = 42,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    resolution: Optional[int] = None,
    octree_resolution: Optional[int] = None,
    num_chunks: Optional[int] = None,
    threshold: Optional[float] = None,
    filename_prefix: str = "hq/hunyuan",
) -> dict:
    """API-format prompt graph: image → Hunyuan3D 2.1 → GLB mesh."""
    steps = HQ_HY_STEPS if steps is None else steps
    cfg = HQ_HY_CFG if cfg is None else cfg
    resolution = HQ_HY_RESOLUTION if resolution is None else resolution
    octree_resolution = HQ_HY_OCTREE if octree_resolution is None else octree_resolution
    num_chunks = HQ_HY_CHUNKS if num_chunks is None else num_chunks
    threshold = HQ_HY_THRESHOLD if threshold is None else threshold
    return {
        "1": {
            "class_type": "ImageOnlyCheckpointLoader",
            "inputs": {"ckpt_name": HUNYUAN_CKPT},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "3": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 1.0},
        },
        "4": {
            "class_type": "EmptyLatentHunyuan3Dv2",
            "inputs": {"resolution": resolution, "batch_size": 1},
        },
        "5": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["1", 1],
                "image": ["2", 0],
                "crop": "center",
            },
        },
        "6": {
            "class_type": "Hunyuan3Dv2Conditioning",
            "inputs": {"clip_vision_output": ["5", 0]},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["4", 0],
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecodeHunyuan3D",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 2],
                "num_chunks": num_chunks,
                "octree_resolution": octree_resolution,
            },
        },
        "9": {
            "class_type": "VoxelToMesh",
            "inputs": {
                "voxel": ["8", 0],
                "algorithm": "surface net",
                "threshold": threshold,
            },
        },
        "10": {
            "class_type": "SaveGLB",
            "inputs": {
                "mesh": ["9", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }


def queue_prompt(workflow: dict, client_id: str = "stl-sandbox-hq") -> str:
    resp = _req("POST", "/prompt", {"prompt": workflow, "client_id": client_id}, timeout=120)
    if not resp or "prompt_id" not in resp:
        raise RuntimeError(f"Comfy queue failed: {resp}")
    if resp.get("node_errors"):
        raise RuntimeError(f"Comfy node errors: {resp['node_errors']}")
    return resp["prompt_id"]


def wait_prompt(prompt_id: str, timeout_s: float = 900, on_progress=None) -> dict:
    """Poll history until the prompt finishes. Returns history entry."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            hist = _req("GET", f"/history/{prompt_id}", timeout=30)
        except Exception as e:
            if on_progress:
                on_progress(f"poll error: {e}")
            time.sleep(3)
            continue
        if hist and prompt_id in hist:
            entry = hist[prompt_id]
            status = (entry.get("status") or {}).get("status_str") or "unknown"
            if on_progress:
                on_progress(status)
            # completed or error
            if entry.get("outputs") or status in ("success", "error"):
                return entry
        else:
            # still queued/running
            try:
                q = _req("GET", "/queue", timeout=10)
                running = q.get("queue_running") or []
                pending = q.get("queue_pending") or []
                if on_progress:
                    on_progress(f"queued r={len(running)} p={len(pending)}")
            except Exception:
                pass
        time.sleep(3)
    raise TimeoutError(f"Comfy prompt {prompt_id} timed out after {timeout_s}s")


def find_output_files(history_entry: dict) -> list[Path]:
    """Collect mesh files produced by SaveGLB from history outputs."""
    found: list[Path] = []
    outputs = history_entry.get("outputs") or {}
    for node_id, out in outputs.items():
        # SaveGLB may put files under various keys
        for key in ("glb", "mesh", "images", "3d", "files"):
            items = out.get(key) or []
            if isinstance(items, dict):
                items = [items]
            for it in items:
                if not isinstance(it, dict):
                    continue
                fname = it.get("filename")
                sub = it.get("subfolder") or ""
                if not fname:
                    continue
                p = COMFY_OUTPUT / sub / fname if sub else COMFY_OUTPUT / fname
                if p.exists():
                    found.append(p)
    # fallback: newest glb under output/hq
    if not found:
        for pattern in ("hq/**/*.glb", "**/*hunyuan*.glb", "**/*.glb"):
            cands = sorted(COMFY_OUTPUT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                # only very recent (last 15 min)
                now = time.time()
                for c in cands[:5]:
                    if now - c.stat().st_mtime < 900:
                        found.append(c)
                if found:
                    break
    return found


def glb_to_stl(glb_path: Path, stl_path: Path) -> dict:
    """Convert GLB mesh to STL via trimesh; scale optional later."""
    import trimesh

    scene_or_mesh = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("GLB has no mesh geometry")
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        mesh = scene_or_mesh

    # Sit on bed Z=0, center XY
    mesh.rezero()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation([-center[0], -center[1], -bounds[0][2]])

    # Scale longest axis to ~100 mm if model is unit-ish (< 5 units)
    ext = mesh.extents
    longest = float(max(ext))
    if longest < 5.0:
        mesh.apply_scale(100.0 / longest)
    elif longest > 400:
        mesh.apply_scale(200.0 / longest)

    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    warnings: list[str] = []
    try:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
    except Exception as e:
        warnings.append(f"repair partial: {e}")

    # Voxel remesh fallback for printability when still open
    if not mesh.is_watertight:
        try:
            pitch = max(float(max(mesh.extents)) / 120.0, 0.35)
            vox = mesh.voxelized(pitch)
            filled = vox.marching_cubes
            if filled is not None and len(filled.faces) > 100:
                mesh = filled
                mesh.rezero()
                b = mesh.bounds
                c = (b[0] + b[1]) / 2.0
                mesh.apply_translation([-c[0], -c[1], -b[0][2]])
                warnings.append(f"voxel-remeshed @ {pitch:.2f} mm for watertight shell")
        except Exception as e:
            warnings.append(f"voxel remesh skipped: {e}")

    if not mesh.is_watertight:
        warnings.append("mesh still not watertight — use slicer repair before printing")

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(stl_path))
    return {
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "bbox_mm": [round(float(x), 1) for x in mesh.extents],
        "volume_cm3": round(float(abs(mesh.volume)) / 1000.0, 2) if mesh.is_watertight else None,
        "warnings": warnings,
    }


def run_hunyuan_to_stl(
    image_path: str,
    stl_path: str,
    *,
    seed: int = 42,
    on_progress=None,
    **workflow_kw,
) -> dict:
    """Full pipeline: stage image → Hunyuan → GLB → STL."""
    if not comfy_up():
        return {"ok": False, "error": "ComfyUI is not reachable on :8188"}
    if not hunyuan_available():
        return {
            "ok": False,
            "error": f"Hunyuan checkpoint '{HUNYUAN_CKPT}' not found in Comfy checkpoints",
        }

    notify = on_progress or (lambda m: None)
    notify("staging image")
    img_name = stage_image(image_path, name="hq_ref" + Path(image_path).suffix.lower())
    prefix = workflow_kw.pop("filename_prefix", "hq/hunyuan")
    wf = build_hunyuan_workflow(img_name, seed=seed, filename_prefix=prefix, **workflow_kw)
    notify("queueing Hunyuan3D")
    t0 = time.time()
    try:
        pid = queue_prompt(wf)
    except Exception as e:
        return {"ok": False, "error": f"queue failed: {e}"}

    notify(f"running prompt {pid}")
    try:
        hist = wait_prompt(pid, timeout_s=1200, on_progress=notify)
    except Exception as e:
        return {"ok": False, "error": str(e), "prompt_id": pid}

    status = (hist.get("status") or {})
    if status.get("status_str") == "error" or status.get("completed") is False:
        msgs = status.get("messages") or []
        return {"ok": False, "error": f"Comfy error: {msgs[-3:]}", "prompt_id": pid}

    files = find_output_files(hist)
    if not files:
        # last-ditch scan
        files = find_output_files({"outputs": {}})
    if not files:
        return {
            "ok": False,
            "error": "No GLB/mesh output found after Hunyuan run",
            "prompt_id": pid,
            "history_keys": list((hist.get("outputs") or {}).keys()),
        }

    glb = files[0]
    notify(f"converting {glb.name} → STL")
    try:
        stats = glb_to_stl(glb, Path(stl_path))
    except Exception as e:
        return {"ok": False, "error": f"GLB→STL failed: {e}", "glb": str(glb), "prompt_id": pid}

    return {
        "ok": True,
        "prompt_id": pid,
        "glb": str(glb),
        "stl": stl_path,
        "stats": stats,
        "elapsed_s": round(time.time() - t0, 1),
        "engine": "hunyuan3d-2.1",
        "source_image": image_path,
    }


def flux_assets_available() -> dict:
    """Check whether Flux schnell text→image support files are present."""
    te = _COMFY_ROOT / "models" / "text_encoders"
    clip = _COMFY_ROOT / "models" / "clip"
    vae = _COMFY_ROOT / "models" / "vae"
    diff = _COMFY_ROOT / "models" / "diffusion_models"
    ckpt = _COMFY_ROOT / "models" / "checkpoints"

    def exists_any(*paths):
        return any(Path(p).exists() for p in paths)

    unet = exists_any(
        diff / "flux1-schnell-fp8.safetensors",
        ckpt / "flux1-schnell-fp8.safetensors",
        diff / "flux1-dev-fp8.safetensors",
        ckpt / "flux1-dev-fp8.safetensors",
    )
    clip_l = exists_any(te / "clip_l.safetensors", clip / "clip_l.safetensors")
    t5 = exists_any(
        te / "t5xxl_fp8_e4m3fn.safetensors",
        clip / "t5xxl_fp8_e4m3fn.safetensors",
        te / "t5xxl_fp16.safetensors",
        clip / "t5xxl_fp16.safetensors",
    )
    ae = (vae / "ae.safetensors").exists()
    return {
        "unet": unet,
        "clip_l": clip_l,
        "t5": t5,
        "vae": ae,
        "ready": unet and clip_l and t5 and ae,
    }


def _flux_filenames() -> dict:
    te = _COMFY_ROOT / "models" / "text_encoders"
    clip = _COMFY_ROOT / "models" / "clip"
    vae = _COMFY_ROOT / "models" / "vae"
    diff = _COMFY_ROOT / "models" / "diffusion_models"
    ckpt = _COMFY_ROOT / "models" / "checkpoints"
    diff.mkdir(parents=True, exist_ok=True)

    def pick(*cands):
        for c in cands:
            if Path(c).exists():
                return Path(c).name
        return None

    # DualCLIPLoader looks in clip/ and text_encoders/
    clip_l = pick(clip / "clip_l.safetensors", te / "clip_l.safetensors")
    t5 = pick(
        clip / "t5xxl_fp8_e4m3fn.safetensors",
        te / "t5xxl_fp8_e4m3fn.safetensors",
        clip / "t5xxl_fp16.safetensors",
        te / "t5xxl_fp16.safetensors",
    )
    # UNETLoader looks in diffusion_models / unet
    unet = pick(
        diff / "flux1-schnell-fp8.safetensors",
        diff / "flux1-dev-fp8.safetensors",
    )
    # Symlink from checkpoints or host models tree if Comfy path is empty
    if not unet:
        link_candidates = [
            ckpt / "flux1-schnell-fp8.safetensors",
            ckpt / "flux1-dev-fp8.safetensors",
            *_FLUX_UNET_FALLBACKS,
        ]
        for src in link_candidates:
            if not src.exists():
                continue
            dest = diff / src.name
            if not dest.exists():
                try:
                    dest.symlink_to(src.resolve())
                except OSError:
                    continue
            if dest.exists():
                unet = dest.name
                break
    ae = "ae.safetensors" if (vae / "ae.safetensors").exists() else None
    return {"unet": unet, "clip_l": clip_l, "t5": t5, "vae": ae}


def build_flux_t2i_workflow(
    prompt: str,
    *,
    seed: int = 42,
    steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    guidance: Optional[float] = None,
    filename_prefix: str = "hq/flux_ref",
) -> dict:
    """Flux Schnell text→image (high-clarity reference for Hunyuan)."""
    names = _flux_filenames()
    missing = [k for k, v in names.items() if not v]
    if missing:
        raise RuntimeError(f"Flux assets missing: {missing}")

    steps = HQ_FLUX_STEPS if steps is None else steps
    size = HQ_FLUX_SIZE
    width = size if width is None else width
    height = size if height is None else height
    guidance = HQ_FLUX_GUIDANCE if guidance is None else guidance

    # Schnell: few steps, cfg=1, empty negative via ConditioningZeroOut
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": names["unet"], "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": names["clip_l"],
                "clip_name2": names["t5"],
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": names["vae"]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": prompt},
        },
        "5": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["4", 0], "guidance": guidance},
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": filename_prefix},
        },
    }


def find_saved_images(history_entry: dict) -> list[Path]:
    found: list[Path] = []
    for _nid, out in (history_entry.get("outputs") or {}).items():
        for it in out.get("images") or []:
            fname = it.get("filename")
            sub = it.get("subfolder") or ""
            if not fname:
                continue
            p = COMFY_OUTPUT / sub / fname if sub else COMFY_OUTPUT / fname
            if p.exists():
                found.append(p)
    return found


def run_flux_text_to_image(
    prompt: str,
    *,
    seed: int = 42,
    on_progress=None,
    out_copy: Optional[str] = None,
    expand_prompt: bool = True,
) -> dict:
    """Generate a reference image with Flux Schnell (3D-recon optimized)."""
    if not comfy_up():
        return {"ok": False, "error": "ComfyUI not up"}
    assets = flux_assets_available()
    if not assets["ready"]:
        return {"ok": False, "error": f"Flux assets incomplete: {assets}"}

    notify = on_progress or (lambda m: None)
    full = expand_hq_image_prompt(prompt) if expand_prompt else (prompt or "").strip()
    notify(f"queueing Flux text→image ({HQ_FLUX_SIZE}², {HQ_FLUX_STEPS} steps)")
    t0 = time.time()
    try:
        wf = build_flux_t2i_workflow(full, seed=seed)
        pid = queue_prompt(wf)
        hist = wait_prompt(pid, timeout_s=600, on_progress=notify)
    except Exception as e:
        return {"ok": False, "error": str(e), "expanded_prompt": full}

    imgs = find_saved_images(hist)
    if not imgs:
        return {
            "ok": False,
            "error": "Flux produced no image",
            "prompt_id": pid,
            "expanded_prompt": full,
        }
    img = imgs[0]
    if out_copy:
        Path(out_copy).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_copy)
        img_out = out_copy
    else:
        img_out = str(img)
    return {
        "ok": True,
        "image": img_out,
        "comfy_image": str(img),
        "prompt_id": pid,
        "elapsed_s": round(time.time() - t0, 1),
        "engine": "flux-schnell",
        "user_prompt": prompt,
        "expanded_prompt": full,
        "flux_size": HQ_FLUX_SIZE,
        "flux_steps": HQ_FLUX_STEPS,
    }


def hq_quality_settings() -> dict:
    """Current Imagine HQ quality knobs (env-overridable)."""
    return {
        "flux_size": HQ_FLUX_SIZE,
        "flux_steps": HQ_FLUX_STEPS,
        "flux_guidance": HQ_FLUX_GUIDANCE,
        "hy_steps": HQ_HY_STEPS,
        "hy_cfg": HQ_HY_CFG,
        "hy_resolution": HQ_HY_RESOLUTION,
        "hy_octree": HQ_HY_OCTREE,
        "hy_chunks": HQ_HY_CHUNKS,
        "hy_threshold": HQ_HY_THRESHOLD,
    }


def run_imagine_hq(
    prompt: str,
    stl_path: str,
    *,
    image_path: Optional[str] = None,
    seed: int = 42,
    on_progress=None,
) -> dict:
    """High-quality path: (optional Flux text→image) → Hunyuan3D → STL."""
    notify = on_progress or (lambda m: None)
    t0 = time.time()
    ref = image_path
    flux_meta = None
    quality = hq_quality_settings()

    if not ref:
        notify("flux_ref")
        flux_meta = run_flux_text_to_image(prompt, seed=seed, on_progress=notify)
        if not flux_meta.get("ok"):
            return {
                "ok": False,
                "error": (
                    "Imagine HQ needs a reference image or working Flux text→image. "
                    f"Flux error: {flux_meta.get('error')}"
                ),
                "path_used": "imagine_hq",
                "flux": flux_meta,
                "quality": quality,
                "expanded_prompt": flux_meta.get("expanded_prompt"),
            }
        ref = flux_meta["image"]

    notify(f"hunyuan3d (octree={HQ_HY_OCTREE}, steps={HQ_HY_STEPS})")
    hy = run_hunyuan_to_stl(ref, stl_path, seed=seed, on_progress=notify)
    if not hy.get("ok"):
        hy["path_used"] = "imagine_hq"
        hy["flux"] = flux_meta
        hy["quality"] = quality
        if flux_meta:
            hy["expanded_prompt"] = flux_meta.get("expanded_prompt")
        return hy

    warnings = list((hy.get("stats") or {}).get("warnings") or [])
    warnings.append(
        "Imagine HQ: Flux/photo → Hunyuan3D mesh "
        f"(flux {HQ_FLUX_SIZE}², hy octree {HQ_HY_OCTREE}, steps {HQ_HY_STEPS})"
    )
    return {
        "ok": True,
        "error": None,
        "code": None,
        "stats": {k: v for k, v in (hy.get("stats") or {}).items() if k != "warnings"},
        "warnings": warnings,
        "attempts": [{"n": 1, "ok": True, "error": None}],
        "path_used": "imagine_hq",
        "elapsed_s": round(time.time() - t0, 1),
        "engine": hy.get("engine"),
        "glb": hy.get("glb"),
        "source_image": ref,
        "flux": flux_meta,
        "prompt": prompt,
        "expanded_prompt": (flux_meta or {}).get("expanded_prompt"),
        "quality": quality,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "hq":
        prompt = sys.argv[2] if len(sys.argv) > 2 else "a cute cat figurine"
        out = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).resolve().parent / "output" / "hq_test.stl")
        print("flux assets", flux_assets_available())
        r = run_imagine_hq(prompt, out, on_progress=lambda m: print("  ·", m, flush=True))
        print(json.dumps({k: v for k, v in r.items() if k != "code"}, indent=2, default=str))
        sys.exit(0 if r.get("ok") else 1)

    img = sys.argv[1] if len(sys.argv) > 1 else str(COMFY_INPUT / "ref_cat.jpg")
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent / "output" / "hq_cat_hunyuan.stl")
    print("comfy_up", comfy_up(), "hunyuan", hunyuan_available(), "flux", flux_assets_available())
    r = run_hunyuan_to_stl(img, out, on_progress=lambda m: print("  ·", m, flush=True))
    print(json.dumps(r, indent=2))
    sys.exit(0 if r.get("ok") else 1)
