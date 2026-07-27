# Quality comparison — STL Sandbox generators (2026-07-09)

## Pipelines tested

| Path | How | Quality ceiling | Speed | Status |
|---|---|---|---|---|
| **Imagine toy** | LLM + spheres/capsules | Low–med stylized toy | ~15–35s | Working |
| **Mechanical CAD** | LLM + CadQuery | High for functional parts | ~10–90s | Working |
| **Imagine HQ** | Flux Schnell → Hunyuan3D 2.1 → repair STL | High organic mesh | ~70–120s | **Working** |
| **TRELLIS.2** | Image → TRELLIS mesh | High (peer of Hunyuan) | — | Weights on disk; **Comfy nodes not in this build** |

## Measured results (cat / chicken)

| Path | Faces | Watertight | Size (mm) | Notes |
|---|---:|---|---|---|
| Imagine toy cat | ~3.5k | yes | ~95×105×97 | Recognizable blob toy |
| HQ photo→Hunyuan cat | ~149k | yes* | ~83×113×121 | Real photo mesh; *after voxel remesh |
| HQ Flux→Hunyuan cat | ~132k | yes* | ~121×81×119 | Printable Yes on Ender3, risk Medium |
| HQ Flux→Hunyuan chicken | ~61k | yes* | ~45×121×65 | Full text→HQ API path ~74s |

## Product-fit / realistic dimensions (2026-07-09)

Mechanical path now **researches real product sizes** before codegen:

| Prompt | Old result | New result |
|---|---|---|
| `iphone 17 pro max cell phone casing` | LLM guess 81×160×75 mm, **64 faces**, solid junk | Research 163.4×78.0×8.75 body → template **82.0×167.4×11.9 mm**, **2020 faces**, hollow shell, size-check pass, **2.2 s** |

Sources: local catalog (Apple/GSMArena-cited) → Wikipedia API → GSMArena HTML → dimensions.com.
Phone cases/stands use parametric CadQuery templates (camera island, USB-C, button recesses).
Other product-fit prompts inject a dimension brief into the LLM and **retry if bbox is >18% off**.

### Feature layout gate (logic errors)

Size-correct cases can still be **wrong** (USB-C and camera on the same end). Cause: CadQuery
`Workplane("XZ").offset(−H/2)` inverted Y so the port landed at **+Y next to the camera**.

Mitigations now in the pipeline:
1. **World-space cutters only** — `.box(...).translate((x,y,z))`; no XZ/YZ offsets for ports
2. **Axis convention locked** — Y− = bottom/USB, Y+ = top/camera, Z0 = back, Z+ = screen
3. **`feature_validate.py`** — solid `isInside` probes + opposite-end rules; fail deletes STL
4. **Regression tests** — `test_feature_layout.py` replays the XZ anti-pattern and expects FAIL

Proven probes: camera air at Y≈+60 back; USB air at Y≈−83 edge; top edge solid; separation ~143 mm.

## Household templates (2026-07-10)

| Prompt | Path | Time | Notes |
|---|---|---:|---|
| coffee mug 85 mm tall | template | **1.3 s** | was ~LLM 10–90s |
| project box 100×60×40 + lid | template | **1.4 s** | base + lid side-by-side |
| desk cable clip | template | **3.1 s** | snap C-channel |
| iPhone 17 Pro Max case | template | **2.4 s** | +MagSafe ring, speaker slots |

## Recommendation

- **Organic “really good”** → mode **`imagine_hq`** (Flux + Hunyuan3D)
- **Fast toy draft** → mode **`imagine`**
- **Functional / product-fit parts** → mode **`auto`/`llm`** (dim research + template/CadQuery)
- **Everyday mug/box/clip/hook/tray/nut** → automatic **template** path (no LLM)
- **TRELLIS.2** next: upgrade Comfy or install native TRELLIS nodes; weights already at:
  - `models/diffusion_models/trellis_2_bf16.safetensors` (9.7G)
  - `models/clip_vision/dino_v3_vit_l.safetensors` (1.2G)
  - `models/vae/trellis_2_shape_vae_bf16.safetensors` (1.1G)

## Imagine HQ quality (2026-07-09 prompt/res upgrade)

User prompt is auto-expanded for **single-view 3D reconstruction** (full body, plain gray
studio bg, ¾ front view, matte opaque, no props/text). Defaults raised from the first HQ cut:

| Knob | Old | New (env override) |
|---|---:|---|
| Flux image size | 512² | **768²** (`STL_HQ_FLUX_SIZE`) |
| Flux steps | 4 | 4 (`STL_HQ_FLUX_STEPS`) |
| Hunyuan steps | 30 | **40** (`STL_HQ_HY_STEPS`) |
| Hunyuan octree | 256 | **384** (`STL_HQ_HY_OCTREE`) |
| Hunyuan chunks | 8000 | **12000** (`STL_HQ_HY_CHUNKS`) |

If Flux/Hunyuan OOMs under concurrent load, set `STL_HQ_FLUX_SIZE=512` and
`STL_HQ_HY_OCTREE=256` on `stl-sandbox.service` and restart. Flux unet is auto-symlinked
from `~/models/flux/flux-schnell-fp8/` when missing under Comfy `diffusion_models/`.

**User tip:** short subject lines work (“a raccoon figurine”); add pose/species detail for
shape accuracy (“standing raccoon, bushy striped tail, sitting upright”). Avoid “chibi /
cartoon blob” if you want accurate anatomy.

## Print notes for HQ meshes

- Always run **Print Assistant** after generate (auto in UI)
- Generative meshes need repair (voxel remesh applied automatically)
- Expect **supports** for legs/ears; medium risk overhangs
- Topology is not CAD-clean — fine for display figurines, not precision fits
