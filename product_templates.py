"""Parametric CadQuery templates for product-fit and everyday printables.

When dimension research succeeds for a phone case (or similar), emit correct
geometry instead of hoping the LLM invents a printable shell.

Also covers common household prints (mug, pen holder, project box, cable clip,
wall hook) with sizes parsed from the prompt — zero LLM latency.

CRITICAL — phone case coordinate convention (do not "simplify" with XZ/YZ
workplane offsets; those invert easily and put USB on the camera end):

  X = width   (−X left, +X right)  phone face-up, top away from you
  Y = height  (−Y BOTTOM / USB-C, +Y TOP / camera & earpiece)
  Z = depth   (Z=0 BACK plate, +Z toward SCREEN / lip)

All feature cutters are axis-aligned boxes placed with .translate() in world
mm so positions are unambiguous and feature_validate can probe them.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import feature_validate
import size_intent


# ─── Prompt size parsing ───────────────────────────────────────────────────

def parse_sizes(prompt: str) -> dict[str, float]:
    """Extract printable dimensions from freeform English. Best-effort, mm."""
    # Prefer the shared size_intent parser (inches, metric, cubes, holes…)
    raw = size_intent.parse_prompt_sizes(prompt)
    # Keep only numeric float fields templates care about
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    # Metric washer/nut fill-ins
    enriched = size_intent._apply_metric(prompt, dict(out))
    for k, v in enriched.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def match_template(prompt: str, dims: Optional[dict] = None) -> Optional[str]:
    """Return CadQuery source for a known product class, or None."""
    p = prompt.lower()
    sizes = parse_sizes(prompt)
    # Merge universal size contract (catalog credit card / soda can / M-series…)
    try:
        contract = size_intent.resolve_size_intent(prompt, researched=dims)
        if contract:
            for k, v in contract.items():
                if k.endswith("_mm") or k in ("angle_deg", "metric_m", "wall_mm"):
                    if isinstance(v, (int, float)) and k not in sizes:
                        sizes[k] = float(v)
    except Exception:
        pass

    # Offline phone catalog: if dims missing but this is clearly a phone case,
    # pull body sizes from dim_research catalog (no network).
    if dims is None and re.search(
        r"\b(iphone|galaxy|pixel|samsung|google)\b", p
    ) and re.search(r"\b(case|casing|cover|sleeve|shell|protector)\b", p):
        try:
            import dim_research
            dims = dim_research.research_dimensions(prompt)
        except Exception:
            dims = None

    part = (dims or {}).get("part_type") or ""

    # ── Product-fit (needs researched body dims) ──
    if dims and all(k in dims for k in ("height_mm", "width_mm", "depth_mm")):
        if part == "phone_case" or re.search(
            r"\b(phone\s+)?(case|casing|cover|sleeve|shell|skin|protector)\b", p
        ):
            # Avoid matching "project case" / instrument case as phone case
            if re.search(r"\b(project|electronics?|junction|switch)\s+box\b", p):
                pass
            elif re.search(
                r"\b(iphone|galaxy|pixel|phone|samsung|google|pro\s*max)\b", p
            ) or part == "phone_case":
                return phone_case_code(dims)

        if part == "holder" or re.search(
            r"\b(phone\s+)?(stand|dock|cradle)\b", p
        ):
            # Don't steal pen holder / can holder / soap / desk cup
            if not re.search(r"\b(pen|pencil|desk\s+cup|mug|can|soap|drink)\b", p):
                return phone_stand_code(dims)

    # ── Everyday household templates (no product research required) ──
    if re.search(r"\b(coffee\s+)?mug\b|\btea\s+cup\b|\bcoffee\s+cup\b", p):
        return mug_code(sizes, prompt)

    if re.search(
        r"\b(pen|pencil)\s*(holder|cup|pot)\b|\bdesk\s+(cup|organizer|caddy)\b"
        r"|\bhex(?:agonal)?\s+(cup|holder|pot)\b",
        p,
    ):
        return pen_holder_code(sizes, prompt)

    if re.search(
        r"\b(project|electronics?|junction|enclosure)\s*box\b"
        r"|\bbox\s+with\s+(a\s+)?lid\b|\bproject\s+enclosure\b",
        p,
    ):
        return project_box_code(sizes, prompt)

    if re.search(
        r"\b(cable|cord|wire)\s*(clip|organizer|clamp|holder|management)\b"
        r"|\bdesk\s+cable\b",
        p,
    ):
        return cable_clip_code(sizes, prompt)

    if re.search(r"\b(wall\s+)?hook\b|\bcoat\s+hook\b|\bkey\s+hook\b", p):
        return wall_hook_code(sizes, prompt)

    if re.search(
        r"\b(name\s*tag|nameplate|name\s*plate|badge\s*base|desk\s*plaque)\b", p
    ):
        return nameplate_code(sizes, prompt)

    if re.search(r"\bhex\s*nut\b|\bnut\b.*\bacross\s+flats\b|\bm\d+\s*nut\b", p):
        return hex_nut_code(sizes, prompt)

    if re.search(r"\bwasher\b", p):
        return washer_code(sizes, prompt)

    if re.search(
        r"\bdesk\s+tray\b|\btray\b.*\bcompartment|\bcompartments?\b.*\btray\b"
        r"|\bwallet\s+tray\b|\bcredit\s+card\b.*\btray\b",
        p,
    ):
        return desk_tray_code(sizes, prompt)

    if re.search(r"\bsoap\s+dish\b|\bsoap\s+holder\b", p):
        return soap_dish_code(sizes, prompt)

    if re.search(
        r"\b(soda|beer|drink|can)\s+holder\b|\bcan\s+cooler\b|\bkoozie\b", p
    ):
        return can_holder_code(sizes, prompt)

    # Phone stand without researched dims (generic phone slot)
    if re.search(r"\b(phone\s+)?(stand|dock|cradle)\b", p) and not re.search(
        r"\b(pen|pencil|mug|can|soap)\b", p
    ):
        return generic_phone_stand_code(sizes, prompt, dims)

    if re.search(r"\bspacer\b|\bstandoff\b", p):
        return spacer_code(sizes, prompt)

    return None


# ─── Phone case ────────────────────────────────────────────────────────────

def phone_case_code(dims: dict) -> str:
    """Protective phone case: hollow shell, open front, camera on BACK/TOP, USB on BOTTOM."""
    layout = feature_validate.phone_case_layout(dims)
    env = layout["envelope"]
    feats = layout["features"]

    ph = float(dims["height_mm"])
    pw = float(dims["width_mm"])
    pd = float(dims["depth_mm"])
    wall = float(dims.get("wall_mm", 1.6))
    clr = float(dims.get("clearance_mm", 0.4))
    lip = float(dims.get("lip_mm", 1.2))
    phone_r = float(dims.get("corner_radius_mm") or min(ph, pw) * 0.12)

    cav_h = ph + 2 * clr
    cav_w = pw + 2 * clr
    cav_d = pd + clr
    out_h = env["h"]
    out_w = env["w"]
    out_d = env["d"]

    outer_r = min(phone_r + clr + wall, min(pw, ph) * 0.18)
    outer_r = min(outer_r, out_w / 2 - 0.5, out_h / 2 - 0.5)
    inner_r = max(min(phone_r + clr * 0.5, outer_r - wall * 0.4), 2.0)

    cam = feats["camera"]
    cam_cx, cam_cy, _ = cam["center"]
    cam_s = cam["size"][0]

    port = feats["usb_c"]
    port_cx, port_cy, port_cz = port["center"]
    port_w, port_depth, port_h = port["size"]  # X, Y extent into wall, Z

    btn_depth = round(wall * 0.55, 3)
    vol_cy = feats["volume"]["center"][1]
    pwr_cy = feats["power"]["center"][1]
    btn_cz = feats["volume"]["center"][2]

    product = (dims.get("product") or "phone").replace('"', "'")
    source = (dims.get("source") or "?").replace('"', "'")

    # Port box: centered at bottom edge, extends inward (+Y) through wall
    port_box_cy = -out_h / 2 + port_depth / 2
    cam_box_cz = wall / 2
    cam_box_d = wall + 0.8

    # MagSafe-ish ring (modern iPhones) — shallow back recess, diameter ~46 mm ID
    magsafe = pw >= 70  # phones only
    ms_od, ms_id = 54.0, 46.0
    ms_depth = min(0.55, wall * 0.35)
    ms_cz = wall - ms_depth / 2

    # Speaker / mic slots beside USB on bottom edge
    spk_w, spk_h = 14.0, 3.2
    spk_depth = port_depth
    spk_off_x = port_w / 2 + 10.0
    spk_cz = port_cz - 1.5
    spk_cy = port_box_cy

    return f'''import cadquery as cq

# Parametric phone case for: {product}
# Dimension source: {source}
# Phone body (researched): {ph} x {pw} x {pd} mm (H x W x D)
# Outer envelope: {out_h:.2f} x {out_w:.2f} x {out_d:.2f} mm (H x W x D)
#
# AXIS CONVENTION (world mm) — do not invert:
#   X = width  (− left, + right)
#   Y = height (− BOTTOM / USB-C, + TOP / camera)
#   Z = depth  (0 = BACK, + = SCREEN)
# Camera cutout: BACK face, TOP end (+Y).  USB-C: BOTTOM edge (−Y). NEVER same end.

PH, PW, PD = {ph}, {pw}, {pd}
WALL, CLR, LIP = {wall}, {clr}, {lip}
CAV_H, CAV_W, CAV_D = {cav_h}, {cav_w}, {cav_d}
OUT_H, OUT_W, OUT_D = {out_h}, {out_w}, {out_d}
OUTER_R, INNER_R = {outer_r:.3f}, {inner_r:.3f}
CAM_S = {cam_s}
CAM_CX, CAM_CY = {cam_cx:.4f}, {cam_cy:.4f}
PORT_W, PORT_H, PORT_DEPTH = {port_w}, {port_h}, {port_depth}
PORT_CZ = {port_cz:.4f}
PORT_BOX_CY = {port_box_cy:.4f}
BTN_D, BTN_CZ = {btn_depth}, {btn_cz:.4f}
VOL_CY, PWR_CY = {vol_cy:.4f}, {pwr_cy:.4f}
RIM = 0.9
MS_OD, MS_ID, MS_DEPTH, MS_CZ = {ms_od}, {ms_id}, {ms_depth:.3f}, {ms_cz:.4f}
SPK_W, SPK_H, SPK_D = {spk_w}, {spk_h}, {spk_depth}
SPK_OFF, SPK_CZ, SPK_CY = {spk_off_x:.3f}, {spk_cz:.4f}, {spk_cy:.4f}

# Outer rounded body (centered on origin in X/Y, sitting on Z=0)
outer = (
    cq.Workplane("XY")
    .rect(OUT_W, OUT_H)
    .extrude(OUT_D)
    .edges("|Z")
    .fillet(OUTER_R)
)

# Phone cavity from above the back wall, open through the front
cavity = (
    cq.Workplane("XY")
    .workplane(offset=WALL)
    .rect(CAV_W, CAV_H)
    .extrude(CAV_D + LIP + 0.5)
    .edges("|Z")
    .fillet(INNER_R)
)
shell = outer.cut(cavity)

# Screen window through the front lip (front = +Z)
screen_w = CAV_W - 2 * RIM
screen_h = CAV_H - 2 * RIM
screen_r = max(INNER_R - RIM, 1.5)
screen = (
    cq.Workplane("XY")
    .workplane(offset=WALL + CAV_D - 0.1)
    .rect(screen_w, screen_h)
    .extrude(LIP + 1.0)
    .edges("|Z")
    .fillet(screen_r)
)
shell = shell.cut(screen)

# ── Camera island: BACK (Z≈0), TOP (+Y), LEFT (−X) ──
camera = (
    cq.Workplane("XY")
    .box(CAM_S, CAM_S, {cam_box_d:.3f})
    .edges("|Z")
    .fillet(3.5)
    .translate((CAM_CX, CAM_CY, {cam_box_cz:.4f}))
)
shell = shell.cut(camera)

# ── USB-C: BOTTOM edge (−Y), mid phone thickness — OPPOSITE end from camera ──
usb = (
    cq.Workplane("XY")
    .box(PORT_W, PORT_DEPTH, PORT_H)
    .translate((0.0, PORT_BOX_CY, PORT_CZ))
)
shell = shell.cut(usb)

# ── Speaker / mic slots beside USB on bottom edge ──
spk_l = (
    cq.Workplane("XY")
    .box(SPK_W, SPK_D, SPK_H)
    .translate((-SPK_OFF, SPK_CY, SPK_CZ))
)
spk_r = (
    cq.Workplane("XY")
    .box(SPK_W, SPK_D, SPK_H)
    .translate((SPK_OFF, SPK_CY, SPK_CZ))
)
shell = shell.cut(spk_l).cut(spk_r)

# ── Side button recesses: volume LEFT (−X), power RIGHT (+X), upper half ──
vol = (
    cq.Workplane("XY")
    .box(BTN_D + 0.4, 18.0, 6.0)
    .translate((-OUT_W / 2 + (BTN_D + 0.4) / 2, VOL_CY, BTN_CZ))
)
pwr = (
    cq.Workplane("XY")
    .box(BTN_D + 0.4, 14.0, 6.0)
    .translate((OUT_W / 2 - (BTN_D + 0.4) / 2, PWR_CY, BTN_CZ))
)
shell = shell.cut(vol).cut(pwr)
''' + (f'''
# ── MagSafe ring recess on BACK (shallow, cosmetic/alignment) ──
ms_outer = cq.Workplane("XY").circle(MS_OD / 2).extrude(MS_DEPTH + 0.2)
ms_inner = cq.Workplane("XY").circle(MS_ID / 2).extrude(MS_DEPTH + 0.4)
ms_ring = ms_outer.cut(ms_inner).translate((0.0, -4.0, MS_CZ - 0.1))
shell = shell.cut(ms_ring)

result = shell
''' if magsafe else '''
result = shell
''')


def phone_stand_code(dims: dict) -> str:
    """Simple desk phone stand with a slot sized to the researched phone."""
    ph = float(dims["height_mm"])
    pw = float(dims["width_mm"])
    pd = float(dims["depth_mm"])
    wall = 3.0
    slot_w = pw + 1.0
    slot_d = pd + 1.2
    base_d = 70.0
    base_w = slot_w + 2 * wall
    base_h = 12.0
    back_h = min(ph * 0.45, 90.0)

    product = (dims.get("product") or "phone").replace('"', "'")
    source = (dims.get("source") or "?").replace('"', "'")

    return f'''import cadquery as cq

# Desk stand for: {product}
# Dimension source: {source}
# Slot sized to phone body {ph} x {pw} x {pd} mm (H x W x D)

SLOT_W, SLOT_D = {slot_w}, {slot_d}
WALL = {wall}
BASE_W, BASE_D, BASE_H = {base_w}, {base_d}, {base_h}
BACK_H = {back_h}

base = cq.Workplane("XY").box(BASE_W, BASE_D, BASE_H, centered=(True, True, False))

back = (
    cq.Workplane("XY")
    .center(0, BASE_D / 2 - WALL)
    .box(BASE_W - 4, WALL * 1.2, BACK_H, centered=(True, True, False))
)

slot_block_h = 18.0
slot_y = BASE_D / 2 - WALL - SLOT_D / 2 - 6
slot_block = (
    cq.Workplane("XY")
    .workplane(offset=BASE_H)
    .center(0, slot_y)
    .box(SLOT_W + 2 * WALL, SLOT_D + 2 * WALL, slot_block_h, centered=(True, True, False))
)
slot_cut = (
    cq.Workplane("XY")
    .workplane(offset=BASE_H + 2)
    .center(0, slot_y)
    .box(SLOT_W, SLOT_D, slot_block_h, centered=(True, True, False))
)
slot_block = slot_block.cut(slot_cut)

result = base.union(back).union(slot_block)
result = result.edges("|Z").fillet(1.2)
'''


# ─── Household templates ───────────────────────────────────────────────────

def mug_code(sizes: dict, prompt: str = "") -> str:
    """Coffee mug: shelled cylinder + toroidal-ish handle ring."""
    h = float(sizes.get("height_mm") or 85.0)
    dia = float(sizes.get("diameter_mm") or 80.0)
    wall = float(sizes.get("wall_mm") or 3.0)
    wall = max(2.0, min(wall, 5.0))
    h = max(50.0, min(h, 140.0))
    dia = max(50.0, min(dia, 110.0))
    r = dia / 2.0
    handle_r_outer = 14.0
    handle_r_inner = 8.0
    handle_thick = 6.0
    handle_cx = r + handle_r_outer * 0.55
    handle_cz = h * 0.48

    return f'''import cadquery as cq

# Parametric coffee mug — template (no LLM)
# Height {h} mm, outer diameter {dia} mm, wall {wall} mm

H, R, WALL = {h}, {r}, {wall}
HO, HI, HT = {handle_r_outer}, {handle_r_inner}, {handle_thick}
HCX, HCZ = {handle_cx:.3f}, {handle_cz:.3f}

body = (
    cq.Workplane("XY")
    .circle(R)
    .extrude(H)
    .faces(">Z")
    .shell(-WALL)
)
handle = (
    cq.Workplane("XZ")
    .center(HCX, HCZ)
    .circle(HO)
    .extrude(HT, both=True)
    .cut(
        cq.Workplane("XZ").center(HCX, HCZ).circle(HI).extrude(HT + 2, both=True)
    )
)
result = body.union(handle)
'''


def pen_holder_code(sizes: dict, prompt: str = "") -> str:
    """Hex (or round) desk cup / pen holder."""
    p = prompt.lower()
    h = float(sizes.get("height_mm") or 100.0)
    dia = float(sizes.get("diameter_mm") or 80.0)
    wall = float(sizes.get("wall_mm") or 3.0)
    h = max(40.0, min(h, 160.0))
    dia = max(40.0, min(dia, 120.0))
    wall = max(1.6, min(wall, 5.0))
    sides = 6 if re.search(r"hex", p) or "hex" in p else 0
    # If "hexagonal outside" / default pen holder → hex
    if re.search(r"pen|pencil|desk", p) and sides == 0:
        sides = 6

    if sides >= 3:
        return f'''import cadquery as cq

# Hex desk pen holder — template
# Outer diameter (vertex-to-vertex) {dia} mm, height {h} mm, wall {wall} mm

SIDES, DIA, H, WALL = {sides}, {dia}, {h}, {wall}

outer = cq.Workplane("XY").polygon(SIDES, DIA).extrude(H)
result = outer.faces(">Z").shell(-WALL)
'''
    return f'''import cadquery as cq

# Round desk cup / pen holder — template
R, H, WALL = {dia/2}, {h}, {wall}
body = cq.Workplane("XY").circle(R).extrude(H)
result = body.faces(">Z").shell(-WALL)
'''


def project_box_code(sizes: dict, prompt: str = "") -> str:
    """Simple electronics project box: hollow base + optional lip for lid."""
    w = float(sizes.get("width_mm") or 100.0)
    d = float(sizes.get("depth_mm") or 60.0)
    h = float(sizes.get("height_mm") or 40.0)
    wall = float(sizes.get("wall_mm") or 2.0)
    w, d, h = max(30.0, min(w, 250.0)), max(20.0, min(d, 200.0)), max(15.0, min(h, 120.0))
    wall = max(1.4, min(wall, 4.0))
    lip = 1.5
    # Split: base is 70% of height, lid is rest (we only emit base+rim as one printable
    # "open box" — lid is a second solid stacked as union offset for single-file library)
    base_h = round(h * 0.72, 2)
    lid_h = round(h - base_h, 2)

    return f'''import cadquery as cq

# Project / electronics box with snap-lip lid — template
# Outer {w} x {d} x {h} mm, wall {wall} mm

W, D, BASE_H, LID_H, WALL, LIP = {w}, {d}, {base_h}, {lid_h}, {wall}, {lip}

# Open base (sits on Z=0)
outer = cq.Workplane("XY").box(W, D, BASE_H, centered=(True, True, False))
inner = (
    cq.Workplane("XY")
    .workplane(offset=WALL)
    .box(W - 2 * WALL, D - 2 * WALL, BASE_H, centered=(True, True, False))
)
base = outer.cut(inner)
# Inner lip shelf for lid
shelf = (
    cq.Workplane("XY")
    .workplane(offset=BASE_H - LIP)
    .box(W - 2 * WALL + 0.6, D - 2 * WALL + 0.6, LIP + 0.2, centered=(True, True, False))
)
base = base.cut(shelf)

# Lid printed beside base (+X)
lid_outer = (
    cq.Workplane("XY")
    .center(W + 8, 0)
    .box(W, D, LID_H, centered=(True, True, False))
)
# Lid plug that seats into the base lip
plug = (
    cq.Workplane("XY")
    .workplane(offset=LID_H)
    .center(W + 8, 0)
    .box(W - 2 * WALL - 0.4, D - 2 * WALL - 0.4, LIP, centered=(True, True, False))
)
lid = lid_outer.union(plug)

result = base.union(lid)
try:
    result = result.edges("|Z").fillet(1.0)
except Exception:
    pass
'''


def cable_clip_code(sizes: dict, prompt: str = "") -> str:
    """Desk cable management clip with adhesive pad and C-channel."""
    w = float(sizes.get("width_mm") or 18.0)
    d = float(sizes.get("depth_mm") or 22.0)
    h = float(sizes.get("height_mm") or 14.0)
    channel = float(sizes.get("diameter_mm") or 6.5)
    w = max(12.0, min(w, 40.0))
    d = max(14.0, min(d, 40.0))
    h = max(10.0, min(h, 25.0))
    channel = max(3.5, min(channel, 12.0))

    return f'''import cadquery as cq

# Desk cable clip — template
# Footprint {w}x{d} mm, height {h} mm, cable channel ~{channel} mm

W, D, H, CH = {w}, {d}, {h}, {channel}

body = cq.Workplane("XY").box(W, D, H, centered=(True, True, False))
# Open C-channel from +Y face
cut = (
    cq.Workplane("XY")
    .workplane(offset=H * 0.35)
    .center(0, D * 0.15)
    .box(W + 1, D * 0.7, CH, centered=(True, True, False))
)
# Mouth opening on top so cable snaps in
mouth = (
    cq.Workplane("XY")
    .workplane(offset=H - 0.5)
    .center(0, D * 0.22)
    .box(CH * 0.85, D * 0.55, 2.0, centered=(True, True, False))
)
result = body.cut(cut).cut(mouth)
try:
    result = result.edges("|Z").fillet(0.8)
except Exception:
    pass
'''


def wall_hook_code(sizes: dict, prompt: str = "") -> str:
    """Simple wall hook with screw holes."""
    w = float(sizes.get("width_mm") or 28.0)
    back_h = float(sizes.get("height_mm") or 50.0)
    depth = float(sizes.get("depth_mm") or 35.0)
    w = max(18.0, min(w, 60.0))
    back_h = max(30.0, min(back_h, 100.0))
    depth = max(20.0, min(depth, 70.0))

    return f'''import cadquery as cq

# Wall hook with two screw holes — template
W, BACK_H, DEPTH = {w}, {back_h}, {depth}
THICK = 4.0

back = cq.Workplane("XY").box(W, THICK, BACK_H, centered=(True, False, False))
# Screw holes
holes = (
    cq.Workplane("XZ")
    .workplane(offset=THICK + 0.1)
    .pushPoints([(0, BACK_H * 0.75), (0, BACK_H * 0.35)])
    .circle(2.2)
    .extrude(-THICK - 0.5)
)
back = back.cut(holes)
# Hook arm
arm = (
    cq.Workplane("XY")
    .workplane(offset=8)
    .center(0, THICK)
    .box(W * 0.7, DEPTH - THICK, 5.0, centered=(True, False, False))
)
# Upturn tip
tip = (
    cq.Workplane("XY")
    .workplane(offset=8)
    .center(0, DEPTH - 3)
    .box(W * 0.7, 4.0, 12.0, centered=(True, False, False))
)
result = back.union(arm).union(tip)
try:
    result = result.edges("|Z").fillet(1.0)
except Exception:
    pass
'''


def nameplate_code(sizes: dict, prompt: str = "") -> str:
    """Flat name tag / desk plaque base with rounded corners."""
    w = float(sizes.get("width_mm") or 60.0)
    d = float(sizes.get("depth_mm") or 25.0)
    h = float(sizes.get("height_mm") or 3.0)
    w, d, h = max(30.0, min(w, 150.0)), max(12.0, min(d, 60.0)), max(2.0, min(h, 8.0))
    fillet = min(4.0, w / 8, d / 4)

    return f'''import cadquery as cq

# Nameplate / badge base — template
W, D, H, F = {w}, {d}, {h}, {fillet}
result = cq.Workplane("XY").box(W, D, H).edges("|Z").fillet(F)
'''


def hex_nut_code(sizes: dict, prompt: str = "") -> str:
    """Hex nut from across-flats + bore."""
    import math
    af = float(sizes.get("across_flats_mm") or sizes.get("width_mm") or 24.0)
    h = float(sizes.get("height_mm") or max(8.0, af * 0.45))
    bore = float(sizes.get("bore_mm") or sizes.get("diameter_mm") or af * 0.45)
    af = max(8.0, min(af, 60.0))
    h = max(4.0, min(h, 30.0))
    bore = max(2.0, min(bore, af * 0.85))
    # polygon diameter = vertex-to-vertex = AF / cos(30°)
    dia = af / math.cos(math.pi / 6)

    return f'''import cadquery as cq
import math

# Hex nut — template
# {af} mm across flats, {h} mm tall, {bore} mm bore
AF, H, BORE = {af}, {h}, {bore}
DIA = AF / math.cos(math.pi / 6)  # vertex-to-vertex for cq.polygon diameter arg
result = (
    cq.Workplane("XY")
    .polygon(6, DIA)
    .extrude(H)
    .faces(">Z")
    .hole(BORE)
)
'''


def desk_tray_code(sizes: dict, prompt: str = "") -> str:
    """Shallow desk tray, optionally 2 compartments. Credit-card sized when asked."""
    p = prompt.lower()
    if re.search(r"credit\s+card|business\s+card|wallet", p):
        # Card is flat: footprint ≈ card W×H, tray depth is small (not card thickness as height)
        # ISO ID-1: 85.6 × 53.98 × 0.76 — tray walls around that pocket
        w = float(sizes.get("width_mm") or 85.6) + 8.0
        d = float(sizes.get("height_mm") or 54.0) + 8.0  # long face of card on tray floor
        if d < 40:
            d = 54.0 + 8.0
        h = 12.0  # shallow tray — ignore catalog card thickness as "height"
    else:
        w = float(sizes.get("width_mm") or 120.0)
        d = float(sizes.get("depth_mm") or 80.0)
        h = float(sizes.get("height_mm") or 18.0)
    wall = float(sizes.get("wall_mm") or 2.0)
    w, d, h = max(50.0, min(w, 250.0)), max(40.0, min(d, 200.0)), max(10.0, min(h, 40.0))
    wall = max(1.4, min(wall, 4.0))
    two = bool(re.search(r"2|two|dual|compartments?", prompt, re.I))

    divider = ""
    if two:
        divider = f'''
div = (
    cq.Workplane("XY")
    .box(WALL, D - 2 * WALL, H - WALL, centered=(True, True, False))
    .translate((0, 0, WALL))
)
tray = tray.union(div)
'''

    return f'''import cadquery as cq

# Desk tray{" with 2 compartments" if two else ""} — template
W, D, H, WALL = {w}, {d}, {h}, {wall}

outer = cq.Workplane("XY").box(W, D, H, centered=(True, True, False))
inner = (
    cq.Workplane("XY")
    .workplane(offset=WALL)
    .box(W - 2 * WALL, D - 2 * WALL, H, centered=(True, True, False))
)
tray = outer.cut(inner)
{divider}
result = tray
try:
    result = result.edges("|Z").fillet(2.0)
except Exception:
    pass
'''


def washer_code(sizes: dict, prompt: str = "") -> str:
    """Flat washer — OD/ID/thickness from metric table or prompt."""
    od = float(sizes.get("diameter_mm") or 16.0)
    id_ = float(sizes.get("bore_mm") or od * 0.52)
    t = float(sizes.get("depth_mm") or sizes.get("height_mm") or 1.6)
    od = max(6.0, min(od, 80.0))
    id_ = max(2.0, min(id_, od - 1.5))
    t = max(0.5, min(t, 8.0))
    m = sizes.get("metric_m")
    label = f"M{int(m)} " if m else ""

    return f'''import cadquery as cq

# {label}washer — template  OD={od} ID={id_} T={t} mm
OD, ID, T = {od}, {id_}, {t}
result = (
    cq.Workplane("XY")
    .circle(OD / 2)
    .circle(ID / 2)
    .extrude(T)
)
'''


def spacer_code(sizes: dict, prompt: str = "") -> str:
    od = float(sizes.get("diameter_mm") or 10.0)
    id_ = float(sizes.get("bore_mm") or od * 0.55)
    h = float(sizes.get("height_mm") or 10.0)
    od = max(5.0, min(od, 40.0))
    id_ = max(2.0, min(id_, od - 1.2))
    h = max(3.0, min(h, 60.0))
    return f'''import cadquery as cq

# Round spacer / standoff — template
OD, ID, H = {od}, {id_}, {h}
result = cq.Workplane("XY").circle(OD/2).circle(ID/2).extrude(H)
'''


def soap_dish_code(sizes: dict, prompt: str = "") -> str:
    """Soap dish with drain slots — cavity sized to a bar of soap, shallow walls."""
    # Catalog "soap" is the BAR (≈90×55×30). Dish must be LARGER footprint, SHALLOW height.
    bar_w = float(sizes.get("width_mm") or 90.0)
    bar_d = float(sizes.get("depth_mm") or 55.0)
    wall = float(sizes.get("wall_mm") or 2.2)
    # Outer dish: bar + margin; height is dish wall, not bar thickness
    w = max(80.0, min(bar_w + 16.0, 160.0))
    d = max(60.0, min(bar_d + 16.0, 110.0))
    h = 16.0  # shallow tray — never use bar thickness as dish height
    wall = max(1.6, min(wall, 3.5))
    # Drain slot spacing from dish width
    span = (w - 2 * wall - 16) / 2
    xs = [-span * 0.66, -span * 0.22, span * 0.22, span * 0.66]

    return f'''import cadquery as cq

# Soap dish with drain slots — template
# Outer {w:.1f}×{d:.1f}×{h:.1f} mm; cavity fits ~{bar_w:.0f}×{bar_d:.0f} mm bar
W, D, H, WALL = {w}, {d}, {h}, {wall}
XS = {xs!r}

outer = cq.Workplane("XY").box(W, D, H, centered=(True, True, False))
inner = (
    cq.Workplane("XY")
    .workplane(offset=WALL)
    .box(W - 2 * WALL, D - 2 * WALL, H, centered=(True, True, False))
)
dish = outer.cut(inner)
for x in XS:
    slot = (
        cq.Workplane("XY")
        .center(x, 0)
        .box(4.0, D - 2 * WALL - 8, WALL + 1, centered=(True, True, False))
    )
    dish = dish.cut(slot)
result = dish
'''


def can_holder_code(sizes: dict, prompt: str = "") -> str:
    """Soda/beer can sleeve — real 66 mm can + clearance."""
    # US 12 oz can ~66 mm OD; sleeve is partial height (not full can)
    dia = float(sizes.get("diameter_mm") or 66.0)
    # Catalog injects full can height (~122); holders should be ~half unless user set
    raw_h = sizes.get("height_mm")
    if raw_h and float(raw_h) > 90:
        h = float(raw_h) * 0.55  # partial sleeve
    else:
        h = float(raw_h or 70.0)
    wall = float(sizes.get("wall_mm") or 2.5)
    clr = 0.6
    inner = dia + 2 * clr
    outer = inner + 2 * wall
    h = max(40.0, min(h, 90.0))
    base = wall + 1.5

    return f'''import cadquery as cq

# Drink can holder / sleeve — template
# Can OD ~{dia} mm; cavity {inner} mm; outer {outer} mm; height {h} mm
INNER, OUTER, H, BASE = {inner}, {outer}, {h}, {base}

body = cq.Workplane("XY").circle(OUTER / 2).extrude(H)
cavity = (
    cq.Workplane("XY")
    .workplane(offset=BASE)
    .circle(INNER / 2)
    .extrude(H)
)
result = body.cut(cavity)
'''


def generic_phone_stand_code(
    sizes: dict, prompt: str = "", dims: Optional[dict] = None
) -> str:
    """Desk phone stand with REAL lean angle and lip — uses researched dims when available."""
    if dims and all(k in dims for k in ("height_mm", "width_mm", "depth_mm")):
        # Prefer researched body for slot width/depth; still use angled generic if angle given
        if not sizes.get("angle_deg"):
            return phone_stand_code(dims)
        pw = float(dims["width_mm"])
        pd = float(dims["depth_mm"])
    else:
        pw = float(sizes.get("width_mm") or 78.0)
        pd = float(sizes.get("depth_mm") or 10.0)

    angle = float(sizes.get("angle_deg") or 60.0)
    angle = max(45.0, min(angle, 75.0))  # from horizontal; 60° is common
    wall = 3.5
    slot_w = pw + 2.0
    slot_d = max(pd + 2.0, 10.0)
    base_d = 90.0
    base_w = slot_w + 2 * wall + 10
    base_h = 6.0
    back_len = 75.0  # length of support plate along lean
    # Lean: back plate rotated about X so its face is at `angle` from the bed
    # CadQuery rotate deg about +X: plate starts in XY extruded Z, rotate so it tilts back
    rot = -(90.0 - angle)

    return f'''import cadquery as cq

# Angled phone stand — template
# Lean {angle}° from horizontal, slot {slot_w:.1f}×{slot_d:.1f} mm (phone width×depth + clearance)
SLOT_W, SLOT_D = {slot_w}, {slot_d}
WALL = {wall}
BASE_W, BASE_D, BASE_H = {base_w}, {base_d}, {base_h}
BACK_LEN = {back_len}
ANGLE, ROT = {angle}, {rot}

base = cq.Workplane("XY").box(BASE_W, BASE_D, BASE_H, centered=(True, True, False))

# Support plate: built upright then rotated to ANGLE from bed, foot at rear of base
support = (
    cq.Workplane("XY")
    .box(BASE_W - 8, WALL, BACK_LEN, centered=(True, True, False))
    .rotate((0, 0, 0), (1, 0, 0), ROT)
    .translate((0, BASE_D * 0.12, BASE_H))
)

# Phone shelf / lip near front of support foot
# Slot pocket on the base so phone sits and leans against support
slot_y = -BASE_D * 0.08
shelf = (
    cq.Workplane("XY")
    .workplane(offset=BASE_H)
    .center(0, slot_y)
    .box(SLOT_W + 2 * WALL, SLOT_D + 2 * WALL + 6, 14, centered=(True, True, False))
)
pocket = (
    cq.Workplane("XY")
    .workplane(offset=BASE_H + 2)
    .center(0, slot_y + 1)
    .box(SLOT_W, SLOT_D + 2, 16, centered=(True, True, False))
)
# Front lip retains the phone bottom edge
lip = (
    cq.Workplane("XY")
    .workplane(offset=BASE_H)
    .center(0, slot_y - SLOT_D / 2 - WALL)
    .box(SLOT_W + 2 * WALL, 4.0, 12, centered=(True, True, False))
)
shelf = shelf.cut(pocket).union(lip)

result = base.union(support).union(shelf)
'''
