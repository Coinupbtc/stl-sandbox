"""Universal size intent — every mechanical part should be the right size.

Three sources of truth (highest wins when they conflict on the same axis):
  1. Explicit numbers in the prompt ("85 mm tall", "100x60x40", "M8", "2 mm thick")
  2. Product research (phones etc. from dim_research)
  3. Real-world object catalog (credit card, soda can, AA battery, soap bar, …)

Outputs a size contract the LLM/template must obey, plus bbox validation.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ─── Real-world reference sizes (mm). height = upright long axis when relevant. ───
# depth = thickness for slabs; diameter lives in diameter_mm when cylindrical.

OBJECT_CATALOG: dict[str, dict[str, Any]] = {
    # Cards / media
    "credit card": {
        "width_mm": 85.6, "height_mm": 53.98, "depth_mm": 0.76,
        "notes": "ISO/IEC 7810 ID-1", "source": "catalog:iso7810",
    },
    "business card": {
        "width_mm": 89.0, "height_mm": 51.0, "depth_mm": 0.3,
        "notes": "US standard ~3.5×2 in", "source": "catalog:us-business-card",
    },
    "sd card": {
        "width_mm": 24.0, "height_mm": 32.0, "depth_mm": 2.1,
        "source": "catalog:sd-association",
    },
    "microsd": {
        "width_mm": 11.0, "height_mm": 15.0, "depth_mm": 1.0,
        "source": "catalog:sd-association",
    },
    # Cylinders / cans / bottles
    "soda can": {
        "diameter_mm": 66.0, "height_mm": 122.0,
        "notes": "US 12 oz aluminum can body", "source": "catalog:us-12oz-can",
    },
    "beer can": {
        "diameter_mm": 66.0, "height_mm": 122.0,
        "source": "catalog:us-12oz-can",
    },
    "water bottle": {
        "diameter_mm": 66.0, "height_mm": 210.0,
        "notes": "typical 500 ml PET body (varies)", "source": "catalog:approx-500ml",
    },
    "coffee mug": {
        "diameter_mm": 80.0, "height_mm": 95.0, "wall_mm": 3.0,
        "source": "catalog:typical-mug",
    },
    "mug": {
        "diameter_mm": 80.0, "height_mm": 95.0, "wall_mm": 3.0,
        "source": "catalog:typical-mug",
    },
    # Batteries
    "aa battery": {
        "diameter_mm": 14.5, "height_mm": 50.5,
        "source": "catalog:iec-lr6",
    },
    "aaa battery": {
        "diameter_mm": 10.5, "height_mm": 44.5,
        "source": "catalog:iec-lr03",
    },
    "18650 battery": {
        "diameter_mm": 18.6, "height_mm": 65.2,
        "source": "catalog:18650",
    },
    # Personal / desk
    "soap bar": {
        "width_mm": 90.0, "depth_mm": 55.0, "height_mm": 30.0,
        "source": "catalog:typical-bar-soap",
    },
    "soap": {
        "width_mm": 90.0, "depth_mm": 55.0, "height_mm": 30.0,
        "source": "catalog:typical-bar-soap",
    },
    "smartphone": {
        "height_mm": 150.0, "width_mm": 72.0, "depth_mm": 8.0,
        "notes": "generic modern phone envelope", "source": "catalog:generic-phone",
    },
    "phone": {
        "height_mm": 150.0, "width_mm": 72.0, "depth_mm": 8.0,
        "source": "catalog:generic-phone",
    },
    "usb stick": {
        "width_mm": 45.0, "depth_mm": 18.0, "height_mm": 8.0,
        "source": "catalog:typical-usb-a",
    },
    "usb-c plug": {
        "width_mm": 8.25, "depth_mm": 2.4, "height_mm": 6.5,
        "source": "catalog:usb-c-receptacle-approx",
    },
    # Fasteners (nominal; bore/OD for washers filled by M-series table)
    "washer": {"kind": "washer"},
    "hex nut": {"kind": "hex_nut"},
    "spacer": {"kind": "spacer"},
    # Printables defaults
    "pen": {"diameter_mm": 9.0, "height_mm": 145.0, "source": "catalog:typical-pen"},
    "pencil": {"diameter_mm": 7.5, "height_mm": 175.0, "source": "catalog:typical-pencil"},
    "coaster": {"diameter_mm": 100.0, "height_mm": 6.0, "source": "catalog:typical-coaster"},
    "key": {"width_mm": 25.0, "depth_mm": 2.5, "height_mm": 60.0, "source": "catalog:typical-key"},
}

# ISO metric coarse approximate: (thread OD, across flats nut, nut height, washer OD, washer ID, washer thick)
_METRIC: dict[int, tuple[float, float, float, float, float, float]] = {
    3: (3.0, 5.5, 2.4, 7.0, 3.2, 0.5),
    4: (4.0, 7.0, 3.2, 9.0, 4.3, 0.8),
    5: (5.0, 8.0, 4.0, 10.0, 5.3, 1.0),
    6: (6.0, 10.0, 5.0, 12.0, 6.4, 1.6),
    8: (8.0, 13.0, 6.5, 16.0, 8.4, 1.6),
    10: (10.0, 17.0, 8.0, 20.0, 10.5, 2.0),
    12: (12.0, 19.0, 10.0, 24.0, 13.0, 2.5),
}


def parse_prompt_sizes(prompt: str) -> dict[str, float]:
    """Extract numeric size cues from freeform English. Units → mm."""
    p = prompt.lower()
    out: dict[str, float] = {}

    def grab(patterns: list[str], key: str):
        if key in out:
            return
        for pat in patterns:
            m = re.search(pat, p, re.I)
            if m:
                try:
                    out[key] = float(m.group(1))
                    return
                except (TypeError, ValueError):
                    pass

    # unit helpers: capture number then optional unit, convert inches
    def mm_val(num: str, unit: str | None = None) -> float:
        v = float(num)
        u = (unit or "mm").lower()
        if u in ("in", "inch", "inches", '"'):
            return v * 25.4
        if u == "cm":
            return v * 10.0
        return v

    # Generic: N mm/in tall|high|wide|deep|thick|diameter
    for key, words in (
        ("height_mm", r"tall|high|height"),
        ("width_mm", r"wide|width"),
        ("depth_mm", r"deep|depth|thick(?:ness)?"),
        ("diameter_mm", r"diameter|dia\.?|ø|od\b"),
    ):
        m = re.search(
            rf"(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches|\")?\s*(?:{words})",
            p,
        )
        if m and key not in out:
            out[key] = mm_val(m.group(1), m.group(2))
        m2 = re.search(
            rf"(?:{words})\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches|\")?",
            p,
        )
        if m2 and key not in out:
            out[key] = mm_val(m2.group(1), m2.group(2))

    # "about 80mm diameter"
    m = re.search(
        r"(?:about|approx(?:imately)?|~)?\s*(\d+(?:\.\d+)?)\s*(mm|cm)?\s*(?:diameter|dia)",
        p,
    )
    if m and "diameter_mm" not in out:
        out["diameter_mm"] = mm_val(m.group(1), m.group(2))

    # walls
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*walls?", p)
    if m:
        out["wall_mm"] = float(m.group(1))
    m = re.search(r"walls?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", p)
    if m:
        out["wall_mm"] = float(m.group(1))

    # angle
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|deg(?:ree)?s?)", p)
    if m:
        out["angle_deg"] = float(m.group(1))

    # across flats
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*across\s+flats|across\s+flats\s*[=:]?\s*(\d+(?:\.\d+)?)", p)
    if m:
        out["across_flats_mm"] = float(m.group(1) or m.group(2))

    # hole / bore
    m = re.search(
        r"(?:hole|bore|id)\s*(?:of\s*|through\s*)?(?:about\s*)?(\d+(?:\.\d+)?)\s*mm"
        r"|(\d+(?:\.\d+)?)\s*mm\s*(?:hole|bore|id)\b",
        p,
    )
    if m:
        out["bore_mm"] = float(m.group(1) or m.group(2))

    # Metric fasteners: M8, M10
    m = re.search(r"\bm\s*(\d{1,2})\b", p)
    if m:
        out["metric_m"] = float(m.group(1))
        if "bore_mm" not in out and re.search(r"\b(nut|washer|bolt|screw|hole|bore)\b", p):
            out["bore_mm"] = float(m.group(1))  # nominal; clearance added later for holes

    # Cube / sphere equal sides: "40mm cube", "cube 40"
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*cube|cube\s*(?:of\s*)?(\d+(?:\.\d+)?)", p)
    if m:
        s = float(m.group(1) or m.group(2))
        out["width_mm"] = out["depth_mm"] = out["height_mm"] = s

    # Box triple W×D×H or L×W×H
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|in)?",
        p,
    )
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        unit = m.group(4)
        if unit:
            a, b, c = mm_val(str(a), unit), mm_val(str(b), unit), mm_val(str(c), unit)
        out.setdefault("width_mm", a)
        out.setdefault("depth_mm", b)
        out.setdefault("height_mm", c)

    # Pair: 120x80 (tray footprint)
    if "width_mm" not in out:
        m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm)?", p)
        if m and not re.search(r"[x×]\s*\d+[x×]", p):  # not a triple already
            out["width_mm"] = float(m.group(1))
            out["depth_mm"] = float(m.group(2))

    # "N mm" alone near object words when only one size given
    if len(out) == 0:
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm\b", p)
        if m:
            out["height_mm"] = float(m.group(1))

    return out


def _catalog_hit(prompt: str) -> Optional[dict[str, Any]]:
    p = prompt.lower()
    # longest key first
    for key in sorted(OBJECT_CATALOG.keys(), key=len, reverse=True):
        # word-boundary-ish
        if re.search(rf"\b{re.escape(key)}\b", p):
            return {"product": key, **OBJECT_CATALOG[key]}
    # fuzzy aliases
    aliases = {
        r"\b(coke|pepsi|soft.?drink)\s*can\b": "soda can",
        r"\bbar\s+of\s+soap\b|\bsoap\s+dish\b": "soap",
        r"\bcell\s*phone\b|\bmobile\s*phone\b": "phone",
        r"\bid.?1\b|\bcr80\b": "credit card",
        r"\bcoaster\b": "coaster",
    }
    for pat, key in aliases.items():
        if re.search(pat, p):
            return {"product": key, **OBJECT_CATALOG[key]}
    return None


def _apply_metric(prompt: str, sizes: dict) -> dict:
    """Fill washer/nut geometry from M-series if present."""
    p = prompt.lower()
    m = int(sizes.get("metric_m") or 0)
    if not m and re.search(r"\bm\s*(\d{1,2})\b", p):
        m = int(re.search(r"\bm\s*(\d{1,2})\b", p).group(1))
    if m not in _METRIC:
        return sizes
    od, af, nh, wod, wid, wt = _METRIC[m]
    sizes = dict(sizes)
    sizes["metric_m"] = float(m)
    if re.search(r"\bwasher\b", p):
        sizes.setdefault("diameter_mm", wod)
        sizes.setdefault("bore_mm", wid)
        sizes.setdefault("depth_mm", sizes.get("depth_mm") or wt)
        sizes["product"] = f"M{m} washer"
        sizes["source"] = "catalog:iso-metric"
    elif re.search(r"\b(hex\s*)?nut\b", p):
        sizes.setdefault("across_flats_mm", af)
        sizes.setdefault("height_mm", nh)
        sizes.setdefault("bore_mm", od)
        sizes["product"] = f"M{m} hex nut"
        sizes["source"] = "catalog:iso-metric"
    elif re.search(r"\b(spacer|standoff)\b", p):
        sizes.setdefault("diameter_mm", wod)
        sizes.setdefault("bore_mm", wid)
        sizes.setdefault("height_mm", sizes.get("height_mm") or 10.0)
        sizes["product"] = f"M{m} spacer"
        sizes["source"] = "catalog:iso-metric"
    return sizes


def resolve_size_intent(
    prompt: str,
    researched: Optional[dict] = None,
) -> Optional[dict[str, Any]]:
    """Build a size contract for this prompt.

    Returns None only when we truly have no size signal (LLM must invent carefully).
    """
    explicit = parse_prompt_sizes(prompt)
    catalog = _catalog_hit(prompt)
    explicit = _apply_metric(prompt, explicit)

    contract: dict[str, Any] = {
        "query": prompt,
        "explicit": dict(explicit),
        "sources": [],
    }

    # Base from catalog
    if catalog:
        for k, v in catalog.items():
            if k in ("kind",):
                continue
            if k.endswith("_mm") or k == "wall_mm":
                contract.setdefault(k, v)
            elif k in ("product", "notes", "source"):
                contract[k] = v
        contract["sources"].append(catalog.get("source") or "catalog")
        contract["confidence"] = "medium"

    # Researched product dims (phones etc.) override catalog body sizes
    if researched:
        for k in (
            "height_mm", "width_mm", "depth_mm", "diameter_mm",
            "corner_radius_mm", "wall_mm", "clearance_mm", "lip_mm",
            "product", "source", "part_type", "outer_height_mm",
            "outer_width_mm", "outer_depth_mm", "expected_bbox_mm",
            "cavity_height_mm", "cavity_width_mm", "cavity_depth_mm",
            "confidence", "notes",
        ):
            if researched.get(k) is not None:
                contract[k] = researched[k]
        contract["sources"].append(researched.get("source") or "research")
        contract["confidence"] = researched.get("confidence") or "high"

    # Explicit prompt numbers always win for those axes
    for k, v in explicit.items():
        if k == "metric_m":
            contract[k] = v
            continue
        contract[k] = v
    if explicit:
        contract["sources"].append("prompt")
        contract["confidence"] = "high"

    # Derived expected outer envelope
    contract["expected_bbox_mm"] = _expected_bbox(prompt, contract)

    has_any = any(
        contract.get(k) is not None
        for k in (
            "height_mm", "width_mm", "depth_mm", "diameter_mm",
            "across_flats_mm", "bore_mm", "expected_bbox_mm",
        )
    )
    if not has_any:
        return None

    if "product" not in contract:
        contract["product"] = (catalog or {}).get("product") or "user-described object"
    if "source" not in contract:
        contract["source"] = "+".join(contract["sources"]) or "inferred"

    return contract


def _expected_bbox(prompt: str, c: dict) -> Optional[list[float]]:
    """Best-effort outer envelope [dx,dy,dz] unordered for comparison."""
    if c.get("expected_bbox_mm"):
        return list(c["expected_bbox_mm"])

    p = prompt.lower()
    h = c.get("height_mm")
    w = c.get("width_mm")
    d = c.get("depth_mm")
    dia = c.get("diameter_mm")
    af = c.get("across_flats_mm")
    wall = float(c.get("wall_mm") or 2.0)

    # Phone case already has outer_* from research
    if c.get("outer_width_mm") and c.get("outer_height_mm") and c.get("outer_depth_mm"):
        return [
            float(c["outer_width_mm"]),
            float(c["outer_height_mm"]),
            float(c["outer_depth_mm"]),
        ]

    # Phone / device stand or dock — envelope is the STAND, not the phone body stack
    if re.search(r"\b(stand|dock|cradle)\b", p) and not re.search(
        r"\b(can|mug|pen|soap|bottle)\b", p
    ):
        # typical desk stand: base depth ~70–90, width ≈ phone width + walls, height ~60–90
        sw = float(w or 78.0) + 2 * wall + 8.0
        sd = 85.0
        sh = min(float(h or 150.0) * 0.5, 90.0)
        return [sw, sd, sh]

    # Holder for cylindrical object (can, bottle, mug body)
    if re.search(r"\b(holder|sleeve)\b", p) and dia:
        od = float(dia) + 2 * wall + 2.0
        oh = float(h or dia)
        if oh > 90:
            oh = oh * 0.55  # partial sleeve
        return [od, od, max(oh, 40.0)]

    if re.search(r"\b(holder|dish|tray)\b", p) and w and d:
        # tray/dish around object: outer slightly larger, modest height
        return [float(w) + 2 * wall + 4.0, float(d) + 2 * wall + 4.0, float(h or 18.0) if (h or 0) < 50 else 18.0]

    if af:
        import math
        # hex vertex span
        dia_v = float(af) / math.cos(math.pi / 6)
        return [dia_v, float(af), float(h or af * 0.45)]

    if dia and h:
        return [float(dia), float(dia), float(h)]
    if dia and d:  # flat disc / washer
        return [float(dia), float(dia), float(d)]
    if dia:
        return [float(dia), float(dia), float(dia)]

    if w and d and h:
        return [float(w), float(d), float(h)]
    if w and d:
        return [float(w), float(d), float(h or 10.0)]
    if w and h:
        return [float(w), float(d or w * 0.6), float(h)]
    if h and not w and not d and not dia:
        # single height: loose check later
        return None

    dims = [x for x in (w, d, h, dia) if x]
    if len(dims) >= 2:
        return [float(x) for x in dims[:3]]
    return None


def format_size_brief(contract: dict) -> str:
    """Inject into LLM user message — mandatory size law."""
    lines = [
        "=== SIZE CONTRACT (mm) — YOU MUST OBEY ===",
        f"Object: {contract.get('product', '?')}",
        f"Source: {contract.get('source', '?')} "
        f"(confidence: {contract.get('confidence', '?')}; "
        f"channels: {', '.join(contract.get('sources') or [])})",
    ]
    for label, key in (
        ("Height / tall axis", "height_mm"),
        ("Width", "width_mm"),
        ("Depth / thickness", "depth_mm"),
        ("Diameter (OD)", "diameter_mm"),
        ("Across flats", "across_flats_mm"),
        ("Bore / hole ID", "bore_mm"),
        ("Wall thickness", "wall_mm"),
        ("Angle (deg)", "angle_deg"),
        ("Metric size", "metric_m"),
    ):
        if contract.get(key) is not None:
            lines.append(f"  {label}: {contract[key]}")
    if contract.get("expected_bbox_mm"):
        e = contract["expected_bbox_mm"]
        lines.append(
            f"  Expected outer bounding box (axes may be permuted): "
            f"~{e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm"
        )
    if contract.get("outer_width_mm"):
        lines.append(
            f"  Case outer envelope: {contract['outer_width_mm']} × "
            f"{contract.get('outer_height_mm')} × {contract.get('outer_depth_mm')} mm (W×H×D)"
        )
    if contract.get("notes"):
        lines.append(f"Notes: {contract['notes']}")
    lines += [
        "",
        "Rules:",
        "  - Use these numbers as named constants at the top of your script.",
        "  - Do NOT invent a different real-world size for a known object.",
        "  - If this is a holder/case for the object, size the cavity to the object",
        "    (+0.3–0.5 mm clearance per side) and walls 1.5–3 mm.",
        "  - Prefer printable scale: if the true object is huge (>200 mm), still use",
        "    the real size unless the user asked for a miniature/toy.",
        "=== END SIZE CONTRACT ===",
    ]
    return "\n".join(lines)


def dims_match_bbox(
    contract: dict,
    bbox_mm: list,
    tol_frac: float = 0.20,
) -> tuple[bool, str]:
    """Compare sorted bbox to expected envelope.

    Special cases:
    - single known height: max(bbox) must be near height (figurines/mugs)
    - diameter-only disc: two axes ≈ diameter, one ≈ thickness
    """
    if not bbox_mm or len(bbox_mm) < 3:
        return True, "no bbox"

    got = sorted(float(x) for x in bbox_mm[:3])
    expected = contract.get("expected_bbox_mm")

    if expected:
        exp = sorted(float(x) for x in expected)
        # Allow project-box style "parts side by side" — if one axis is ~2× expected, check loosely
        for e, g in zip(exp, got):
            if e <= 0:
                continue
            err = abs(g - e) / e
            if err > tol_frac:
                # side-by-side assemblies: largest axis up to 2.3× expected largest
                if g > exp[-1] * 1.05 and g <= exp[-1] * 2.4 and e == exp[-1]:
                    continue
                if g > e * (1 + tol_frac) and any(
                    abs(g - 2 * ee) / max(2 * ee, 0.01) < tol_frac for ee in exp
                ):
                    continue  # lid printed beside base
                return (
                    False,
                    f"bbox {got} mm far from expected ~{exp} mm "
                    f"(>{tol_frac:.0%} off; size contract violated)",
                )
        return True, "bbox within size contract"

    # Height-only contract (common for mugs)
    h = contract.get("height_mm")
    dia = contract.get("diameter_mm")
    if h and dia:
        # one axis ~h, two ~dia (or one dia + handle stick-out)
        h_ok = any(abs(g - float(h)) / float(h) < tol_frac for g in got)
        # at least one axis near diameter
        d_ok = any(abs(g - float(dia)) / float(dia) < tol_frac for g in got) or any(
            g >= float(dia) * 0.85 for g in got
        )
        if h_ok and d_ok:
            return True, "height+diameter contract ok"
        return False, f"bbox {got} does not match H={h} dia={dia}"

    if h and not dia and not contract.get("width_mm"):
        # longest axis should be close to height OR height present in axes
        if any(abs(g - float(h)) / float(h) < tol_frac for g in got):
            return True, "height present in bbox"
        # allow figurine scale where height is max
        if abs(got[-1] - float(h)) / float(h) < tol_frac:
            return True, "max axis matches height"
        return False, f"bbox {got} missing height {h} mm"

    return True, "no strict envelope (advisory only)"


def quality_should_retry(stats: dict, warnings: list, prompt: str) -> Optional[str]:
    """Return a retry reason if the solid is not good enough to accept."""
    if not stats:
        return "missing stats"
    if stats.get("watertight") is False:
        return "mesh not watertight after export/repair"
    faces = int(stats.get("faces") or 0)
    bbox = stats.get("bbox_mm") or [0, 0, 0]
    mx = max(bbox) if bbox else 0
    mn = min(bbox) if bbox else 0
    warn_txt = " ".join(warnings or [])

    # Case/enclosure solid brick — check before generic face-count (old bad cases were 64 faces)
    if re.search(r"\b(case|casing|enclosure|shell|cover)\b", prompt, re.I):
        if "solid brick" in warn_txt.lower() or "fills >95%" in warn_txt:
            return "case/enclosure looks like a solid brick, not a hollow shell"
        # very thick "case" envelope (old bug: 75 mm thick for a phone)
        if len(bbox) >= 3 and sorted(bbox)[0] > 40 and max(bbox) > 100:
            return "case thickness implausible — likely a solid brick, not a shell"

    # Degenerate / oversimplified
    if faces < 12:
        return f"only {faces} faces"
    if faces < 80 and mx > 50 and not re.search(
        r"\b(cube|box|plate|washer|coin|spacer|disc|disk)\b", prompt, re.I
    ):
        return f"oversimplified geometry ({faces} faces on {mx:.0f} mm part)"
    if mn < 0.6 and not re.search(r"\b(gasket|shim|card|sheet|washer)\b", prompt, re.I):
        return f"too thin ({mn:.2f} mm) — not printable as a solid part"
    if "degenerate" in warn_txt.lower() or "under 2%" in warn_txt:
        return "degenerate fill ratio"
    if "not fully valid" in warn_txt.lower():
        return "OpenCascade solid invalid"
    return None
