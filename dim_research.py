"""Product dimension research for realistic CAD generation.

When a prompt names a real product (e.g. "iPhone 17 Pro Max case"), look up
height × width × depth in mm from:
  1. Local curated catalog (instant, high confidence)
  2. Wikipedia API / infobox parse
  3. GSMArena (phones)
  4. Generic web page scrape with mm triple extraction

Returns a structured brief the LLM (or a product template) must obey.
Never invents dimensions when research fails — callers fall back to heuristics.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional
from urllib.parse import quote, urlparse

import requests

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "dim_cache.json"
)
USER_AGENT = (
    "STL-Sandbox/1.0 (+local 3D-print helper; product dimension lookup only)"
)
REQUEST_TIMEOUT = float(os.environ.get("STL_DIM_TIMEOUT", "10"))
CACHE_TTL_S = int(os.environ.get("STL_DIM_CACHE_TTL", str(14 * 24 * 3600)))

# ─── Curated catalog (official / widely cited mm) ───
# Keys are normalized: lowercase, no punctuation, collapsed spaces.
# height = long axis, width = short axis, depth = thickness (phone convention).

CATALOG: dict[str, dict[str, Any]] = {
    # iPhone 17 series (Apple specs 2025)
    "iphone 17 pro max": {
        "product": "iPhone 17 Pro Max",
        "height_mm": 163.4,
        "width_mm": 78.0,
        "depth_mm": 8.75,
        "corner_radius_mm": 11.5,
        "weight_g": 233,
        "source": "catalog:apple-specs",
        "notes": "Camera island top-left when phone face-up, Lightning→USB-C bottom.",
    },
    "iphone 17 pro": {
        "product": "iPhone 17 Pro",
        "height_mm": 150.0,
        "width_mm": 71.9,
        "depth_mm": 8.75,
        "corner_radius_mm": 11.0,
        "weight_g": 206,
        "source": "catalog:apple-specs",
    },
    "iphone 17": {
        "product": "iPhone 17",
        "height_mm": 149.6,
        "width_mm": 71.5,
        "depth_mm": 7.95,
        "corner_radius_mm": 11.0,
        "source": "catalog:approx-gsmarena",
    },
    "iphone 16 pro max": {
        "product": "iPhone 16 Pro Max",
        "height_mm": 163.0,
        "width_mm": 77.6,
        "depth_mm": 8.25,
        "corner_radius_mm": 11.5,
        "source": "catalog:apple-specs",
    },
    "iphone 16 pro": {
        "product": "iPhone 16 Pro",
        "height_mm": 149.6,
        "width_mm": 71.5,
        "depth_mm": 8.25,
        "corner_radius_mm": 11.0,
        "source": "catalog:apple-specs",
    },
    "iphone 16": {
        "product": "iPhone 16",
        "height_mm": 147.6,
        "width_mm": 71.6,
        "depth_mm": 7.80,
        "corner_radius_mm": 11.0,
        "source": "catalog:apple-specs",
    },
    "iphone 15 pro max": {
        "product": "iPhone 15 Pro Max",
        "height_mm": 159.9,
        "width_mm": 76.7,
        "depth_mm": 8.25,
        "corner_radius_mm": 11.5,
        "source": "catalog:apple-specs",
    },
    "iphone 15 pro": {
        "product": "iPhone 15 Pro",
        "height_mm": 146.6,
        "width_mm": 70.6,
        "depth_mm": 8.25,
        "corner_radius_mm": 11.0,
        "source": "catalog:apple-specs",
    },
    "iphone 14 pro max": {
        "product": "iPhone 14 Pro Max",
        "height_mm": 160.7,
        "width_mm": 77.6,
        "depth_mm": 7.85,
        "corner_radius_mm": 11.0,
        "source": "catalog:apple-specs",
    },
    "iphone 13 pro max": {
        "product": "iPhone 13 Pro Max",
        "height_mm": 160.8,
        "width_mm": 78.1,
        "depth_mm": 7.65,
        "corner_radius_mm": 11.0,
        "source": "catalog:apple-specs",
    },
    # Samsung
    "samsung galaxy s25 ultra": {
        "product": "Samsung Galaxy S25 Ultra",
        "height_mm": 162.8,
        "width_mm": 77.6,
        "depth_mm": 8.2,
        "corner_radius_mm": 10.0,
        "source": "catalog:gsmarena",
    },
    "galaxy s25 ultra": {
        "product": "Samsung Galaxy S25 Ultra",
        "height_mm": 162.8,
        "width_mm": 77.6,
        "depth_mm": 8.2,
        "corner_radius_mm": 10.0,
        "source": "catalog:gsmarena",
    },
    "samsung galaxy s24 ultra": {
        "product": "Samsung Galaxy S24 Ultra",
        "height_mm": 162.3,
        "width_mm": 79.0,
        "depth_mm": 8.6,
        "corner_radius_mm": 10.0,
        "source": "catalog:gsmarena",
    },
    # Google
    "google pixel 9 pro xl": {
        "product": "Google Pixel 9 Pro XL",
        "height_mm": 162.8,
        "width_mm": 76.6,
        "depth_mm": 8.5,
        "corner_radius_mm": 10.5,
        "source": "catalog:gsmarena",
    },
    "pixel 9 pro xl": {
        "product": "Google Pixel 9 Pro XL",
        "height_mm": 162.8,
        "width_mm": 76.6,
        "depth_mm": 8.5,
        "corner_radius_mm": 10.5,
        "source": "catalog:gsmarena",
    },
    # Common household objects with known sizes (for holders / adapters)
    "airpods pro 2 charging case": {
        "product": "AirPods Pro 2 charging case",
        "height_mm": 45.2,
        "width_mm": 60.6,
        "depth_mm": 21.7,
        "source": "catalog:apple-specs",
    },
    "nintendo switch oled": {
        "product": "Nintendo Switch OLED (console only)",
        "height_mm": 102.0,
        "width_mm": 242.0,
        "depth_mm": 13.9,
        "source": "catalog:nintendo",
    },
    "ps5 dualsense controller": {
        "product": "PlayStation DualSense controller",
        "height_mm": 106.0,
        "width_mm": 160.0,
        "depth_mm": 66.0,
        "source": "catalog:sony-approx",
    },
}

# Prompt tokens that mean "this part must fit a named product"
_FIT_KEYWORDS = re.compile(
    r"\b("
    r"case|casing|cover|sleeve|shell|skin|enclosure|housing|holder|stand|"
    r"mount|dock|cradle|adapter|adaptor|bracket|clip|sleeve|protector|"
    r"bezel|tray|insert|jig|fixture|sleeve|holster|pouch|wallet case|"
    r"phone case|cell.?phone|protective"
    r")\b",
    re.I,
)

# Strip accessory words to recover the product name for lookup
_STRIP_FOR_PRODUCT = re.compile(
    r"\b("
    r"a|an|the|for|my|make|build|create|design|print|3d|printable|"
    r"case|casing|cover|sleeve|shell|skin|enclosure|housing|holder|stand|"
    r"mount|dock|cradle|adapter|adaptor|bracket|clip|protector|bezel|"
    r"tray|insert|jig|fixture|holster|pouch|wallet|protective|cell|phone|"
    r"cellular|simple|basic|sturdy|rugged|thin|slim|clear|matte|silicone|"
    r"tpu|hard|soft|custom|realistic|accurate|mm|millimeter|millimeters"
    r")\b",
    re.I,
)

_MM_TRIPLE = re.compile(
    r"(?P<a>\d{2,4}(?:\.\d+)?)\s*[×xX]\s*"
    r"(?P<b>\d{2,4}(?:\.\d+)?)\s*[×xX]\s*"
    r"(?P<c>\d{1,3}(?:\.\d+)?)\s*(?:mm|millimet)",
    re.I,
)

_CONVERT_MM = re.compile(
    r"\{\{convert\|(?P<a>[\d.]+)x{1,2}(?P<b>[\d.]+)x{1,2}(?P<c>[\d.]+)\|mm",
    re.I,
)

_DIM_LINE = re.compile(
    r"(?:height|length|width|depth|thickness)\s*[:=]?\s*"
    r"(?P<v>\d{1,4}(?:\.\d+)?)\s*(?:mm|millimet)",
    re.I,
)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s.+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=1)
    except Exception:
        pass


def needs_dimension_research(prompt: str) -> bool:
    """True when the prompt likely needs real product dimensions."""
    p = prompt.lower()
    if _FIT_KEYWORDS.search(p):
        return True
    # Named consumer electronics without fit keyword still often need dims
    if re.search(
        r"\b(iphone|galaxy|pixel|ipad|macbook|airpods|switch|dualsense|"
        r"xbox|ps5|kindle|watch)\b",
        p,
    ):
        return True
    return False


def extract_product_query(prompt: str) -> str:
    """Best-effort product name from a freeform prompt."""
    # Prefer longest catalog key contained in the prompt
    n = _norm(prompt)
    hits = [k for k in CATALOG if k in n]
    if hits:
        return max(hits, key=len)

    cleaned = _STRIP_FOR_PRODUCT.sub(" ", prompt)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
    # Drop trailing numbers that are sizes ("case 2mm walls")
    cleaned = re.sub(r"\b\d+(\.\d+)?\s*(mm|cm|in|inch|inches)?\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
    return cleaned or prompt.strip()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/json"})
    return s


def _catalog_lookup(query: str) -> Optional[dict]:
    n = _norm(query)
    if n in CATALOG:
        d = dict(CATALOG[n])
        d["confidence"] = "high"
        d["query"] = query
        return d
    # substring / reverse containment
    for key, val in sorted(CATALOG.items(), key=lambda kv: -len(kv[0])):
        if key in n or n in key:
            d = dict(val)
            d["confidence"] = "high"
            d["query"] = query
            d["matched_key"] = key
            return d
    return None


def _parse_mm_triple(text: str) -> Optional[tuple[float, float, float]]:
    """Return (height, width, depth) with height ≥ width ≥ depth when plausible."""
    m = _MM_TRIPLE.search(text)
    if not m:
        # Wikipedia convert template
        m2 = _CONVERT_MM.search(text)
        if m2:
            vals = [float(m2.group(k)) for k in ("a", "b", "c")]
        else:
            return None
    else:
        vals = [float(m.group(k)) for k in ("a", "b", "c")]

    # Sanity: phones/tablets roughly 50–400 × 40–300 × 4–30 mm
    if any(v <= 0 or v > 2000 for v in vals):
        return None
    # Order as height (largest), width (mid), depth (smallest) for slab-like devices
    ordered = sorted(vals, reverse=True)
    h, w, d = ordered[0], ordered[1], ordered[2]
    # If all three similar (cube-ish), keep original order
    if max(vals) / max(min(vals), 0.01) < 1.4:
        return vals[0], vals[1], vals[2]
    return h, w, d


def _wiki_lookup(query: str) -> Optional[dict]:
    sess = _session()
    try:
        r = sess.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": 6,
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        if not titles:
            return None

        # Prefer titles that share significant tokens with the query.
        # Numbers (15 vs 17) must match — otherwise "iPhone 15" can land on 17.
        q_tokens = set(_norm(query).split())
        q_nums = {t for t in q_tokens if t.isdigit() or re.fullmatch(r"\d+\.\d+", t)}

        def _title_score(t: str) -> tuple:
            tt = set(_norm(t).split())
            t_nums = {x for x in tt if x.isdigit() or re.fullmatch(r"\d+\.\d+", x)}
            num_ok = 1 if (not q_nums or q_nums <= t_nums) else 0
            overlap = len(q_tokens & tt)
            # Penalize titles that introduce a different model number
            num_penalty = len(t_nums - q_nums) if q_nums else 0
            return (num_ok, overlap, -num_penalty, -len(t))

        titles = sorted(titles, key=_title_score, reverse=True)

        for title in titles[:4]:
            # Skip clearly wrong model numbers when the query specifies one
            t_nums = {
                x for x in _norm(title).split()
                if x.isdigit() or re.fullmatch(r"\d+\.\d+", x)
            }
            if q_nums and not (q_nums <= t_nums):
                continue
            pr = sess.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "parse",
                    "page": title,
                    "prop": "wikitext",
                    "format": "json",
                    "formatversion": 2,
                    "redirects": 1,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if pr.status_code != 200:
                continue
            wikitext = (pr.json().get("parse") or {}).get("wikitext") or ""
            if not wikitext:
                continue

            # Prefer Pro Max / specific model lines when query asks for them
            qn = _norm(query)
            chunks = [wikitext]
            if "pro max" in qn:
                for line in wikitext.splitlines():
                    if re.search(r"pro\s*max", line, re.I):
                        chunks.insert(0, line)
            elif re.search(r"\bpro\b", qn) and "max" not in qn:
                for line in wikitext.splitlines():
                    if re.search(r"\bpro\b", line, re.I) and "max" not in line.lower():
                        chunks.insert(0, line)

            for chunk in chunks:
                triple = _parse_mm_triple(chunk)
                if triple:
                    h, w, d = triple
                    return {
                        "product": title,
                        "height_mm": h,
                        "width_mm": w,
                        "depth_mm": d,
                        "source": f"wikipedia:{title}",
                        "confidence": "medium",
                        "query": query,
                        "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    }
    except requests.RequestException:
        return None
    return None


def _gsmarena_lookup(query: str) -> Optional[dict]:
    """Search GSMArena for a phone and parse Dimensions row."""
    # Only attempt for phone-like queries
    if not re.search(r"iphone|galaxy|pixel|samsung|google|oneplus|xiaomi|motorola", query, re.I):
        return None
    sess = _session()
    try:
        # GSMArena search
        r = sess.get(
            "https://www.gsmarena.com/results.php3",
            params={"sQuickSearch": "yes", "sName": query},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        if r.status_code != 200:
            return None
        html = r.text
        # Follow first phone result if search page
        m = re.search(r'href="((?:apple|samsung|google|xiaomi|oneplus|motorola)_[^"]+\.php)"', html, re.I)
        if m and "results.php" in (r.url or ""):
            url = "https://www.gsmarena.com/" + m.group(1)
            r = sess.get(url, timeout=REQUEST_TIMEOUT)
            html = r.text
        # Dimensions: 163.4 x 78 x 8.8 mm
        dm = re.search(
            r'data-spec="dimensions"[^>]*>.*?</th>\s*<td[^>]*>\s*'
            r"([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)\s*mm",
            html,
            re.I | re.S,
        )
        if not dm:
            dm = re.search(
                r"([\d.]{2,4})\s*x\s*([\d.]{2,4})\s*x\s*([\d.]{1,3})\s*mm\s*\(",
                html,
                re.I,
            )
        if not dm:
            return None
        h, w, d = float(dm.group(1)), float(dm.group(2)), float(dm.group(3))
        # GSMArena uses H x W x D already
        name_m = re.search(r"<title>([^|<]+)", html)
        product = (name_m.group(1).strip() if name_m else query).replace(" - Full phone specifications", "")
        return {
            "product": product,
            "height_mm": h,
            "width_mm": w,
            "depth_mm": d,
            "source": f"gsmarena:{urlparse(r.url).path.lstrip('/')}",
            "confidence": "medium",
            "query": query,
            "url": r.url,
        }
    except requests.RequestException:
        return None


def _dimensions_com_lookup(query: str) -> Optional[dict]:
    """dimensions.com often has clean mm for consumer products."""
    sess = _session()
    slug = re.sub(r"[^a-z0-9]+", "-", _norm(query)).strip("-")
    candidates = [
        f"https://www.dimensions.com/element/{slug}",
        f"https://www.dimensions.com/element/apple-{slug}",
    ]
    # Known slug aliases
    aliases = {
        "iphone 17 pro max": "apple-iphone-17-pro-max-19th-gen",
        "iphone 17 pro": "apple-iphone-17-pro-19th-gen",
    }
    n = _norm(query)
    if n in aliases:
        candidates.insert(0, f"https://www.dimensions.com/element/{aliases[n]}")

    for url in candidates:
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text)
            # "height of 6.43" (163.4 mm), width of 3.07" (78 mm), depth of .34" (8.75 mm)"
            hm = re.search(r"height[^()]{0,40}\(([\d.]+)\s*mm\)", text, re.I)
            wm = re.search(r"width[^()]{0,40}\(([\d.]+)\s*mm\)", text, re.I)
            dm = re.search(r"depth[^()]{0,40}\(([\d.]+)\s*mm\)", text, re.I)
            if hm and wm and dm:
                return {
                    "product": query,
                    "height_mm": float(hm.group(1)),
                    "width_mm": float(wm.group(1)),
                    "depth_mm": float(dm.group(1)),
                    "source": f"dimensions.com:{urlparse(url).path}",
                    "confidence": "medium",
                    "query": query,
                    "url": url,
                }
            triple = _parse_mm_triple(text)
            if triple:
                h, w, d = triple
                return {
                    "product": query,
                    "height_mm": h,
                    "width_mm": w,
                    "depth_mm": d,
                    "source": f"dimensions.com:{urlparse(url).path}",
                    "confidence": "medium",
                    "query": query,
                    "url": url,
                }
        except requests.RequestException:
            continue
    return None


def research_dimensions(prompt: str, force: bool = False) -> Optional[dict]:
    """Look up product dimensions for a generation prompt.

    Returns dict with height_mm, width_mm, depth_mm, product, source, confidence,
    plus fit guidance fields — or None if nothing found / not needed.
    """
    if not force and not needs_dimension_research(prompt):
        # Still try if the user embedded explicit mm triple
        triple = _parse_mm_triple(prompt)
        if triple:
            h, w, d = triple
            return {
                "product": "user-specified",
                "height_mm": h,
                "width_mm": w,
                "depth_mm": d,
                "source": "prompt",
                "confidence": "high",
                "query": prompt,
            }
        return None

    query = extract_product_query(prompt)
    if not query:
        return None

    cache = _load_cache()
    cache_key = _norm(query)
    hit = cache.get(cache_key)
    if hit and (time.time() - hit.get("_ts", 0)) < CACHE_TTL_S:
        out = dict(hit)
        out.pop("_ts", None)
        out["from_cache"] = True
        return _enrich_fit(out, prompt)

    # 1) Catalog
    found = _catalog_lookup(query) or _catalog_lookup(prompt)

    # 2) Online sources
    if not found:
        found = _wiki_lookup(query)
    if not found:
        found = _gsmarena_lookup(query)
    if not found:
        found = _dimensions_com_lookup(query)

    # 3) Explicit dimensions in the prompt win for overrides
    triple = _parse_mm_triple(prompt)
    if triple and found:
        # User may specify overall case size — keep product body dims from research
        pass
    elif triple and not found:
        h, w, d = triple
        found = {
            "product": query,
            "height_mm": h,
            "width_mm": w,
            "depth_mm": d,
            "source": "prompt",
            "confidence": "high",
            "query": query,
        }

    if not found:
        return None

    # Round for readability
    for k in ("height_mm", "width_mm", "depth_mm", "corner_radius_mm"):
        if k in found and found[k] is not None:
            found[k] = round(float(found[k]), 2)

    cache[cache_key] = {**found, "_ts": time.time()}
    _save_cache(cache)
    return _enrich_fit(found, prompt)


def _enrich_fit(dims: dict, prompt: str) -> dict:
    """Add print-fit clearances and expected outer envelope for cases/holders."""
    h = float(dims["height_mm"])
    w = float(dims["width_mm"])
    d = float(dims["depth_mm"])
    is_case = bool(re.search(r"\b(case|casing|cover|sleeve|shell|skin|protector)\b", prompt, re.I))
    is_holder = bool(re.search(r"\b(holder|stand|mount|dock|cradle|holster|tray)\b", prompt, re.I))

    wall = 1.6
    clearance = 0.4  # per side — snug but printable FDM
    corner = float(dims.get("corner_radius_mm") or min(h, w) * 0.12)

    dims = dict(dims)
    dims["wall_mm"] = wall
    dims["clearance_mm"] = clearance
    dims["corner_radius_mm"] = round(corner, 2)
    dims["cavity_height_mm"] = round(h + 2 * clearance, 2)
    dims["cavity_width_mm"] = round(w + 2 * clearance, 2)
    dims["cavity_depth_mm"] = round(d + clearance, 2)  # open face side often less

    if is_case:
        # Outer envelope of a protective phone case
        dims["part_type"] = "phone_case"
        dims["outer_height_mm"] = round(h + 2 * clearance + 2 * wall, 2)
        dims["outer_width_mm"] = round(w + 2 * clearance + 2 * wall, 2)
        # Depth: back wall + phone + raised lip (~1.2 mm) above screen plane
        lip = 1.2
        dims["outer_depth_mm"] = round(wall + d + clearance + lip, 2)
        dims["lip_mm"] = lip
        dims["expected_bbox_mm"] = [
            dims["outer_width_mm"],
            dims["outer_height_mm"],
            dims["outer_depth_mm"],
        ]
    elif is_holder:
        dims["part_type"] = "holder"
        dims["outer_height_mm"] = round(max(h * 0.55, 40), 1)
        dims["outer_width_mm"] = round(w + 2 * wall + 4, 1)
        dims["outer_depth_mm"] = round(d + 20, 1)
        dims["expected_bbox_mm"] = [
            dims["outer_width_mm"],
            dims["outer_depth_mm"],
            dims["outer_height_mm"],
        ]
    else:
        dims["part_type"] = "product_fit"
        dims["expected_bbox_mm"] = [w, h, d]

    return dims


def format_dimension_brief(dims: dict) -> str:
    """Human/LLM-readable block injected into the codegen prompt."""
    lines = [
        "=== RESEARCHED REAL-WORLD DIMENSIONS (mm) — YOU MUST USE THESE ===",
        f"Product: {dims.get('product', '?')}",
        f"Source: {dims.get('source', '?')} (confidence: {dims.get('confidence', '?')})",
        f"Body height (long axis): {dims['height_mm']} mm",
        f"Body width  (short axis): {dims['width_mm']} mm",
        f"Body depth  (thickness):  {dims['depth_mm']} mm",
    ]
    if dims.get("corner_radius_mm"):
        lines.append(f"Corner radius (approx):   {dims['corner_radius_mm']} mm")
    if dims.get("part_type") == "phone_case":
        lines += [
            "",
            "This is a PHONE CASE that must FIT the product above:",
            f"  wall thickness:     {dims['wall_mm']} mm",
            f"  per-side clearance: {dims['clearance_mm']} mm",
            f"  cavity (inner):     {dims['cavity_width_mm']} × {dims['cavity_height_mm']} × ~{dims['cavity_depth_mm']} mm (W×H×D)",
            f"  outer envelope:     {dims['outer_width_mm']} × {dims['outer_height_mm']} × {dims['outer_depth_mm']} mm (W×H×D)",
            f"  screen lip height:  {dims.get('lip_mm', 1.2)} mm above phone face",
            "  Required features:",
            "    - Hollow shell open on the front (screen side), NOT a solid brick",
            "    - Rounded corners matching the phone",
            "    - Bottom USB-C / charging cutout (~12×7 mm)",
            "    - Camera island cutout on the back (top-left, ~40×40 mm for Pro Max class)",
            "    - Side button recesses (power right, volume left) — shallow pockets, not through-holes if fragile",
            "    - Minimum printable wall 1.2 mm; prefer 1.5–1.8 mm",
            "  DO NOT invent different phone dimensions. DO NOT make a flat 0.8 mm plate.",
        ]
    elif dims.get("part_type") == "holder":
        lines += [
            "",
            "This is a HOLDER/STAND that must accommodate the product above.",
            f"  Provide a pocket/slot sized to width {dims['width_mm']}+clearance and depth {dims['depth_mm']}+clearance.",
            "  Stable base; phone/product leans or sits securely.",
        ]
    else:
        lines += [
            "",
            "Use these dimensions as the primary size of the object (or the object it fits).",
        ]
    if dims.get("notes"):
        lines.append(f"Notes: {dims['notes']}")
    if dims.get("url"):
        lines.append(f"URL: {dims['url']}")
    lines.append("=== END RESEARCHED DIMENSIONS ===")
    return "\n".join(lines)


def dims_match_bbox(
    dims: dict,
    bbox_mm: list,
    tol_frac: float = 0.18,
) -> tuple[bool, str]:
    """Check whether an exported bbox is roughly the expected outer envelope.

    bbox_mm is [dx, dy, dz] unordered relative to W/H/D — we compare sorted triples.
    """
    expected = dims.get("expected_bbox_mm")
    if not expected or not bbox_mm or len(bbox_mm) < 3:
        return True, "no expected envelope"
    exp = sorted(float(x) for x in expected)
    got = sorted(float(x) for x in bbox_mm[:3])
    for e, g in zip(exp, got):
        if e <= 0:
            continue
        err = abs(g - e) / e
        if err > tol_frac:
            return (
                False,
                f"bbox {got} mm far from expected ~{exp} mm (>{tol_frac:.0%} off on an axis)",
            )
    return True, "bbox within tolerance"
