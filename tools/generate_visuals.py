#!/usr/bin/env python3
"""Generate AccelMark brand & empirical data visuals directly from live leaderboard data.

Outputs:
  - docs/assets/rank-shift-light.svg
  - docs/assets/rank-shift-dark.svg
  - docs/assets/roofline-suites-light.svg
  - docs/assets/roofline-suites-dark.svg
  - docs/assets/og-card.svg
  - leaderboard/site/og-image.png  (1200x630 via rsvg-convert when available)

Zero third-party Python dependencies (stdlib only).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD_JS = ROOT / "leaderboard" / "site" / "leaderboard.js"
DOCS_ASSETS = ROOT / "docs" / "assets"
SITE_DIR = ROOT / "leaderboard" / "site"

# Gen-1 exact traced vector paths (64x64 viewBox)
GEN1_LEFT = "30.03,10.5 59.0,10.5 54.55,16.71 33.05,16.71 13.57,53.41 5.0,53.41"
GEN1_RIGHT = "34.81,19.57 37.92,19.57 59.0,53.41 50.27,53.41 43.63,41.66 21.8,41.66 39.26,34.01 33.13,22.68"
GEN1_DROP = (
    "M 31.58 26.2 C 31.58 26.2 34.14 29.73 34.14 31.37 "
    "A 2.56 2.56 0 1 1 29.02 31.37 C 29.02 29.73 31.58 26.2 31.58 26.2 Z"
)


def load_leaderboard_rows() -> list[dict]:
    txt = LEADERBOARD_JS.read_text(encoding="utf-8")
    m = re.search(r"const\s+LEADERBOARD_DATA\s*=\s*(\[.*?\]);\s*\n", txt, re.S)
    if not m:
        raise RuntimeError(f"Could not parse LEADERBOARD_DATA from {LEADERBOARD_JS}")
    return json.loads(m.group(1))


def compute_rank_shift_series(rows: list[dict]) -> tuple[list[dict], dict]:
    """Select the 5 canonical reference vLLM chips present across A, D, and F (value)."""
    ref = [
        r for r in rows
        if (r.get("runner_tier") or "reference") == "reference"
        and not r.get("flagged")
        and (r.get("chip_count") or 1) == 1
        and r.get("framework") == "vLLM"
    ]
    by_chip: dict[str, dict] = {}
    for r in ref:
        c = r.get("chip")
        if not c:
            continue
        cur = by_chip.setdefault(c, {"chip": c})
        suite = r.get("suite")
        off = r.get("offline_throughput")
        if suite == "suite_A" and isinstance(off, (int, float)):
            if off > cur.get("A", -1):
                cur["A"] = float(off)
                cur["QPS"] = float(r.get("online_max_qps") or 0.0)
        elif suite == "suite_D" and isinstance(off, (int, float)):
            if off > cur.get("D", -1):
                cur["D"] = float(off)
        elif suite == "suite_F":
            val = r.get("cost_efficiency_toks_per_dollar_hr")
            if isinstance(val, (int, float)) and val > cur.get("F_val", -1):
                cur["F_val"] = float(val)

    # Prefer the 5 iconic hardware tiers when available
    preferred = [
        ("NVIDIA H200", "H200"),
        ("NVIDIA H100 80GB HBM3", "H100 80GB"),
        ("NVIDIA GeForce RTX 5090", "RTX 5090"),
        ("NVIDIA A100-SXM4-80GB", "A100 80GB"),
        ("NVIDIA RTX A6000", "RTX A6000"),
    ]
    picked = []
    for full, short in preferred:
        if full in by_chip and all(k in by_chip[full] for k in ("A", "QPS", "D", "F_val")):
            item = dict(by_chip[full])
            item["short"] = short
            picked.append(item)

    if len(picked) < 4:
        # Dynamic fallback: top 5 chips with all 3 suites
        cands = [v for v in by_chip.values() if all(k in v for k in ("A", "QPS", "D", "F_val"))]
        cands.sort(key=lambda x: -x["A"])
        picked = []
        for v in cands[:5]:
            v = dict(v)
            v["short"] = v["chip"].replace("NVIDIA ", "").replace("GeForce ", "")
            picked.append(v)

    # Compute ranks (1..N) per axis with deterministic tie-breaking by Suite A throughput
    axes = [
        ("A", lambda x: (-x["A"], x["short"])),
        ("QPS", lambda x: (-x["QPS"], -x["A"], x["short"])),
        ("D", lambda x: (-x["D"], x["short"])),
        ("F_val", lambda x: (-x["F_val"], x["short"])),
    ]
    for key, sort_fn in axes:
        ordered = sorted(picked, key=sort_fn)
        for idx, item in enumerate(ordered, start=1):
            item[f"rank_{key}"] = idx

    # Overall dataset stats
    stats = {
        "runs": len(rows),
        "chips": len({r.get("chip") for r in rows if r.get("chip")}),
        "vendors": len({r.get("vendor") for r in rows if r.get("vendor")}),
        "verified": sum(1 for r in rows if r.get("tier") == "verified"),
    }
    return picked, stats


def fmt_k(v: float) -> str:
    if v >= 10000:
        return f"{v / 1000:.1f}k"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def render_rank_shift_svg(series: list[dict], stats: dict, dark: bool = False) -> str:
    W, H = 880, 384
    if dark:
        bg = "#09090B"
        card = "#111114"
        border = "#27272A"
        grid = "#222228"
        fg_strong = "#FAFAFA"
        fg = "#D4D4D8"
        fg_muted = "#A1A1AA"
        fg_faint = "#71717A"
        pill_bg = "#18181D"
        palette = {
            "RTX 5090": ("#3B82F6", 3.2, 1.0),
            "H100 80GB": ("#34D399", 2.6, 0.95),
            "H200": ("#FAFAFA", 2.4, 0.92),
            "A100 80GB": ("#F59E0B", 2.2, 0.9),
            "RTX A6000": ("#71717A", 1.9, 0.78),
        }
    else:
        bg = "#FFFFFF"
        card = "#FAFAFA"
        border = "#E4E4E7"
        grid = "#ECECF0"
        fg_strong = "#09090B"
        fg = "#3F3F46"
        fg_muted = "#71717A"
        fg_faint = "#A1A1AA"
        pill_bg = "#FFFFFF"
        palette = {
            "RTX 5090": ("#2563EB", 3.2, 1.0),
            "H100 80GB": ("#059669", 2.6, 0.95),
            "H200": ("#18181B", 2.4, 0.92),
            "A100 80GB": ("#D97706", 2.2, 0.9),
            "RTX A6000": ("#A1A1AA", 1.9, 0.8),
        }

    cols = [
        ("A", "Short Chat", "Suite A · Offline tok/s"),
        ("QPS", "500 ms SLA", "Suite A · Online QPS"),
        ("D", "28K Prefill", "Suite D · Long Context"),
        ("F_val", "Cost Efficiency", "Suite F · Tok / $/hr"),
    ]
    xs = [184, 372, 560, 748]
    y_top, y_step = 132, 48
    ys = [y_top + i * y_step for i in range(len(series))]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="AccelMark Rank-Shift Chart">',
        "<title>Hardware rankings reverse across LLM serving regimes on AccelMark</title>",
        "<style>",
        "  .f { font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', sans-serif; }",
        "  .m { font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; }",
        "</style>",
        f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="12" fill="{bg}" stroke="{border}"/>',
        # Header
        f'<text x="28" y="34" class="f" font-size="15" font-weight="700" fill="{fg_strong}" letter-spacing="-0.015em">Same serving stack (vLLM), four regimes — hardware rankings invert</text>',
        f'<text x="28" y="54" class="f" font-size="12" fill="{fg_muted}">Verified reference runs across 5 accelerators. Higher = better rank (#1 at top). Watch RTX 5090 &amp; H200 cross.</text>',
        # Legend pill top-right
        f'<rect x="{W-182}" y="20" width="154" height="24" rx="12" fill="{card}" stroke="{border}"/>',
        f'<circle cx="{W-168}" cy="32" r="4" fill="{palette["RTX 5090"][0]}"/>',
        f'<text x="{W-158}" y="36" class="m" font-size="10.5" font-weight="600" fill="{fg}">Live Verified · vLLM</text>',
    ]

    # Column guides and headers
    for (key, title, sub), x in zip(cols, xs):
        parts.append(f'<line x1="{x}" y1="106" x2="{x}" y2="{ys[-1] + 14}" stroke="{grid}" stroke-width="1" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{x}" y="82" text-anchor="middle" class="f" font-size="12.5" font-weight="650" fill="{fg_strong}">{escape(title)}</text>')
        parts.append(f'<text x="{x}" y="97" text-anchor="middle" class="f" font-size="10.5" fill="{fg_muted}">{escape(sub)}</text>')

    # Rank row markers on far left
    for idx, y in enumerate(ys, start=1):
        parts.append(f'<text x="28" y="{y + 4}" class="m" font-size="11" font-weight="600" fill="{fg_faint}">#{idx}</text>')

    # Draw trajectories (muted ones first, highlighted RTX 5090 on top)
    draw_order = sorted(series, key=lambda s: 1 if s["short"] == "RTX 5090" else 0)
    for item in draw_order:
        name = item["short"]
        color, sw, op = palette.get(name, ("#71717A", 2.0, 0.85))
        pts = [(xs[i], ys[item[f"rank_{cols[i][0]}"] - 1]) for i in range(4)]

        # Smooth cubic Bézier path through the 4 regime points
        d_cmds = [f"M {pts[0][0]} {pts[0][1]}"]
        for i in range(3):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            cx = (x0 + x1) / 2
            d_cmds.append(f"C {cx:.1f} {y0}, {cx:.1f} {y1}, {x1} {y1}")
        path_d = " ".join(d_cmds)

        # Halo for clean visual separation at crossings
        parts.append(f'<path d="{path_d}" fill="none" stroke="{bg}" stroke-width="{sw + 4}" stroke-linecap="round"/>')
        parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" opacity="{op}"/>')

        # Left & right labels (clear of 48px-wide value pills)
        fw = "700" if name in ("RTX 5090", "H100 80GB", "H200") else "500"
        parts.append(f'<text x="{pts[0][0] - 36}" y="{pts[0][1] + 4}" text-anchor="end" class="f" font-size="12" font-weight="{fw}" fill="{color if name == "RTX 5090" else fg_strong}">{escape(name)}</text>')
        parts.append(f'<text x="{pts[-1][0] + 36}" y="{pts[-1][1] + 4}" text-anchor="start" class="f" font-size="12" font-weight="{fw}" fill="{color if name == "RTX 5090" else fg_strong}">{escape(name)}</text>')

        # Value pills on each node
        val_labels = [
            f'{fmt_k(item["A"])}',
            "SLA fail" if item["QPS"] == 0 else f'{int(item["QPS"])} QPS',
            f'{fmt_k(item["D"])}',
            f'{fmt_k(item["F_val"])}',
        ]
        for (px, py), vtxt in zip(pts, val_labels):
            is_fail = vtxt == "SLA fail"
            pw = 54 if is_fail else 48
            pill_stroke = "#DC2626" if is_fail else color
            txt_fill = "#F87171" if (is_fail and dark) else ("#DC2626" if is_fail else fg_strong)
            parts.append(f'<rect x="{px - pw/2:.1f}" y="{py - 11}" width="{pw}" height="22" rx="11" fill="{pill_bg}" stroke="{pill_stroke}" stroke-width="1.35"/>')
            parts.append(f'<text x="{px}" y="{py + 3.5}" text-anchor="middle" class="m" font-size="10" font-weight="650" fill="{txt_fill}">{escape(vtxt)}</text>')

    # Footer caption
    parts.append(f'<line x1="28" y1="{H-36}" x2="{W-28}" y2="{H-36}" stroke="{border}" stroke-width="1"/>')
    parts.append(f'<text x="28" y="{H-15}" class="f" font-size="11" fill="{fg_muted}">Source: AccelMark live dataset ({stats["runs"]} runs · {stats["chips"]} chips · {stats["verified"]} verified) · Reference vLLM stack</text>')
    parts.append(f'<text x="{W-28}" y="{H-15}" text-anchor="end" class="m" font-size="10.5" fill="{fg_faint}">freedomintelligence.github.io/AccelMark</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_roofline_svg(dark: bool = False) -> str:
    """Architectural 2-color Roofline Spectrum SVG (Zinc + Cobalt Blue, zero rainbow dots)."""
    W, H = 880, 310
    if dark:
        bg, surface, border = "#09090B", "#141418", "#27272A"
        grid = "#1D1D23"
        fg_strong, fg, fg_muted = "#FAFAFA", "#D4D4D8", "#A1A1AA"
        accent = "#3B82F6"
        accent_soft = "rgba(59,130,246,0.16)"
        badge_bg = "#1C2434"
        badge_fg = "#93C5FD"
        stem = "#3F3F46"
        area_top, area_bot = "#3B82F6", "#09090B"
        area_op = "0.16"
    else:
        bg, surface, border = "#FFFFFF", "#FFFFFF", "#E4E4E7"
        grid = "#F1F1F4"
        fg_strong, fg, fg_muted = "#09090B", "#3F3F46", "#71717A"
        accent = "#2563EB"
        accent_soft = "rgba(37,99,235,0.08)"
        badge_bg = "#EFF6FF"
        badge_fg = "#1D4ED8"
        stem = "#D4D4D8"
        area_top, area_bot = "#2563EB", "#FFFFFF"
        area_op = "0.10"

    x0, y0, x1, y1 = 56, 60, 844, 260
    knee_x, knee_y = 548, 144

    # Helper: y on roofline curve for any x in [x0+18, x1-16]
    slope_x0, slope_y0 = x0 + 24, y1 - 10
    def roof_y(x: float) -> float:
        if x >= knee_x:
            return float(knee_y)
        t = (x - slope_x0) / (knee_x - slope_x0)
        return slope_y0 + t * (knee_y - slope_y0)

    # 8 suites placed cleanly along the roofline with staggered hairline callout cards
    # (x_anchor, letter, title, sub, card_x, card_y, card_w)
    cards = [
        (114, "F", "Edge",        "Qwen-0.5B",   64,  156, 102),
        (198, "G", "Sparse MoE",  "Mixtral-8x7B",144, 114, 118),
        (286, "A", "Short Chat",  "8B + SLA",    234, 214, 114),
        (374, "H", "Sliding Win", "Mistral-7B",  314, 102, 118),
        (462, "C", "Quantize",    "5 precisions",384, 198, 116),
        (548, "B", "70B Multi",   "Llama-70B",   516, 178, 112),
        (668, "E", "TP Scaling",  "1/2/4/8×",    614, 90,  112),
        (782, "D", "28K Prefill", "Long ctx",    722, 178, 114),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="AccelMark 8-Suite Roofline Spectrum">',
        "<title>AccelMark 8-Suite Roofline Spectrum</title>",
        "<style>",
        "  .f { font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', 'Segoe UI', sans-serif; }",
        "  .m { font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; }",
        "</style>",
        "<defs>",
        f'  <linearGradient id="rfArea" x1="0" y1="0" x2="0" y2="1">',
        f'    <stop offset="0%" stop-color="{area_top}" stop-opacity="{area_op}"/>',
        f'    <stop offset="100%" stop-color="{area_bot}" stop-opacity="0"/>',
        "  </linearGradient>",
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="12" fill="{bg}" stroke="{border}"/>',
        # Header
        f'<text x="28" y="32" class="f" font-size="14.5" font-weight="700" fill="{fg_strong}" letter-spacing="-0.015em">The 8-Suite Roofline Spectrum</text>',
        f'<text x="{W-28}" y="32" text-anchor="end" class="f" font-size="11.5" fill="{fg_muted}">Each suite isolates a distinct hardware or serving bottleneck</text>',
        f'<line x1="28" y1="46" x2="{W-28}" y2="46" stroke="{grid}" stroke-width="1"/>',
    ]

    # Subtle horizontal grid lines
    for gy in (96, 136, 176, 216):
        parts.append(f'<line x1="{x0}" y1="{gy}" x2="{x1}" y2="{gy}" stroke="{grid}" stroke-width="1"/>')

    # Vertical Ridge knee divider + regime captions
    parts.append(f'<line x1="{knee_x}" y1="{y0}" x2="{knee_x}" y2="{y1}" stroke="{border}" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{x0 + 8}" y="{y0 + 14}" class="m" font-size="9.5" font-weight="600" fill="{accent}" letter-spacing="0.08em">MEMORY-BANDWIDTH BOUND</text>')
    parts.append(f'<text x="{x1 - 8}" y="{y0 + 14}" text-anchor="end" class="m" font-size="9.5" font-weight="600" fill="{fg_muted}" letter-spacing="0.08em">COMPUTE &amp; INTERCONNECT BOUND</text>')

    # Shaded area under roofline + crisp roofline curve
    parts.append(f'<path d="M {slope_x0} {y1} L {slope_x0} {slope_y0} L {knee_x} {knee_y} L {x1 - 12} {knee_y} L {x1 - 12} {y1} Z" fill="url(#rfArea)"/>')
    parts.append(f'<path d="M {slope_x0} {slope_y0} L {knee_x} {knee_y} L {x1 - 12} {knee_y}" fill="none" stroke="{accent}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')

    # Clean axes
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{border}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{border}" stroke-width="1.2"/>')

    # Callout stems, ticks, and crisp architectural pill-cards
    card_h = 36
    for ax, letter, title, sub, cx, cy, cw in cards:
        ry = roof_y(ax)
        # Stem from roofline anchor to nearest edge of card
        stem_y = cy + card_h if cy + card_h < ry else cy
        parts.append(f'<line x1="{ax}" y1="{ry:.1f}" x2="{ax}" y2="{stem_y}" stroke="{stem}" stroke-width="1.2"/>')
        # Precision tick dot on the curve (tiny 3.5px architectural node, NOT rainbow balls)
        parts.append(f'<circle cx="{ax}" cy="{ry:.1f}" r="3.5" fill="{bg}" stroke="{accent}" stroke-width="2"/>')
        # Hairline callout card
        parts.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{card_h}" rx="7" fill="{surface}" stroke="{border}" stroke-width="1"/>')
        # Monospace letter badge inside card
        parts.append(f'<rect x="{cx + 6}" y="{cy + 7}" width="20" height="22" rx="4.5" fill="{badge_bg}"/>')
        parts.append(f'<text x="{cx + 16}" y="{cy + 21.5}" text-anchor="middle" class="m" font-size="11" font-weight="700" fill="{badge_fg}">{letter}</text>')
        # Concise title + model/regime subtitle
        parts.append(f'<text x="{cx + 32}" y="{cy + 16.5}" class="f" font-size="11" font-weight="650" fill="{fg_strong}">{escape(title)}</text>')
        parts.append(f'<text x="{cx + 32}" y="{cy + 29}" class="m" font-size="9.2" fill="{fg_muted}">{escape(sub)}</text>')

    # Axis labels
    parts.append(f'<text x="{(x0 + x1)/2:.0f}" y="{H - 18}" text-anchor="middle" class="f" font-size="11" fill="{fg_muted}">Arithmetic Intensity (FLOPs / Byte)  →  short decode to 28K prefill</text>')
    parts.append(f'<text x="24" y="{(y0 + y1)/2:.0f}" text-anchor="middle" class="f" font-size="11" fill="{fg_muted}" transform="rotate(-90 24 {(y0 + y1)/2:.0f})">Throughput</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_og_card_svg(series: list[dict], stats: dict) -> str:
    """1200x630 Studio Dark Open Graph / Social Preview Card."""
    W, H = 1200, 630
    bg = "#09090B"
    card = "#111114"
    border = "#27272A"
    fg = "#FAFAFA"
    muted = "#A1A1AA"
    accent = "#3B82F6"

    # Mini rank-shift trajectories inside right panel (x: 645..1135, y: 170..480)
    cols_x = [718, 836, 954, 1072]
    col_names = ["Chat", "SLA", "28K", "Tok/$"]
    col_keys = ["A", "QPS", "D", "F_val"]
    y0, y_step = 232, 54
    ys = [y0 + i * y_step for i in range(len(series))]
    palette = {
        "RTX 5090": ("#3B82F6", 4.0),
        "H100 80GB": ("#34D399", 3.0),
        "H200": ("#FAFAFA", 2.8),
        "A100 80GB": ("#F59E0B", 2.6),
        "RTX A6000": ("#71717A", 2.2),
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        "<style>",
        "  .f { font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Display', 'Segoe UI', sans-serif; }",
        "  .m { font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; }",
        "</style>",
        "<defs>",
        '  <pattern id="g" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#18181D" stroke-width="1"/></pattern>',
        '  <radialGradient id="glow" cx="78%" cy="35%" r="55%"><stop offset="0%" stop-color="#2563EB" stop-opacity="0.22"/><stop offset="100%" stop-color="#09090B" stop-opacity="0"/></radialGradient>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{bg}"/>',
        f'<rect width="{W}" height="{H}" fill="url(#g)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#glow)"/>',
        # Top-left Gen-1 Wordmark
        '<g transform="translate(64,62) scale(0.78)">',
        f'  <polygon points="{GEN1_LEFT}" fill="#FAFAFA"/>',
        f'  <polygon points="{GEN1_RIGHT}" fill="#FAFAFA"/>',
        f'  <path d="{GEN1_DROP}" fill="#3B82F6"/>',
        "</g>",
        '<text x="124" y="96" class="f" font-size="28" font-weight="700" fill="#FAFAFA" letter-spacing="-0.03em">AccelMark</text>',
        # Left Hero Copy
        '<text x="64" y="218" class="f" font-size="52" font-weight="750" fill="#FAFAFA" letter-spacing="-0.03em">No single fastest</text>',
        '<text x="64" y="278" class="f" font-size="52" font-weight="750" fill="#FAFAFA" letter-spacing="-0.03em">accelerator.</text>',
        '<text x="64" y="346" class="f" font-size="30" font-weight="600" fill="#3B82F6" letter-spacing="-0.02em">Only the best serving recipe.</text>',
        '<text x="64" y="408" class="f" font-size="20" fill="#A1A1AA">Open, reproducible LLM inference benchmarks across</text>',
        '<text x="64" y="438" class="f" font-size="20" fill="#A1A1AA">NVIDIA · AMD · Ascend · Apple · TPU · MTT · Iluvatar</text>',
        # Stats bar bottom-left
        f'<rect x="64" y="492" width="516" height="72" rx="12" fill="{card}" stroke="{border}"/>',
    ]

    stat_items = [
        (f'{stats["runs"]}', "RUNS"),
        (f'{stats["chips"]}', "CHIPS"),
        ("7", "ECOSYSTEMS"),
        (f'{stats["verified"]}', "VERIFIED"),
    ]
    for i, (num, lbl) in enumerate(stat_items):
        sx = 124 + i * 128
        parts.append(f'<text x="{sx}" y="527" text-anchor="middle" class="m" font-size="24" font-weight="700" fill="{fg}">{num}</text>')
        parts.append(f'<text x="{sx}" y="549" text-anchor="middle" class="m" font-size="10.5" font-weight="600" fill="{muted}" letter-spacing="0.08em">{lbl}</text>')
        if i < 3:
            parts.append(f'<line x1="{sx + 64}" y1="508" x2="{sx + 64}" y2="548" stroke="{border}" stroke-width="1"/>')

    # Right chart card
    parts.append(f'<rect x="624" y="96" width="516" height="468" rx="16" fill="{card}" stroke="{border}"/>')
    parts.append('<text x="654" y="138" class="m" font-size="12" font-weight="600" fill="#3B82F6" letter-spacing="0.08em">LIVE RANK INVERSION · vLLM</text>')
    parts.append('<text x="654" y="164" class="f" font-size="16" font-weight="600" fill="#E4E4E7">RTX 5090 wins Chat &amp; Tok/$, loses SLA &amp; 28K</text>')

    for cx, cname in zip(cols_x, col_names):
        parts.append(f'<line x1="{cx}" y1="208" x2="{cx}" y2="480" stroke="#22222A" stroke-width="1" stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{cx}" y="198" text-anchor="middle" class="m" font-size="12" font-weight="600" fill="{muted}">{cname}</text>')

    draw_order = sorted(series, key=lambda s: 1 if s["short"] == "RTX 5090" else 0)
    for item in draw_order:
        name = item["short"]
        col, sw = palette.get(name, ("#71717A", 2.4))
        pts = [(cols_x[i], ys[item[f"rank_{col_keys[i]}"] - 1]) for i in range(4)]
        d_cmds = [f"M {pts[0][0]} {pts[0][1]}"]
        for i in range(3):
            x0, y_a = pts[i]
            x1, y_b = pts[i + 1]
            mx = (x0 + x1) / 2
            d_cmds.append(f"C {mx:.1f} {y_a}, {mx:.1f} {y_b}, {x1} {y_b}")
        pd = " ".join(d_cmds)
        parts.append(f'<path d="{pd}" fill="none" stroke="{card}" stroke-width="{sw + 5}" stroke-linecap="round"/>')
        parts.append(f'<path d="{pd}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"/>')
        for px, py in pts:
            parts.append(f'<circle cx="{px}" cy="{py}" r="5.5" fill="{card}" stroke="{col}" stroke-width="2.5"/>')
        short_lbl = name.replace(" 80GB", "")
        parts.append(f'<text x="{pts[0][0] - 12}" y="{pts[0][1] + 4}" text-anchor="end" class="f" font-size="11.5" font-weight="650" fill="{col if name == "RTX 5090" else "#E4E4E7"}">{escape(short_lbl)}</text>')
        parts.append(f'<text x="{pts[-1][0] + 12}" y="{pts[-1][1] + 4}" text-anchor="start" class="f" font-size="11.5" font-weight="650" fill="{col if name == "RTX 5090" else "#E4E4E7"}">{escape(short_lbl)}</text>')

    parts.append('<text x="654" y="534" class="m" font-size="12" fill="#71717A">freedomintelligence.github.io/AccelMark</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    rows = load_leaderboard_rows()
    series, stats = compute_rank_shift_series(rows)
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)

    (DOCS_ASSETS / "rank-shift-light.svg").write_text(render_rank_shift_svg(series, stats, dark=False), encoding="utf-8")
    (DOCS_ASSETS / "rank-shift-dark.svg").write_text(render_rank_shift_svg(series, stats, dark=True), encoding="utf-8")
    (DOCS_ASSETS / "roofline-suites-light.svg").write_text(render_roofline_svg(dark=False), encoding="utf-8")
    (DOCS_ASSETS / "roofline-suites-dark.svg").write_text(render_roofline_svg(dark=True), encoding="utf-8")

    og_svg = DOCS_ASSETS / "og-card.svg"
    og_svg.write_text(render_og_card_svg(series, stats), encoding="utf-8")

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        og_png = SITE_DIR / "og-image.png"
        subprocess.run([rsvg, "-w", "1200", "-h", "630", str(og_svg), "-o", str(og_png)], check=True)
        print(f"Rendered {og_png} ({og_png.stat().st_size // 1024} KB)")
    else:
        print("Note: rsvg-convert not found; skipped rasterizing og-image.png")

    print(f"Generated visuals from {stats['runs']} runs ({len(series)} trajectories).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
