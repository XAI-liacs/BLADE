"""The BLADE platform figure (issue #235): pipeline layout with icons.

Writes ``docs/media/blade-platform.svg``; render the PNG with
``node docs/media/figure/render.js`` (Chromium via Playwright).

Run from the repository root:
    uv run python docs/media/figure/make_platform_figure.py
"""

from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ICONS = HERE / "icons"
OUT = ROOT / "docs" / "media" / "blade-platform.svg"
OUT_DARK = ROOT / "docs" / "media" / "blade-platform-dark.svg"

W, H = 2400, 1000
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace"

THEMES = {
    "light": dict(
        INK="#14161c", MUTED="#5b5f6b", BG="#ffffff", SURFACE="#ffffff",
        AXIS="#c9ccd6", CARD="#ffd0e4", CODE_BG="#0b0d12", CODE_EDGE="#0b0d12",
        LEAF="logo.png",
        COL={  # accent, tint, strong text
            "violet": ("#7c5cff", "#f0ecff", "#5a3de0"),
            "pink": ("#ff7ab6", "#ffecf4", "#c83c7f"),
            "amber": ("#ffb547", "#fff4e2", "#b8730a"),
            "grey": ("#8a8f9c", "#f3f4f7", "#4a4f5c"),
        },
    ),
    "dark": dict(
        INK="#eef0f5", MUTED="#a3a8b5", BG="#0b0d12", SURFACE="#12151d",
        AXIS="#3a3f4c", CARD="#4a2a3c", CODE_BG="#05060a", CODE_EDGE="#2a2e3a",
        LEAF="logo-dark.png",
        COL={
            "violet": ("#7c5cff", "#17142b", "#b8a9ff"),
            "pink": ("#ff7ab6", "#261320", "#ff9cc9"),
            "amber": ("#ffb547", "#241b0c", "#ffc76e"),
            "grey": ("#6b7080", "#151820", "#c3c7d1"),
        },
    ),
}


def set_theme(name: str) -> None:
    """Set the module-level colours used by the drawing helpers."""
    globals().update(THEMES[name])


set_theme("light")
BRAND = {"mlflow": "#0194E2", "python": "#3776AB", "yaml": "#CB171E"}

parts: list[str] = []


def add(s: str) -> None:
    parts.append(s)


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def icon_uri(name: str) -> str:
    data = (ICONS / f"{name}.svg").read_text()
    if name in BRAND and "fill=" not in data.split(">", 1)[0]:
        data = data.replace("<svg ", f'<svg fill="{BRAND[name]}" ', 1)
    return "data:image/svg+xml;base64," + base64.b64encode(data.encode()).decode()


def leaf_uri() -> str:
    image = Image.open(ROOT / LEAF).convert("RGBA")
    leaf = image.crop((0, 0, image.width, 452))
    buffer = io.BytesIO()
    leaf.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# Primitives -------------------------------------------------------------------
def rect(x, y, w, h, color="grey", fill=True, r=18, lw=2, dash=None):
    accent, tint, _ = COL[color]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{tint if fill else SURFACE}" stroke="{accent}" '
        f'stroke-width="{lw}"{d}/>'
    )


def text(x, y, s, size=15, color=INK, weight=400, anchor="start", family=FONT,
         style=""):
    add(
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
        f'xml:space="preserve" {style}>{esc(s)}</text>'
    )


def heading(x, y, s, color, number=None, size=24):
    accent, _, strong = COL[color]
    if number is not None:
        add(f'<circle cx="{x + 15}" cy="{y - 8}" r="16" fill="{accent}"/>')
        text(x + 15, y - 2, str(number), 15, "#ffffff", 700, "middle")
        x += 42
    text(x, y, s, size, strong, 700)


def chip(x, y, label, color="grey", size=13.5, mono=True):
    accent, _, strong = COL[color]
    w = (0.61 if mono else 0.55) * size * len(label) + 24
    add(
        f'<rect x="{x}" y="{y}" width="{w:.1f}" height="30" rx="15" '
        f'fill="{SURFACE}" stroke="{accent}" stroke-width="1.4"/>'
    )
    text(x + w / 2, y + 20, label, size, strong, 500, "middle",
         MONO if mono else FONT)
    return w


def chips(x, y, labels, color, max_w, size=13.5, gap=8, row=40):
    cx, cy = x, y
    for label in labels:
        w = 0.61 * size * len(label) + 24
        if cx + w > x + max_w:
            cx, cy = x, cy + row
        chip(cx, cy, label, color, size)
        cx += w + gap
    return cy + 30


def image(x, y, w, h, uri):
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}"/>')


def arrow(x1, y, x2, color=MUTED):
    add(
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 12}" y2="{y}" stroke="{color}" '
        f'stroke-width="3.5" stroke-linecap="round"/>'
        f'<path d="M{x2 - 14},{y - 9} L{x2},{y} L{x2 - 14},{y + 9} Z" '
        f'fill="{color}"/>'
    )


# Small illustrations ------------------------------------------------------------
def thumb_frame(x, y, s, color):
    accent, _, _ = COL[color]
    add(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="12" '
        f'fill="{SURFACE}" stroke="{accent}" stroke-width="1.4"/>')


def thumb_landscape(x, y, s):
    """Contour lines of a multimodal function."""
    thumb_frame(x, y, s, "amber")
    cx, cy = x + s / 2, y + s / 2
    for i, r in enumerate([0.08, 0.16, 0.25, 0.34, 0.42]):
        rx, ry = r * s * 1.05, r * s * 0.8
        add(f'<ellipse cx="{cx + 4}" cy="{cy}" rx="{rx}" ry="{ry}" '
            f'fill="none" stroke="#ffb547" stroke-width="1.6" '
            f'opacity="{0.95 - 0.15 * i}"/>')
    for dx, dy in [(-0.3, -0.28), (0.3, 0.26)]:
        add(f'<ellipse cx="{cx + dx * s}" cy="{cy + dy * s}" rx="{0.06 * s}" '
            f'ry="{0.045 * s}" fill="none" stroke="#ffb547" stroke-width="1.4"/>')
    add(f'<circle cx="{cx + 4}" cy="{cy}" r="4" fill="#b8730a"/>')


def thumb_packing(x, y, s):
    thumb_frame(x, y, s, "amber")
    rng = np.random.default_rng(4)
    circles = [(0.27, 0.27, 0.17), (0.7, 0.3, 0.2), (0.3, 0.72, 0.19),
               (0.72, 0.73, 0.17), (0.5, 0.5, 0.09), (0.12, 0.5, 0.07),
               (0.88, 0.52, 0.08), (0.5, 0.12, 0.07), (0.5, 0.88, 0.07)]
    for cx, cy, r in circles:
        a = rng.uniform(0.35, 0.8)
        add(f'<circle cx="{x + cx * s}" cy="{y + cy * s}" r="{r * s}" '
            f'fill="#ffb547" fill-opacity="{a:.2f}" stroke="#b8730a" '
            f'stroke-width="1"/>')


def thumb_route(x, y, s):
    thumb_frame(x, y, s, "amber")
    pts = [(0.15, 0.2), (0.42, 0.14), (0.8, 0.22), (0.86, 0.55), (0.66, 0.82),
           (0.36, 0.86), (0.14, 0.62), (0.4, 0.5), (0.15, 0.2)]
    d = " ".join(f"{'M' if i == 0 else 'L'}{x + px * s:.1f},{y + py * s:.1f}"
                 for i, (px, py) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="#ffb547" stroke-width="2.2" '
        f'stroke-linejoin="round"/>')
    for px, py in pts[:-1]:
        add(f'<circle cx="{x + px * s}" cy="{y + py * s}" r="4.5" '
            f'fill="#b8730a"/>')


def thumb_matrix(x, y, s):
    thumb_frame(x, y, s, "amber")
    n, pad = 4, 0.17
    cell = s * (1 - 2 * pad) / n
    rng = np.random.default_rng(2)
    for i in range(n):
        for j in range(n):
            a = rng.choice([0.15, 0.45, 0.85])
            add(f'<rect x="{x + pad * s + j * cell + 2}" '
                f'y="{y + pad * s + i * cell + 2}" width="{cell - 4}" '
                f'height="{cell - 4}" rx="3" fill="#ffb547" '
                f'fill-opacity="{a}"/>')


def mini_tree(x, y, w, h):
    """An evolution tree with the best lineage highlighted."""
    rng = np.random.default_rng(11)
    gens = 6
    nodes = [(0.5, 0.0, 0.1, None)]
    layer = [0]
    for g in range(1, gens):
        new = []
        for _ in range(int(rng.integers(2, 4))):
            p = int(rng.choice(layer))
            px, _, pf, _ = nodes[p]
            nodes.append((float(np.clip(px + rng.normal(0, 0.16), 0.06, 0.94)),
                          g / (gens - 1),
                          float(np.clip(pf + rng.normal(0.12, 0.1), 0, 1)), p))
            new.append(len(nodes) - 1)
        layer = new
    best = max(range(len(nodes)), key=lambda i: nodes[i][2])
    path, i = set(), best
    while i is not None:
        path.add(i)
        i = nodes[i][3]

    def pos(n):
        return x + n[0] * w, y + h - n[1] * h

    for i, n in enumerate(nodes):
        if n[3] is None:
            continue
        (x1, y1), (x2, y2) = pos(nodes[n[3]]), pos(n)
        on = i in path and n[3] in path
        add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{"#ff7ab6" if on else "#b7aaf5"}" '
            f'stroke-width="{3 if on else 1.4}" stroke-linecap="round"/>')
    for i, n in enumerate(nodes):
        cx, cy = pos(n)
        fill = "#ffb547" if i == best else ("#ff7ab6" if i in path else "#7c5cff")
        add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{7 if i == best else 5}" '
            f'fill="{fill}" stroke="{SURFACE}" stroke-width="1.5"/>')


def mini_convergence(x, y, w, h):
    add(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" '
        f'stroke="{AXIS}" stroke-width="1.4"/>'
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="{AXIS}" '
        f'stroke-width="1.4"/>')
    for k, (color, rate) in enumerate([("#7c5cff", 3.6), ("#ff7ab6", 2.4),
                                       ("#ffb547", 1.5)]):
        rng = np.random.default_rng(k)
        t = np.linspace(0, 1, 24)
        v = np.maximum.accumulate(1 - np.exp(-rate * t) + rng.normal(0, .02, 24))
        v = v / 1.05
        d = " ".join(f"{'M' if i == 0 else 'L'}{x + ti * w:.1f},"
                     f"{y + h - vi * h:.1f}" for i, (ti, vi) in enumerate(zip(t, v)))
        add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.6" '
            f'stroke-linejoin="round"/>')


def mini_boxes(x, y, w, h):
    for k, (color, lo, q1, med, q3, hi) in enumerate([
        ("#7c5cff", .55, .68, .76, .84, .93), ("#ff7ab6", .35, .5, .6, .7, .8),
        ("#ffb547", .2, .33, .42, .52, .66),
    ]):
        cx = x + (k + 0.5) * w / 3
        bw = w / 3 * 0.5

        def Y(v):
            return y + h - v * h

        add(f'<line x1="{cx}" y1="{Y(lo)}" x2="{cx}" y2="{Y(hi)}" '
            f'stroke="{color}" stroke-width="1.6"/>'
            f'<rect x="{cx - bw / 2}" y="{Y(q3)}" width="{bw}" '
            f'height="{Y(q1) - Y(q3)}" rx="3" fill="{color}" fill-opacity=".25" '
            f'stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="{cx - bw / 2}" y1="{Y(med)}" x2="{cx + bw / 2}" '
            f'y2="{Y(med)}" stroke="{color}" stroke-width="2.4"/>')


def mini_podium(x, y, w, h):
    bars = [(0.35, "#ff7ab6", "2"), (0.5, "#ffb547", "1"), (0.2, "#7c5cff", "3")]
    bw = w / 3
    order = [0, 1, 2]
    heights = [0.62, 0.95, 0.42]
    for k in order:
        bh = heights[k] * h
        color = bars[k][1]
        add(f'<rect x="{x + k * bw + 4}" y="{y + h - bh}" width="{bw - 8}" '
            f'height="{bh}" rx="4" fill="{color}" fill-opacity=".85"/>')
        text(x + k * bw + bw / 2, y + h - bh + 22, bars[k][2], 17, "#ffffff",
             700, "middle")


# Line icons (24x24 grid, scaled) -------------------------------------------------
LINE_ICONS = {
    "box": "M4 8l8-4 8 4v8l-8 4-8-4zM4 8l8 4 8-4M12 12v8",
    "clock": "M12 3a9 9 0 1 0 0.01 0zM12 7v5l3 3",
    "seeds": "M5 5h6v6H5zM13 13h6v6h-6zM7 7.5h.01M9 9h.01M15 15h.01M17 17h.01"
             "M15 17h.01M17 15h.01",
    "parallel": "M4 6h16M4 12h16M4 18h16M17 3l3 3-3 3M17 9l3 3-3 3M17 15l3 3-3 3",
    "gauge": "M4 16a8 8 0 1 1 16 0M12 16l4-5",
    "tune": "M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4",
    "db": "M5 6c0-1.7 3.1-3 7-3s7 1.3 7 3-3.1 3-7 3-7-1.3-7-3zM5 6v12c0 1.7 "
          "3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3",
    "file": "M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6",
    "hash": "M5 9h14M5 15h14M10 4l-2 16M16 4l-2 16",
    "terminal": "M4 5h16v14H4zM7 10l3 2-3 2M12 15h5",
    "robot": "M7 8h10v9H7zM12 4v4M9.5 12h.01M14.5 12h.01M10 15h4M4 11v3M20 11v3",
}


def line_icon(name, x, y, s=30, color=MUTED, lw=1.8):
    scale = s / 24
    add(f'<g transform="translate({x},{y}) scale({scale})" fill="none" '
        f'stroke="{color}" stroke-width="{lw / scale}" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="{LINE_ICONS[name]}"/></g>')


def icon_row(x, y, name, label, color, size=16):
    line_icon(name, x, y - 22, 28, COL[color][2])
    text(x + 40, y, label, size, INK)


# Figure -------------------------------------------------------------------------
def build() -> str:
    parts.clear()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">')
    add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    # Header.
    image(40, 28, 52, 58, leaf_uri())
    text(108, 72, "BLADE", 42, INK, 800)
    text(262, 72, "Benchmarking LLM-driven Automated Design and Evolution "
         "of iterative optimisation heuristics", 19, MUTED)

    top, bottom = 120, 960
    cols = [(40, 400, "Define", "grey"), (490, 920, "Components", "violet"),
            (1460, 290, "Run", "amber"), (1800, 260, "Record", "grey"),
            (2110, 250, "Analyse", "pink")]
    for i, (x, w, name, color) in enumerate(cols):
        rect(x, top, w, bottom - top, color, fill=(color != "violet"), r=22,
             lw=2.2)
        heading(x + 24, top + 48, name, color, number=i + 1)
        if i < len(cols) - 1:
            arrow(x + w + 6, (top + bottom) / 2, cols[i + 1][0] - 6)

    # 1 Define: interfaces and a YAML card.
    x0 = 40
    image(x0 + 28, top + 82, 30, 30, icon_uri("yaml"))
    text(x0 + 70, top + 104, "YAML experiment", 17, INK, 500)
    line_icon("terminal", x0 + 28, top + 128, 30, COL["grey"][2])
    text(x0 + 70, top + 150, "iohblade CLI", 17, INK, 500)
    text(x0 + 70, top + 174, "init · validate · check · run", 14, MUTED,
         family=MONO)
    image(x0 + 28, top + 194, 30, 30, icon_uri("python"))
    text(x0 + 70, top + 216, "Python API", 17, INK, 500)
    line_icon("robot", x0 + 28, top + 240, 30, COL["grey"][2])
    text(x0 + 70, top + 262, "Coding-agent skill", 17, INK, 500)
    cy = top + 300
    add(f'<rect x="{x0 + 22}" y="{cy}" width="356" height="500" rx="14" '
        f'fill="{CODE_BG}" stroke="{CODE_EDGE}" stroke-width="1.5"/>')
    for k, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        add(f'<circle cx="{x0 + 46 + k * 20}" cy="{cy + 22}" r="6" fill="{c}"/>')
    text(x0 + 120, cy + 27, "experiment.yml", 13, "#8f86c9", family=MONO)
    yaml = [
        ("experiment:", "#b8a9ff"), ("  runs: 5", "#e6e6f0"),
        ("  budget: 100", "#e6e6f0"), ("llms:", "#b8a9ff"),
        ("  qwen:", "#e6e6f0"), ("    provider: openai", "#ffb547"),
        ("    model: Qwen3-32B", "#ffb547"), ("methods:", "#b8a9ff"),
        ("  - type: llamea", "#ff7ab6"), ("  - type: shinka_evolve", "#ff7ab6"),
        ("  - type: funsearch", "#ff7ab6"), ("problems:", "#b8a9ff"),
        ("  - suite: full", "#8ee6c0"), ("    params:", "#e6e6f0"),
        ("      dim: 10", "#e6e6f0"),
    ]
    for k, (line, color) in enumerate(yaml):
        text(x0 + 42, cy + 68 + k * 28, line, 15, color, family=MONO)

    # 2 Components.
    x0, w0 = cols[1][0], cols[1][1]
    sw = (w0 - 3 * 24) / 2
    lx, rx = x0 + 24, x0 + 48 + sw
    ty, th = top + 80, 380
    by, bh = ty + th + 24, bottom - 64 - (ty + th + 24)

    # Problems.
    rect(lx, ty, sw, th, "amber", r=16)
    heading(lx + 22, ty + 40, "Problems", "amber", size=20)
    s = 78
    thumbs = [(thumb_landscape, "Continuous"), (thumb_packing, "Packing"),
              (thumb_route, "Routing"), (thumb_matrix, "Matrices")]
    for k, (fn, label) in enumerate(thumbs):
        tx = lx + 30 + k * ((sw - 60 - s) / 3)
        fn(tx, ty + 62, s)
        text(tx + s / 2, ty + 62 + s + 24, label, 13.5, INK, 500, "middle")
    rows = ["BBOB function groups · MA-BBOB",
            "AlphaEvolve mathematics (17 problems)",
            "Steiner · TSP · CVRP · graph colouring",
            "Photonics · AutoML · kernel tuning",
            "+ your own Problem class"]
    for k, line in enumerate(rows):
        text(lx + 22, ty + 208 + k * 34, line, 16, INK if k < 4 else MUTED)

    # Methods.
    rect(rx, ty, sw, th, "violet", r=16)
    heading(rx + 22, ty + 40, "Methods", "violet", size=20)
    mini_tree(rx + sw - 175, ty + 26, 150, 120)
    mchips = ["LLaMEA", "EoH", "ReEvo", "FunSearch", "ShinkaEvolve", "LHNS",
              "MCTS-AHD", "MoEH", "Random search"]
    text(rx + 22, ty + 78, "LLM-driven algorithm", 16, INK)
    text(rx + 22, ty + 102, "design methods", 16, INK)
    end = chips(rx + 22, ty + 165, mchips, "violet", sw - 44, size=14)
    text(rx + 22, end + 36, "+ your own Method class", 16, MUTED)

    # LLMs.
    rect(lx, by, sw, bh, "pink", r=16)
    heading(lx + 22, by + 40, "LLMs", "pink", size=20)
    logos = [("openai", "OpenAI"), ("claude-color", "Claude"),
             ("gemini-color", "Gemini"), ("deepseek-color", "DeepSeek"),
             ("qwen-color", "Qwen"), ("ollama", "Ollama"),
             ("vllm-color", "vLLM"), ("lmstudio", "LM Studio")]
    per_row, gx = 4, (sw - 44) / 4
    for k, (name, label) in enumerate(logos):
        cx = lx + 22 + (k % per_row) * gx + gx / 2
        cy = by + 70 + (k // per_row) * 96
        add(f'<circle cx="{cx}" cy="{cy + 22}" r="30" fill="#ffffff" '
            f'stroke="#ffd0e4" stroke-width="1.4"/>')
        image(cx - 18, cy + 4, 36, 36, icon_uri(name))
        text(cx, cy + 74, label, 13.5, INK, 500, "middle")
    text(lx + 22, by + bh - 24, "hosted APIs or local servers · multi: "
         "combine several", 14, MUTED)

    # Suites.
    rect(rx, by, sw, bh, "amber", r=16)
    heading(rx + 22, by + 40, "Default suites", "amber", size=20)
    end = chips(rx + 22, by + 62, ["quick", "continuous_bbo", "alphaevolve_math",
                                    "combinatorial", "full"], "amber", sw - 44,
                size=14)
    text(rx + 22, end + 38, "Fixed instances and settings per suite;", 15.5, INK)
    text(rx + 22, end + 62, "dim · budget_factor · use_best for all problems",
         15.5, INK)
    text(rx + 22, by + bh - 24, "BBOB groups: train iid 1-5, test iid 6-15",
         14, MUTED)
    text(x0 + w0 / 2, bottom - 26, "built-in by registry key, or your own class",
         15, MUTED, anchor="middle")

    # 3 Run.
    x0 = cols[2][0]
    items = [("seeds", "Seeds and repeats"),
             ("parallel", "Parallel runs (n_jobs)"),
             ("box", "Isolated, cached"), ("", "evaluation envs"),
             ("gauge", "Exact budgets"), ("clock", "Timeouts · harness checks"),
             ("tune", "HPO with SMAC")]
    yy = top + 110
    for name, label in items:
        if name:
            icon_row(x0 + 24, yy, name, label, "amber")
            yy += 62
        else:
            text(x0 + 64, yy - 36, label, 16, INK)
    # A small run timeline.
    add(f'<text x="{x0 + 24}" y="{bottom - 178}" font-family="{FONT}" '
        f'font-size="14" fill="{MUTED}">each generation</text>')
    steps = [("LLM", "#ff7ab6"), ("code", "#7c5cff"), ("eval", "#ffb547")]
    for k, (label, color) in enumerate(steps):
        bx = x0 + 24 + k * 82
        add(f'<rect x="{bx}" y="{bottom - 160}" width="72" height="34" rx="17" '
            f'fill="{color}" fill-opacity=".9"/>')
        text(bx + 36, bottom - 137, label, 14, "#ffffff", 700, "middle")
    text(x0 + 24, bottom - 96, "× budget, per run", 14, MUTED)

    # 4 Record.
    x0 = cols[3][0]
    icon_row(x0 + 24, top + 110, "file", "File logger", "grey")
    image(x0 + 24, top + 150, 28, 28, icon_uri("mlflow"))
    text(x0 + 64, top + 172, "MLflow", 16, INK)
    icon_row(x0 + 24, top + 234, "db", "Trackio", "grey")
    icon_row(x0 + 24, top + 296, "hash", "Configs, sources", "grey")
    text(x0 + 64, top + 320, "& hashes", 16, INK)
    icon_row(x0 + 24, top + 382, "terminal", "Prompts, code", "grey")
    text(x0 + 64, top + 406, "& feedback", 16, INK)
    text(x0 + 24, bottom - 60, "everything needed to", 14, MUTED)
    text(x0 + 24, bottom - 38, "reproduce a run", 14, MUTED)

    # 5 Analyse.
    x0 = cols[4][0]
    blocks = [(mini_convergence, "Convergence & test scores"),
              (mini_boxes, "Compare methods"),
              (mini_tree, "Code evolution"),
              (mini_podium, "Leaderboard (coming)")]
    yy = top + 84
    for fn, label in blocks:
        add(f'<rect x="{x0 + 20}" y="{yy}" width="{cols[4][1] - 40}" '
            f'height="150" rx="12" fill="{SURFACE}" stroke="{CARD}"/>')
        fn(x0 + 44, yy + 16, cols[4][1] - 88, 92)
        text(x0 + cols[4][1] / 2, yy + 136, label, 14.5, INK, 500, "middle")
        yy += 166
    text(x0 + cols[4][1] / 2, bottom - 24, "web app · behaviour metrics",
         14, MUTED, anchor="middle")

    add("</svg>")
    return "\n".join(parts)


def main() -> None:
    for theme, path in (("light", OUT), ("dark", OUT_DARK)):
        set_theme(theme)
        path.write_text(build())
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
