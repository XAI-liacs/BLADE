"""Key frames (first and last) for the video intro and outro, for Veo.

Three styles, four frames each (1920x1080):

* ``a`` "Constellation": scattered points (generated algorithms) that
  gather into the BLADE leaf.
* ``b`` "Code to leaf": an experiment file that turns into the logo.
* ``c`` "Convergence": best-so-far curves that rise and meet in the logo.

Frames: ``<style>-intro-first``, ``<style>-intro-last`` (= the outro's
first frame, so intro and outro match), ``<style>-outro-last``.

Usage (from the repository root)::

    python docs/videos/intro-outro/make_frames.py
    NODE_PATH=$(npm root -g) node docs/videos/intro-outro/render.js
"""

from __future__ import annotations

import base64
import io
import math
import random
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W, H = 1920, 1080

BG = "#0b0d12"
VIOLET, PINK, AMBER, MINT, SOFT = "#7c5cff", "#ff7ab6", "#ffb547", "#8ee6c0", "#b8a9ff"
PALETTE = [VIOLET, PINK, AMBER, SOFT, MINT]
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace"
TAGLINE = "Benchmarking LLM-driven automated algorithm design"
LINKS = [
    ("github.com/XAI-liacs/BLADE", SOFT),
    ("pip install iohblade", AMBER),
    ("xai-liacs.github.io/BLADE", MINT),
]


# ----------------------------------------------------------------- assets ---
def _leaf_image() -> Image.Image:
    image = Image.open(ROOT / "docs" / "_static" / "logo-dark.png").convert("RGBA")
    leaf = image.crop((0, 0, image.width, 452))
    return leaf.crop(leaf.getbbox())


LEAF = _leaf_image()


def leaf_uri() -> str:
    buffer = io.BytesIO()
    LEAF.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def leaf_points(n: int, x: float, y: float, height: float, seed: int = 1):
    """``n`` points inside the leaf silhouette, scaled to ``height``."""
    rng = random.Random(seed)
    alpha = LEAF.getchannel("A")
    w, h = LEAF.size
    scale = height / h
    points = []
    while len(points) < n:
        px, py = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        if alpha.getpixel((int(px), int(py))) > 160:
            points.append((x + px * scale, y + py * scale))
    return points


# ----------------------------------------------------------- primitives ---
class Frame:
    def __init__(self):
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def svg(self) -> str:
        defs = f"""<defs>
<radialGradient id="glowV" cx="85%" cy="5%" r="70%"><stop offset="0" stop-color="{VIOLET}" stop-opacity=".40"/><stop offset="1" stop-color="{VIOLET}" stop-opacity="0"/></radialGradient>
<radialGradient id="glowP" cx="5%" cy="100%" r="60%"><stop offset="0" stop-color="{PINK}" stop-opacity=".22"/><stop offset="1" stop-color="{PINK}" stop-opacity="0"/></radialGradient>
<radialGradient id="dot" r="50%"><stop offset="0" stop-color="#fff" stop-opacity=".9"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
<linearGradient id="grad" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{VIOLET}"/><stop offset=".55" stop-color="{PINK}"/><stop offset="1" stop-color="{AMBER}"/></linearGradient>
<filter id="blur8"><feGaussianBlur stdDeviation="8"/></filter>
<filter id="blur3"><feGaussianBlur stdDeviation="3"/></filter>
</defs>"""
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">{defs}'
            f'<rect width="{W}" height="{H}" fill="{BG}"/>'
            f'<rect width="{W}" height="{H}" fill="url(#glowV)"/>'
            f'<rect width="{W}" height="{H}" fill="url(#glowP)"/>{body}</svg>'
        )

    def text(self, x, y, s, size, color="#ffffff", weight=400, anchor="start",
             family=FONT, spacing=0, opacity=1.0):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.add(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{spacing}" opacity="{opacity}" '
            f'xml:space="preserve">{s}</text>'
        )

    def glow_dot(self, x, y, r, color, opacity=1.0):
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r * 3:.1f}" fill="{color}" '
            f'opacity="{0.18 * opacity:.2f}" filter="url(#blur8)"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'opacity="{opacity:.2f}"/>'
        )

    def line(self, x1, y1, x2, y2, color, width=1.2, opacity=0.3):
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity:.2f}"/>'
        )

    def leaf(self, x, y, height, opacity=1.0):
        width = height * LEAF.width / LEAF.height
        self.add(
            f'<image href="{leaf_uri()}" x="{x}" y="{y}" width="{width:.1f}" '
            f'height="{height}" opacity="{opacity}"/>'
        )
        return width

    def wordmark(self, x, y, size=150, tagline=True, anchor="start"):
        """``BLADE`` and the gradient tagline; (x, y) is the baseline."""
        self.text(x, y, "BLADE", size, weight=800, anchor=anchor, spacing=size * 0.04)
        if tagline:
            self.text(x, y + size * 0.42, TAGLINE, size * 0.22, color="url(#grad)",
                      weight=600, anchor=anchor)

    def end_card(self, cy=560):
        """Centred leaf, wordmark and links."""
        leaf_h = 120
        self.leaf(W / 2 - leaf_h * LEAF.width / LEAF.height / 2, cy - 330, leaf_h)
        self.wordmark(W / 2, cy - 70, size=110, anchor="middle")
        for k, (s, color) in enumerate(LINKS):
            self.text(W / 2, cy + 90 + 62 * k, s, 38, color=color, weight=500,
                      anchor="middle", family=MONO)


def scatter(n, seed, margin=60):
    rng = random.Random(seed)
    return [(rng.uniform(margin, W - margin), rng.uniform(margin, H - margin))
            for _ in range(n)]


def neighbours(points, k=2):
    pairs = set()
    for i, (x, y) in enumerate(points):
        near = sorted(range(len(points)),
                      key=lambda j: (points[j][0] - x) ** 2 + (points[j][1] - y) ** 2)
        for j in near[1:k + 1]:
            pairs.add((min(i, j), max(i, j)))
    return pairs


def draw_graph(f, points, colours, radius, line_opacity, dot_opacity=1.0, k=2):
    for i, j in neighbours(points, k):
        f.line(*points[i], *points[j], colours[i], 1.2, line_opacity)
    for (x, y), c in zip(points, colours):
        f.glow_dot(x, y, radius, c, dot_opacity)


# -------------------------------------------------- style A: constellation ---
N_DOTS = 260
DOT_COLOURS = [PALETTE[i % 4] for i in range(N_DOTS)]
LEAF_X, LEAF_Y, LEAF_H = 330, 250, 560


def a_intro_first():
    f = Frame()
    draw_graph(f, scatter(N_DOTS, 3), DOT_COLOURS, 3.2, 0.18)
    return f


def a_logo():
    f = Frame()
    pts = leaf_points(N_DOTS, LEAF_X, LEAF_Y, LEAF_H, seed=5)
    f.leaf(LEAF_X, LEAF_Y, LEAF_H, opacity=0.10)  # the shape the points fill
    draw_graph(f, pts, DOT_COLOURS, 3.4, 0.30, k=3)
    f.wordmark(870, 590, size=170)
    return f


def a_outro_last():
    f = Frame()
    draw_graph(f, scatter(N_DOTS, 11), DOT_COLOURS, 2.6, 0.08, dot_opacity=0.35)
    f.end_card()
    return f


# ------------------------------------------------ style B: code to leaf ---
YAML = [
    ("experiment: {output_dir: results/demo, runs: 5, budget: 100}", MINT),
    ("llms:", SOFT),
    ('  qwen: {provider: ollama, model: "qwen2.5-coder:14b"}', AMBER),
    ("methods:", SOFT),
    ('  - {type: llamea, name: "LLaMEA-{llm}", llm: qwen}', PINK),
    ('  - {type: shinka_evolve, llm: qwen}', PINK),
    ('  - {type: random_search, llm: qwen}', PINK),
    ("problems:", SOFT),
    ("  - {suite: bbob_groups, params: {dim: 10}}", "#e6e6f0"),
    ("  - {suite: alphaevolve_math}", "#e6e6f0"),
]


def editor(f, x, y, w, h, opacity=1.0, title="experiment.yml", lines=YAML,
           size=30, blur=False):
    flt = ' filter="url(#blur3)"' if blur else ""
    f.add(f'<g opacity="{opacity}"{flt}>')
    f.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" '
          f'fill="#05060a" stroke="#2a2e3d" stroke-width="2"/>')
    for k, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        f.add(f'<circle cx="{x + 34 + 28 * k}" cy="{y + 34}" r="9" fill="{c}"/>')
    f.text(x + 130, y + 43, title, 24, color="#8c90a3", family=MONO)
    for k, (s, color) in enumerate(lines):
        f.text(x + 44, y + 110 + k * size * 1.55, s, size, color=color, family=MONO)
    f.add("</g>")


def b_intro_first():
    f = Frame()
    editor(f, 260, 170, 1400, 740)
    f.add(f'<rect x="{304 + 30 * 18.1:.0f}" y="{170 + 110 + 9 * 46.5 - 30:.0f}" '
          f'width="16" height="36" fill="{SOFT}" opacity=".9"/>')  # cursor
    return f


def b_logo():
    f = Frame()
    # the code, faded and enlarged in the background
    editor(f, 120, 60, 1680, 960, opacity=0.07, size=36, blur=True)
    lw = 240 * LEAF.width / LEAF.height
    f.leaf(W / 2 - lw / 2, 190, 240)
    f.wordmark(W / 2, 640, size=180, anchor="middle")
    return f


def b_outro_last():
    f = Frame()
    terminal = [
        ("$ pip install iohblade", AMBER),
        ("$ iohblade init my-exp", "#e6e6f0"),
        ("$ iohblade check my-exp/experiment.yml", "#e6e6f0"),
        ("  3 ok - ready to run.", MINT),
    ]
    editor(f, 150, 300, 820, 420, title="terminal", lines=terminal, size=30)
    lw = 110 * LEAF.width / LEAF.height
    f.leaf(1060, 280, 110)
    f.text(1060 + lw + 30, 370, "BLADE", 110, weight=800, spacing=4)
    f.text(1060, 470, TAGLINE, 30, color="url(#grad)", weight=600)
    for k, (s, color) in enumerate(LINKS):
        f.text(1060, 580 + 60 * k, s, 36, color=color, weight=500, family=MONO)
    return f


# ------------------------------------------------- style C: convergence ---
PLOT = (220, 330, 1700, 960)  # x0, y0, x1, y1
# Per curve: convergence rate and how far below the target it levels off
# before the final jump, so the four methods look different.
CURVES = {1: (6.0, 0.0), 2: (2.6, 0.0), 3: (4.0, 0.0), 4: (1.6, 0.0)}


def curve(seed, frac, target):
    """Best-so-far curve from the bottom left towards ``target``."""
    rng = random.Random(seed)
    rate, _ = CURVES[seed]
    x0, y0, x1, y1 = PLOT
    start = y1 - 30 - 40 * (seed % 2)
    pts, best = [], start
    steps = 70
    for k in range(int(steps * frac) + 1):
        t = k / steps
        x = x0 + t * (target[0] - x0)
        level = start - (start - target[1]) * (1 - math.exp(-rate * t)) / (1 - math.exp(-rate))
        best = min(best, level + rng.uniform(0, 60) * (1 - t) ** 2)
        if rng.random() < 0.55 and k:  # best-so-far: flat between improvements
            best = pts[-1][1]
        pts.append((x, best))
    pts[-1] = (pts[-1][0], target[1]) if frac >= 1 else pts[-1]
    return pts


def plot_frame(f, frac, opacity=1.0, target=(1420, 420)):
    x0, y0, x1, y1 = PLOT
    for k in range(1, 6):
        yy = y0 + k * (y1 - y0) / 6
        f.line(x0, yy, x1, yy, "#ffffff", 1, 0.05 * opacity)
    f.line(x0, y1, x1, y1, "#ffffff", 2, 0.25 * opacity)
    f.line(x0, y0, x0, y1, "#ffffff", 2, 0.25 * opacity)
    for seed, color in zip((1, 2, 3, 4), (VIOLET, PINK, AMBER, MINT)):
        pts = curve(seed, frac, target)
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        f.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="10" '
              f'opacity="{0.25 * opacity:.2f}" filter="url(#blur8)"/>')
        f.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4" '
              f'stroke-linecap="round" opacity="{opacity:.2f}"/>')
        f.glow_dot(*pts[-1], 7, color, opacity)


def c_intro_first():
    f = Frame()
    plot_frame(f, 0.12)
    return f


def c_logo():
    f = Frame()
    plot_frame(f, 1.0)
    # the curves meet in the leaf
    lw = 230 * LEAF.width / LEAF.height
    f.add(f'<circle cx="1420" cy="420" r="170" fill="{VIOLET}" opacity=".22" filter="url(#blur8)"/>')
    f.leaf(1420 - lw / 2, 305, 230)
    f.wordmark(240, 230, size=150)
    return f


def c_outro_last():
    f = Frame()
    plot_frame(f, 1.0, opacity=0.10)
    f.end_card()
    return f


FRAMES = {
    "a-intro-first": a_intro_first,
    "a-intro-last": a_logo,
    "a-outro-last": a_outro_last,
    "b-intro-first": b_intro_first,
    "b-intro-last": b_logo,
    "b-outro-last": b_outro_last,
    "c-intro-first": c_intro_first,
    "c-intro-last": c_logo,
    "c-outro-last": c_outro_last,
}


def main() -> None:
    out = HERE / "frames"
    out.mkdir(exist_ok=True)
    for name, fn in FRAMES.items():
        (out / f"{name}.svg").write_text(fn().svg(), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
