"""Key frames (first and last) for the video intro and outro, for Veo.

Style "Convergence": four best-so-far curves climb from the bottom left and
meet in the BLADE leaf; the outro dims them behind an end card.

Frames (1920x1080): ``intro-first``, ``intro-last`` (also the outro's first
frame, so intro and outro match) and ``outro-last``.

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


# ------------------------------------------------------------ frames ---
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
        f.line(x0, yy, x1, yy, "#ffffff", 1, 0.07 * opacity)
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
    plot_frame(f, 1.0, opacity=0.07)
    f.end_card()
    return f


FRAMES = {
    "intro-first": c_intro_first,
    "intro-last": c_logo,  # also the outro's first frame
    "outro-last": c_outro_last,
}


def main() -> None:
    out = HERE / "frames"
    out.mkdir(exist_ok=True)
    for name, fn in FRAMES.items():
        (out / f"{name}.svg").write_text(fn().svg(), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
