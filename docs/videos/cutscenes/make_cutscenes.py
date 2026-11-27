"""Cutscenes (1920x1080 stills) for the tutorial videos, in the style of
the intro and outro (``../intro-outro``).

- ``00-blank``: only the background.
- ``bench-*``: one per benchmark family, an illustration on the left and
  its registry keys, name and tagline on the right. The BBOB scenes use
  real 2-D landscapes computed with ``ioh``.
- ``ceg``, ``convergence``: the analysis plots (illustrative data).
- ``framework``, ``framework-detailed``, ``aad-loop``: how BLADE and
  LLM-driven algorithm design work.
- ``episode-N``, ``methods``, ``lower-third`` (transparent).

Usage (from the repository root)::

    uv run python docs/videos/cutscenes/make_cutscenes.py
    NODE_PATH=$(npm root -g) node docs/videos/cutscenes/render.js
"""

from __future__ import annotations

import base64
import importlib.util
import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import landscapes as L  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "make_frames", HERE.parent / "intro-outro" / "make_frames.py"
)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

Frame, W, H = M.Frame, M.W, M.H
VIOLET, PINK, AMBER, MINT, SOFT = M.VIOLET, M.PINK, M.AMBER, M.MINT, M.SOFT
FONT, MONO = M.FONT, M.MONO
MUTED = "#9aa0b4"
SURFACE = "#12151d"
EDGE = "#2a2e3a"
PALETTE = [VIOLET, PINK, AMBER, MINT, SOFT]

_ids = itertools.count()


def uid(prefix="c"):
    return f"{prefix}{next(_ids)}"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


# ------------------------------------------------------------- helpers ---
def card(f, x, y, w, h, stroke=EDGE, fill=SURFACE, r=24, width=2, opacity=1):
    f.add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
        f'opacity="{opacity}"/>'
    )


def pill(f, x, y, label, color=AMBER, size=26):
    w = 0.6 * size * len(label) + 2 * size * 0.8
    h = size * 1.9
    f.add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{h / 2:.1f}" fill="{color}" fill-opacity=".08" stroke="{color}" '
        f'stroke-width="2"/>'
    )
    f.text(
        x + w / 2, y + h * 0.66, label, size, color=color, anchor="middle", family=MONO
    )
    return w, h


def image(f, x, y, w, h, uri, r=0, opacity=1):
    clip = ""
    if r:
        cid = uid("clip")
        f.add(
            f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" '
            f'height="{h}" rx="{r}"/></clipPath>'
        )
        clip = f' clip-path="url(#{cid})"'
    f.add(
        f'<image href="{uri}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid slice" opacity="{opacity}"{clip}/>'
    )


def path(points, close=False):
    d = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points)
    )
    return d + (" Z" if close else "")


def arrow_defs(f):
    f.add(
        "<defs>"
        + "".join(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="22" markerHeight="22" markerUnits="userSpaceOnUse" '
            f'orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 Z" fill="{color}"/></marker>'
            for name, color in [
                ("v", VIOLET),
                ("p", PINK),
                ("a", AMBER),
                ("m", MINT),
                ("s", SOFT),
                ("g", MUTED),
            ]
        )
        + "</defs>"
    )


MARK = {VIOLET: "v", PINK: "p", AMBER: "a", MINT: "m", SOFT: "s", MUTED: "g"}


def brand(f, x=1585, y=1000, size=40):
    lw = f.leaf(x, y - size * 1.05, size * 1.25)
    f.text(x + lw + 16, y, "BLADE", size, weight=800, spacing=2)


def title(f, s, sub=None, x=150, y=150, size=60):
    f.text(x, y, s, size, weight=800)
    if sub:
        f.text(x, y + size * 0.85, sub, size * 0.45, color=MUTED)


def colour_at(t):
    """Fitness colour: violet (low) - pink - amber (high)."""
    stops = [(0, (124, 92, 255)), (0.55, (255, 122, 182)), (1, (255, 181, 71))]
    t = min(max(t, 0), 1)
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            r, g, b = (round(a + (b_ - a) * u) for a, b_ in zip(c0, c1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return AMBER


# ---------------------------------------------------- benchmark visuals ---
def v_bbob(f, x, y, s):
    cols, rows, gap = 6, 4, 10
    cell = (s - (cols - 1) * gap) / cols
    top = y + (s - rows * cell - (rows - 1) * gap) / 2
    for fid in range(1, 25):
        i, j = divmod(fid - 1, cols)
        cx, cy = x + j * (cell + gap), top + i * (cell + gap)
        image(
            f,
            cx,
            cy,
            cell,
            cell,
            L.bbob(fid, px=200, n=110, contours=6, mark=False),
            r=10,
        )
        f.text(cx + 8, cy + 24, f"f{fid}", 18, weight=700, opacity=0.9)


GROUPS = [
    (3, "1 · separable"),
    (8, "2 · moderate cond."),
    (11, "3 · high cond."),
    (16, "4 · multimodal"),
    (22, "5 · weak structure"),
]


def v_bbob_groups(f, x, y, s):
    gap = 14
    big = (s - gap) / 2
    small = (s - 2 * gap) / 3
    slots = [(x, y, big), (x + big + gap, y, big)] + [
        (x + k * (small + gap), y + big + gap, small) for k in range(3)
    ]
    for (fid, label), (cx, cy, size) in zip(GROUPS, slots):
        image(f, cx, cy, size, size, L.bbob(fid, px=320, n=150, contours=10), r=14)
        f.add(
            f'<rect x="{cx}" y="{cy + size - 44}" width="{size}" height="44" '
            f'fill="#0b0d12" opacity=".6"/>'
        )
        f.text(cx + 14, cy + size - 14, label, 19 if size < 250 else 22, weight=600)


def v_ma_bbob(f, x, y, s):
    image(f, x, y, s, s, L.ma_bbob(21, px=700, n=280, contours=6), r=18)


def v_packing(f, x, y, s):
    rng = np.random.default_rng(5)
    circles = []
    for _ in range(60):
        pts = rng.uniform(0, 1, (8000, 2))
        r = np.minimum.reduce(
            [
                pts[:, 0],
                pts[:, 1],
                1 - pts[:, 0],
                1 - pts[:, 1],
                np.full(len(pts), 0.15),
            ]
        )
        for cx, cy, cr in circles:
            r = np.minimum(r, np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - cr)
        k = int(r.argmax())
        circles.append((pts[k, 0], pts[k, 1], r[k] - 0.006))
    card(f, x, y, s, s, stroke=AMBER, fill="none", r=6, width=3)
    for k, (cx, cy, r) in enumerate(circles):
        color = PALETTE[k % 4]
        f.add(
            f'<circle cx="{x + cx * s:.1f}" cy="{y + cy * s:.1f}" '
            f'r="{r * s:.1f}" fill="{color}" fill-opacity=".28" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )


def v_heilbronn(f, x, y, s):
    a, b, c = (x + s / 2, y + 30), (x + 10, y + s - 40), (x + s - 10, y + s - 40)
    f.add(
        f'<path d="{path([a, b, c], True)}" fill="{VIOLET}" fill-opacity=".08" '
        f'stroke="{SOFT}" stroke-width="3"/>'
    )
    rng = random.Random(3)
    pts = []
    while len(pts) < 11:
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        p = (
            a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
            a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
        )
        if all(math.dist(p, q) > s * 0.13 for q in pts):
            pts.append(p)

    def area(t):
        (x1, y1), (x2, y2), (x3, y3) = t
        return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2

    triples = sorted(itertools.combinations(pts, 3), key=area)
    for t in triples[1:6]:
        f.add(
            f'<path d="{path(t, True)}" fill="none" stroke="{SOFT}" '
            f'stroke-width="1.2" opacity=".35"/>'
        )
    f.add(
        f'<path d="{path(triples[0], True)}" fill="{PINK}" fill-opacity=".45" '
        f'stroke="{PINK}" stroke-width="3"/>'
    )
    for p in pts:
        f.glow_dot(*p, 10, AMBER)


def v_spheres(f, x, y, s):
    cx, cy, R = x + s / 2, y + s / 2, s * 0.45
    f.add(
        f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="{VIOLET}" fill-opacity=".1" '
        f'stroke="{SOFT}" stroke-width="2.5"/>'
    )
    for k in (-0.6, -0.2, 0.2, 0.6):
        f.add(
            f'<ellipse cx="{cx}" cy="{cy + k * R}" rx="{R * math.sqrt(1 - k * k)}" '
            f'ry="{R * 0.22 * math.sqrt(1 - k * k)}" fill="none" stroke="{SOFT}" '
            f'stroke-width="1" opacity=".25"/>'
        )
    for k in (0.3, 0.7):
        f.add(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{R * k}" ry="{R}" fill="none" '
            f'stroke="{SOFT}" stroke-width="1" opacity=".25"/>'
        )
    n, golden = 44, math.pi * (3 - math.sqrt(5))
    pts = []
    for i in range(n):
        zz = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(1 - zz * zz)
        t = golden * i
        px, py, pz = r * math.cos(t), zz, r * math.sin(t)
        ty, tz = py * math.cos(0.5) - pz * math.sin(0.5), py * math.sin(
            0.5
        ) + pz * math.cos(0.5)
        pts.append((cx + px * R, cy - ty * R, tz))
    front = [p for p in pts if p[2] > 0]
    for p, q in itertools.combinations(front, 2):
        if math.dist(p[:2], q[:2]) < R * 0.42:
            f.line(p[0], p[1], q[0], q[1], PINK, 1.5, 0.35)
    for px, py, pz in sorted(pts, key=lambda p: p[2]):
        if pz > 0:
            f.glow_dot(px, py, 6 + 6 * pz, AMBER)
        else:
            f.add(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{SOFT}" '
                f'opacity=".3"/>'
            )


def v_analysis(f, x, y, s):
    rng = np.random.default_rng(7)
    heights = np.clip(rng.gamma(2.0, 0.5, 14), 0.15, None)
    heights /= heights.max()
    top_h = s * 0.4
    bw = s / len(heights)
    for k, h in enumerate(heights):
        f.add(
            f'<rect x="{x + k * bw + 2:.1f}" y="{y + top_h * (1 - h):.1f}" '
            f'width="{bw - 4:.1f}" height="{top_h * h:.1f}" rx="4" '
            f'fill="{VIOLET}" fill-opacity=".55" stroke="{SOFT}" stroke-width="1.5"/>'
        )
    f.text(x, y + top_h + 40, "f: a step function", 24, color=SOFT, weight=600)
    fine = np.repeat(heights, 20)
    conv = np.convolve(fine, fine)
    conv /= conv.max()
    base = y + s - 10
    ch = s * 0.36
    pts = [(x + i / (len(conv) - 1) * s, base - v * ch) for i, v in enumerate(conv)]
    gid = uid("fill")
    f.add(
        f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0" stop-color="{PINK}" stop-opacity=".5"/>'
        f'<stop offset="1" stop-color="{PINK}" stop-opacity="0"/></linearGradient></defs>'
    )
    f.add(f'<path d="{path(pts)} L{x + s},{base} L{x},{base} Z" fill="url(#{gid})"/>')
    f.add(f'<path d="{path(pts)}" fill="none" stroke="{PINK}" stroke-width="4"/>')
    peak = max(pts, key=lambda p: -p[1])
    f.line(x, peak[1], x + s, peak[1], AMBER, 2, 0.8)
    f.text(x, base - ch - 30, "f ∗ f: keep its maximum low", 24, color=PINK, weight=600)


MSTD = [0, 2, 3, 4, 7, 11, 12, 14]


def v_numbers(f, x, y, s):
    sums = sorted({a + b for a in MSTD for b in MSTD})
    diffs = sorted({a - b for a in MSTD for b in MSTD})
    rows = [
        ("A", MSTD, 0, 14, SOFT),
        ("A + A", sums, 0, 28, AMBER),
        ("A − A", diffs, -14, 14, PINK),
    ]
    for k, (label, values, lo, hi, color) in enumerate(rows):
        yy = y + 70 + k * s * 0.33
        f.text(
            x, yy - 30, f"{label}   {len(values)} numbers", 26, color=color, weight=700
        )
        f.line(x, yy + 20, x + s, yy + 20, MUTED, 2, 0.4)
        span = hi - lo
        for v in range(lo, hi + 1):
            px = x + (v - lo) / span * s
            if v in values:
                f.glow_dot(px, yy + 20, 8 if span > 20 else 11, color)
            else:
                f.add(
                    f'<circle cx="{px:.1f}" cy="{yy + 20}" r="3" fill="{MUTED}" '
                    f'opacity=".35"/>'
                )
    f.text(x, y + s - 10, "more sums than differences", 26, color=MINT, weight=600)


def v_matrix(f, x, y, s):
    cell = s * 0.125
    gy = y + s * 0.1

    def grid(gx, letter, color):
        for i in range(2):
            for j in range(2):
                f.add(
                    f'<rect x="{gx + j * cell:.1f}" y="{gy + i * cell:.1f}" '
                    f'width="{cell - 6:.1f}" height="{cell - 6:.1f}" rx="8" '
                    f'fill="{color}" fill-opacity=".35" stroke="{color}" '
                    f'stroke-width="2"/>'
                )
                f.text(
                    gx + j * cell + cell / 2 - 3,
                    gy + i * cell + cell * 0.62,
                    f"{letter}{i + 1}{j + 1}",
                    24,
                    anchor="middle",
                    family=MONO,
                )
        return gx + 2 * cell

    gx = grid(x, "a", VIOLET)
    f.text(gx + 30, gy + cell * 1.2, "×", 48, color=MUTED, anchor="middle")
    gx = grid(gx + 60, "b", PINK)
    f.text(gx + 30, gy + cell * 1.2, "=", 48, color=MUTED, anchor="middle")
    grid(gx + 60, "c", AMBER)
    top = gy + 2 * cell + 90
    bw = (s - 6 * 12) / 7
    for k in range(7):
        bx = x + k * (bw + 12)
        f.add(
            f'<rect x="{bx:.1f}" y="{top}" width="{bw:.1f}" height="{bw:.1f}" '
            f'rx="12" fill="{MINT}" fill-opacity=".2" stroke="{MINT}" '
            f'stroke-width="2"/>'
        )
        f.text(
            bx + bw / 2,
            top + bw * 0.64,
            f"m{k + 1}",
            26,
            color=MINT,
            anchor="middle",
            family=MONO,
            weight=600,
        )
    f.text(x, top + bw + 80, "7 products instead of 8", 34, weight=700)
    f.text(x, top + bw + 126, "a low-rank decomposition of the tensor", 24, color=MUTED)


def v_graphs(f, x, y, s):
    # top: a full Steiner tree for four terminals (120 degree joints)
    w, h = s * 0.9, s * 0.3
    ox, oy = x + s * 0.05, y + 20
    d = h / (2 * math.sqrt(3))
    t = [(ox, oy), (ox, oy + h), (ox + w, oy), (ox + w, oy + h)]
    s1, s2 = (ox + d, oy + h / 2), (ox + w - d, oy + h / 2)
    for p, q in [(t[0], s1), (t[1], s1), (s1, s2), (s2, t[2]), (s2, t[3])]:
        f.line(*p, *q, MINT, 4, 0.9)
    for p in t:
        f.glow_dot(*p, 11, AMBER)
    for p in (s1, s2):
        f.add(
            f'<rect x="{p[0] - 9}" y="{p[1] - 9}" width="18" height="18" '
            f'fill="{MINT}" transform="rotate(45 {p[0]} {p[1]})"/>'
        )
    f.text(
        x, oy + h + 55, "Steiner tree: connect the points, shortest", 22, color=MUTED
    )
    # bottom: the Petersen graph, properly 3-coloured
    cx, cy, R = x + s / 2, y + s * 0.73, s * 0.24
    outer = [
        (cx + R * math.sin(2 * math.pi * k / 5), cy - R * math.cos(2 * math.pi * k / 5))
        for k in range(5)
    ]
    inner = [
        (
            cx + R * 0.48 * math.sin(2 * math.pi * k / 5),
            cy - R * 0.48 * math.cos(2 * math.pi * k / 5),
        )
        for k in range(5)
    ]
    nodes = outer + inner
    edges = (
        [(k, (k + 1) % 5) for k in range(5)]
        + [(k, k + 5) for k in range(5)]
        + [(5 + k, 5 + (k + 2) % 5) for k in range(5)]
    )
    for colours in itertools.product(range(3), repeat=10):
        if all(colours[a] != colours[b] for a, b in edges):
            break
    for a, b in edges:
        f.line(*nodes[a], *nodes[b], SOFT, 2.5, 0.5)
    for k, p in enumerate(nodes):
        f.glow_dot(*p, 15, [VIOLET, PINK, AMBER][colours[k]])
    f.text(x, y + s, "colouring: neighbours differ, fewest colours", 22, color=MUTED)


def v_routing(f, x, y, s):
    cx, cy = x + s / 2, y + s / 2
    rng = random.Random(4)
    for k, color in enumerate([VIOLET, PINK, AMBER, MINT]):
        a0 = k * math.pi / 2 + 0.25
        stops = []
        for j in range(5):
            a = a0 + (j + 0.5) / 5 * (math.pi / 2 - 0.5)
            r = s * rng.uniform(0.22, 0.45)
            stops.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        route = [(cx, cy)] + stops + [(cx, cy)]
        f.add(
            f'<path d="{path(route)}" fill="none" stroke="{color}" '
            f'stroke-width="4" stroke-linejoin="round" opacity=".85"/>'
        )
        for p in stops:
            f.glow_dot(*p, rng.uniform(7, 13), color)
    f.add(
        f'<rect x="{cx - 22}" y="{cy - 22}" width="44" height="44" rx="8" '
        f'fill="#ffffff"/>'
    )
    f.text(cx, cy + 80, "depot", 22, color=MUTED, anchor="middle")


def v_automl(f, x, y, s):
    # data table
    tw, th = s * 0.34, s * 0.42
    for i in range(7):
        for j in range(4):
            f.add(
                f'<rect x="{x + j * tw / 4 + 2:.1f}" y="{y + i * th / 7 + 2:.1f}" '
                f'width="{tw / 4 - 4:.1f}" height="{th / 7 - 4:.1f}" rx="3" '
                f'fill="{SOFT if i == 0 else MUTED}" '
                f'fill-opacity="{.6 if i == 0 else .18}"/>'
            )
    f.text(x, y + th + 40, "tabular data", 22, color=MUTED)
    steps = [
        ("impute", VIOLET),
        ("scale", VIOLET),
        ("select features", PINK),
        ("model", AMBER),
    ]
    bx, bw, bh = x + s * 0.48, s * 0.52, 62
    for k, (label, color) in enumerate(steps):
        by = y + k * (bh + 22)
        card(f, bx, by, bw, bh, stroke=color, fill=color + "22", r=14)
        f.text(bx + bw / 2, by + bh * 0.64, label, 24, anchor="middle", family=MONO)
        if k:
            f.add(
                f'<line x1="{bx + bw / 2}" y1="{by - 22}" x2="{bx + bw / 2}" '
                f'y2="{by - 4}" stroke="{MUTED}" stroke-width="2.5" '
                f'marker-end="url(#arrow-g)"/>'
            )
    f.add(
        f'<line x1="{x + tw + 10}" y1="{y + th / 2}" x2="{bx - 10}" '
        f'y2="{y + th / 2}" stroke="{MUTED}" stroke-width="2.5" '
        f'marker-end="url(#arrow-g)"/>'
    )
    # ROC curve
    rx, ry, rs = x + s * 0.2, y + s * 0.56, s * 0.42
    f.line(rx, ry + rs, rx + rs, ry + rs, MUTED, 2, 0.6)
    f.line(rx, ry, rx, ry + rs, MUTED, 2, 0.6)
    f.line(rx, ry + rs, rx + rs, ry, MUTED, 1.5, 0.3)
    pts = [
        (rx + t * rs, ry + rs - (1 - (1 - t) ** 4) * rs) for t in np.linspace(0, 1, 40)
    ]
    f.add(f'<path d="{path(pts)}" fill="none" stroke="{MINT}" stroke-width="4"/>')
    f.text(rx + rs + 24, ry + rs * 0.5, "ROC AUC", 26, color=MINT, weight=700)


def v_kernels(f, x, y, s):
    n = 9
    cell = s * 0.86 / n
    ox, oy = x + s * 0.12, y
    rng = np.random.default_rng(1)
    best, best_v = None, -1
    for i in range(n):
        for j in range(n):
            v = math.exp(-((i - 5.6) ** 2 + (j - 3.2) ** 2) / 14) + rng.normal(0, 0.08)
            v = max(0, min(1, v))
            if v > best_v:
                best, best_v = (i, j), v
            f.add(
                f'<rect x="{ox + j * cell + 2:.1f}" y="{oy + i * cell + 2:.1f}" '
                f'width="{cell - 4:.1f}" height="{cell - 4:.1f}" rx="6" '
                f'fill="{colour_at(v)}" fill-opacity="{0.2 + 0.8 * v:.2f}"/>'
            )
    i, j = best
    f.add(
        f'<rect x="{ox + j * cell - 2:.1f}" y="{oy + i * cell - 2:.1f}" '
        f'width="{cell + 4:.1f}" height="{cell + 4:.1f}" rx="8" fill="none" '
        f'stroke="#ffffff" stroke-width="4"/>'
    )
    f.text(
        ox + s * 0.43,
        y + s * 0.86 + 50,
        "block_size_x",
        22,
        color=MUTED,
        anchor="middle",
        family=MONO,
    )
    yy = y + s * 0.43
    f.add(
        f'<text x="{x + 20}" y="{yy}" font-family="{MONO}" font-size="22" '
        f'fill="{MUTED}" text-anchor="middle" '
        f'transform="rotate(-90 {x + 20} {yy})">block_size_y</text>'
    )
    f.text(
        ox + s * 0.43,
        y + s * 0.86 + 90,
        "fastest configuration",
        22,
        anchor="middle",
        weight=600,
    )


def v_photonics(f, x, y, s):
    widths = [0.042, 0.06] * 7
    total = sum(widths)
    bx = x + s * 0.08
    span = s * 0.84
    hx = bx
    for k, w in enumerate(widths):
        ww = w / total * span
        color = VIOLET if k % 2 else SOFT
        f.add(
            f'<rect x="{hx:.1f}" y="{y + 30}" width="{ww:.1f}" '
            f'height="{s * 0.36:.1f}" fill="{color}" '
            f'fill-opacity="{.75 if k % 2 else .35}"/>'
        )
        hx += ww
    f.add(
        f'<line x1="{x}" y1="{y + 90}" x2="{bx + 30}" y2="{y + s * 0.2}" '
        f'stroke="{AMBER}" stroke-width="5" marker-end="url(#arrow-a)"/>'
    )
    f.add(
        f'<line x1="{bx + 30}" y1="{y + s * 0.2}" x2="{x}" y2="{y + s * 0.34}" '
        f'stroke="{AMBER}" stroke-width="5" stroke-dasharray="10 8" '
        f'marker-end="url(#arrow-a)"/>'
    )
    f.text(x, y + s * 0.36 + 80, "layer thicknesses → reflectance", 22, color=MUTED)
    base, ch = y + s - 20, s * 0.38
    lam = np.linspace(0, 1, 200)
    refl = 0.96 * np.exp(-(((lam - 0.5) / 0.16) ** 8)) + 0.12 * np.sin(
        lam * 60
    ) ** 2 * (np.abs(lam - 0.5) > 0.16)
    pts = [(x + t * s, base - r * ch) for t, r in zip(lam, refl)]
    f.line(x, base, x + s, base, MUTED, 2, 0.5)
    f.add(f'<path d="{path(pts)}" fill="none" stroke="{PINK}" stroke-width="4"/>')
    f.text(
        x + s * 0.5,
        base - ch - 20,
        "stop band",
        22,
        color=PINK,
        anchor="middle",
        weight=600,
    )


def v_pareto(f, x, y, s):
    rng = np.random.default_rng(3)
    pts = []
    while len(pts) < 70:
        a, b = rng.uniform(0.05, 1, 2)
        if a * b > 0.12 or rng.random() < 0.08:
            pts.append((a, b))
    front = sorted(
        p for p in pts if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)
    )

    def P(p):
        return x + p[0] * s, y + s - p[1] * s

    f.line(x, y + s, x + s, y + s, MUTED, 2, 0.6)
    f.line(x, y, x, y + s, MUTED, 2, 0.6)
    steps = [front[0]]
    for a, b in zip(front, front[1:]):
        steps += [(b[0], a[1]), b]
    f.add(
        f'<path d="{path([P(p) for p in steps])}" fill="none" stroke="{PINK}" '
        f'stroke-width="3.5"/>'
    )
    for p in pts:
        if p in front:
            f.glow_dot(*P(p), 11, PINK)
        else:
            f.add(
                f'<circle cx="{P(p)[0]:.1f}" cy="{P(p)[1]:.1f}" r="7" '
                f'fill="{SOFT}" opacity=".45"/>'
            )
    f.text(x + s, y + s + 40, "objective 1 →", 22, color=MUTED, anchor="end")
    f.add(
        f'<text x="{x - 18}" y="{y}" font-family="{FONT}" font-size="22" '
        f'fill="{MUTED}" text-anchor="end" transform="rotate(-90 {x - 18} {y})">'
        f"objective 2 →</text>"
    )


BENCHMARKS = [
    (
        "bbob",
        ["bbob"],
        ["BBOB, one function", "at a time"],
        ["The 24 noiseless BBOB functions,", "each its own problem."],
        ["10-D · train instances 1-10 · test 11-30"],
        v_bbob,
    ),
    (
        "bbob-groups",
        ["bbob_groups"],
        ["BBOB function", "groups"],
        ["One problem per function group:", "a generalist for each class."],
        ["5 problems · 10-D", "train instances 1-5 · test 6-15"],
        v_bbob_groups,
    ),
    (
        "ma-bbob",
        ["ma_bbob"],
        ["MA-BBOB"],
        ["Many-affine combinations of BBOB", "functions: new landscapes."],
        ["10-D · 25 training · 50 test instances"],
        v_ma_bbob,
    ),
    (
        "packing",
        [
            "circle_packing",
            "hexagon_packing",
            "rectangle_packing",
            "unit_square_packing",
        ],
        ["Packing"],
        ["Fit shapes as tightly", "as possible."],
        ["circles, hexagons, rectangles, squares"],
        v_packing,
    ),
    (
        "heilbronn",
        ["heilbronn_triangle", "heilbronn_convex_region", "min_max_distance_ratio"],
        ["Point", "configurations"],
        ["Place points so that the smallest", "triangle is as large as possible."],
        ["Heilbronn problems · min/max distance ratio"],
        v_heilbronn,
    ),
    (
        "spheres",
        ["spherical_code", "kissing_number_11d"],
        ["Spheres and codes"],
        ["Spread points over a sphere;", "touch as many spheres as possible."],
        ["spherical codes · kissing number in 11-D"],
        v_spheres,
    ),
    (
        "analysis",
        ["auto_correlation_ineq1-3", "uncertainty_inequality", "erdos_min_overlap"],
        ["Inequalities", "in analysis"],
        ["Construct functions that", "tighten known bounds."],
        ["auto-correlation · uncertainty · Erdős overlap"],
        v_analysis,
    ),
    (
        "number-theory",
        ["sum_vs_difference"],
        ["Sums versus", "differences"],
        ["Find sets with many more sums", "than differences."],
        ["additive combinatorics"],
        v_numbers,
    ),
    (
        "matrix",
        ["matrix_multiplication"],
        ["Matrix", "multiplication"],
        ["Multiply matrices with fewer", "multiplications."],
        ["low-rank tensor decompositions"],
        v_matrix,
    ),
    (
        "graphs",
        ["euclidian_steiner_tree", "graph_coloring"],
        ["Trees and", "colourings"],
        ["Classic graph problems with", "known instances."],
        ["Steiner trees (50, 100, 250 points)", "graph colouring"],
        v_graphs,
    ),
    (
        "routing",
        ["tsp", "vrp"],
        ["Routing"],
        ["Short tours, from one city to", "a fleet of vehicles."],
        ["travelling salesman · capacitated VRP"],
        v_routing,
    ),
    (
        "automl",
        ["automl"],
        ["AutoML"],
        ["Design machine-learning pipelines", "for a tabular dataset."],
        ["scored by test ROC AUC"],
        v_automl,
    ),
    (
        "kerneltuner",
        ["kerneltuner"],
        ["GPU kernel", "tuning"],
        ["Optimisers that find the fastest", "configuration of a GPU kernel."],
        ["GEMM · convolution · dedispersion · hotspot", "with Kernel Tuner"],
        v_kernels,
    ),
    (
        "photonics",
        ["photonics"],
        ["Photonics"],
        ["Design optical structures", "layer by layer."],
        ["Bragg mirror · ellipsometry · photovoltaics"],
        v_photonics,
    ),
    (
        "multi-objective",
        ["toy_multiobjective"],
        ["Multi-objective"],
        ["Trade-offs instead of one score:", "a Pareto front."],
        ["Pareto dominance in Fitness"],
        v_pareto,
    ),
]


def bench_frame(keys, title_lines, tagline, details, draw):
    f = Frame()
    arrow_defs(f)
    tx, ty, ts = 150, 190, 700
    f.add(
        f'<circle cx="{tx + ts / 2}" cy="{ty + ts / 2}" r="{ts * 0.48}" '
        f'fill="{VIOLET}" opacity=".18" filter="url(#blur8)"/>'
    )
    card(f, tx, ty, ts, ts, stroke=AMBER, r=36, width=2.5)
    draw(f, tx + 50, ty + 50, ts - 100)
    # right column, vertically centred
    x = 960
    chip_rows, row, used = [], [], 0
    for key in keys:
        w = 0.6 * 24 * len(key) + 2 * 24 * 0.8
        if row and used + w > 800:
            chip_rows.append(row)
            row, used = [], 0
        row.append(key)
        used += w + 14
    chip_rows.append(row)
    height = (
        len(chip_rows) * 60
        + 50
        + len(title_lines) * 100
        + 30
        + len(tagline) * 52
        + 30
        + len(details) * 44
    )
    y = 540 - height / 2
    for row in chip_rows:
        cx = x
        for key in row:
            w, _ = pill(f, cx, y, key, size=24)
            cx += w + 14
        y += 60
    y += 50
    for line in title_lines:
        y += 100
        f.text(x, y, line, 92, weight=800, spacing=-1)
    y += 30
    for line in tagline:
        y += 52
        f.text(x, y, line, 40, color="url(#grad)", weight=600)
    y += 30
    for line in details:
        y += 44
        f.text(x, y, line, 30, color=MUTED)
    brand(f)
    return f


# ---------------------------------------------------------- analysis ---
def evolution(n=150, seed=8):
    rng = random.Random(seed)
    nodes = [dict(fit=0.15, tok=300, parent=None, kids=0)]
    for i in range(1, n):
        ranked = sorted(range(len(nodes)), key=lambda k: -(nodes[k]["fit"] or -1))
        p = rng.choice(ranked[:6]) if rng.random() < 0.8 else rng.randrange(len(nodes))
        parent = nodes[p]
        if rng.random() < 0.12:
            fit = None  # failed evaluation
        else:
            base = parent["fit"] if parent["fit"] is not None else 0.2
            fit = min(1, max(0, base + rng.gauss(0.015, 0.06)))
        tok = max(120, parent["tok"] + rng.gauss(25, 70))
        parent["kids"] += 1
        nodes.append(dict(fit=fit, tok=tok, parent=p, kids=0))
    return nodes


def ceg_frame():
    f = Frame()
    title(
        f,
        "Code evolution graph",
        "Every dot is a generated algorithm; lines connect it to its parent.",
    )
    x0, y0, x1, y1 = 200, 290, 1640, 930
    nodes = evolution()
    toks = [n["tok"] for n in nodes]
    lo, hi = min(toks), max(toks)

    def P(i):
        return (
            x0 + i / (len(nodes) - 1) * (x1 - x0),
            y1 - (nodes[i]["tok"] - lo) / (hi - lo) * (y1 - y0),
        )

    best = max(
        (i for i, n in enumerate(nodes) if n["fit"] is not None),
        key=lambda i: nodes[i]["fit"],
    )
    lineage, i = set(), best
    while i is not None:
        lineage.add(i)
        i = nodes[i]["parent"]
    f.line(x0 - 20, y1 + 30, x1 + 20, y1 + 30, "#ffffff", 2, 0.25)
    f.line(x0 - 20, y0 - 20, x0 - 20, y1 + 30, "#ffffff", 2, 0.25)
    f.text((x0 + x1) / 2, y1 + 80, "evaluation", 26, color=MUTED, anchor="middle")
    ym = (y0 + y1) / 2
    f.add(
        f'<text x="{x0 - 50}" y="{ym}" font-family="{FONT}" font-size="26" '
        f'fill="{MUTED}" text-anchor="middle" transform="rotate(-90 {x0 - 50} '
        f'{ym})">code size (tokens)</text>'
    )
    for i, n in enumerate(nodes):
        if n["parent"] is None:
            continue
        on = i in lineage and n["parent"] in lineage
        a, b = P(n["parent"]), P(i)
        f.add(
            f'<path d="M{a[0]:.1f},{a[1]:.1f} C{(a[0] + b[0]) / 2:.1f},'
            f'{a[1]:.1f} {(a[0] + b[0]) / 2:.1f},{b[1]:.1f} {b[0]:.1f},{b[1]:.1f}" '
            f'fill="none" stroke="{PINK if on else SOFT}" '
            f'stroke-width="{4 if on else 1.3}" opacity="{.95 if on else .3}"/>'
        )
    for i, n in enumerate(nodes):
        x, y = P(i)
        r = 6 + 3 * math.sqrt(n["kids"])
        if n["fit"] is None:
            f.add(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" '
                f'stroke="{MUTED}" stroke-width="2" opacity=".55"/>'
            )
        else:
            f.glow_dot(x, y, r, colour_at(n["fit"]), 1)
    bx, by = P(best)
    f.add(
        f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="22" fill="none" '
        f'stroke="#ffffff" stroke-width="3"/>'
    )
    f.text(bx + 30, by - 20, "best algorithm", 24, weight=700)
    # legend
    lx, ly = 1700, 330
    gid = uid("leg")
    f.add(
        f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="1" y2="0">'
        f'<stop offset="0" stop-color="{VIOLET}"/><stop offset=".55" '
        f'stop-color="{PINK}"/><stop offset="1" stop-color="{AMBER}"/>'
        f"</linearGradient></defs>"
    )
    f.add(
        f'<rect x="{lx}" y="{ly}" width="22" height="300" rx="11" '
        f'fill="url(#{gid})"/>'
    )
    f.text(lx + 36, ly + 20, "better", 22, color=MUTED)
    f.text(lx + 36, ly + 300, "worse", 22, color=MUTED)
    f.text(lx - 4, ly - 30, "score", 24, weight=600)
    f.add(
        f'<circle cx="{lx + 11}" cy="{ly + 370}" r="9" fill="none" '
        f'stroke="{MUTED}" stroke-width="2"/>'
    )
    f.text(lx + 36, ly + 378, "failed", 22, color=MUTED)
    f.text(lx - 4, ly + 440, "size =", 22, color=MUTED)
    f.text(lx - 4, ly + 470, "children", 22, color=MUTED)
    return f


def convergence_frame():
    f = Frame()
    title(
        f,
        "Convergence",
        "Best score so far per evaluation: mean and spread over seeds.",
    )
    x0, y0, x1, y1 = 220, 290, 1500, 930
    for k in range(1, 6):
        yy = y0 + k * (y1 - y0) / 6
        f.line(x0, yy, x1, yy, "#ffffff", 1, 0.07)
    f.line(x0, y1, x1, y1, "#ffffff", 2, 0.25)
    f.line(x0, y0, x0, y1, "#ffffff", 2, 0.25)
    f.text((x0 + x1) / 2, y1 + 60, "evaluations", 26, color=MUTED, anchor="middle")
    ym = (y0 + y1) / 2
    f.add(
        f'<text x="{x0 - 40}" y="{ym}" font-family="{FONT}" font-size="26" '
        f'fill="{MUTED}" text-anchor="middle" transform="rotate(-90 {x0 - 40} '
        f'{ym})">best score so far</text>'
    )
    methods = [
        ("Method A", VIOLET, 5.0, 0.88),
        ("Method B", PINK, 3.0, 0.8),
        ("Method C", AMBER, 2.0, 0.66),
        ("Method D", MINT, 1.2, 0.48),
    ]
    t = np.linspace(0, 1, 120)
    for k, (name, color, rate, top) in enumerate(methods):
        rng = np.random.default_rng(k)
        runs = []
        for _ in range(5):
            v = top * (1 - np.exp(-rate * t * rng.uniform(0.7, 1.3)))
            v = v + rng.normal(0, 0.03, len(t)) * (1 - t)
            runs.append(np.maximum.accumulate(np.clip(v, 0, 1)))
        runs = np.array(runs)
        mean, sd = runs.mean(0), runs.std(0)

        def P(i, v):
            return x0 + t[i] * (x1 - x0), y1 - v * (y1 - y0)

        upper = [P(i, mean[i] + sd[i]) for i in range(len(t))]
        lower = [P(i, mean[i] - sd[i]) for i in range(len(t))][::-1]
        f.add(
            f'<path d="{path(upper + lower, True)}" fill="{color}" ' f'opacity=".16"/>'
        )
        pts = [P(i, mean[i]) for i in range(len(t))]
        f.add(
            f'<path d="{path(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="10" opacity=".25" filter="url(#blur8)"/>'
        )
        f.add(
            f'<path d="{path(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="4.5" stroke-linejoin="round"/>'
        )
        f.glow_dot(*pts[-1], 8, color)
        ly = 360 + k * 70
        f.add(
            f'<rect x="1580" y="{ly - 22}" width="44" height="8" rx="4" '
            f'fill="{color}"/>'
        )
        f.text(1640, ly - 8, name, 28, weight=600)
    f.text(1580, 680, "illustrative data", 22, color=MUTED)
    return f


# ---------------------------------------------------------- framework ---
STAGES = [
    (
        "Define",
        SOFT,
        ["YAML experiment", "iohblade CLI", "Python API", "Coding-agent skill"],
    ),
    ("Components", VIOLET, ["Problems", "Methods", "LLMs", "Default suites"]),
    (
        "Run",
        AMBER,
        ["Seeds and repeats", "Parallel runs", "Isolated evaluation", "Exact budgets"],
    ),
    (
        "Record",
        MINT,
        [
            "Code and scores",
            "Prompts, feedback",
            "Configs and hashes",
            "MLflow, Trackio",
        ],
    ),
    ("Analyse", PINK, ["Convergence", "Compare methods", "Code evolution", "Web app"]),
]


def framework_frame():
    f = Frame()
    arrow_defs(f)
    lw = f.leaf(150, 95, 70)
    f.text(150 + lw + 24, 150, "How BLADE works", 60, weight=800)
    cw, gap, top, ch = 312, 48, 270, 640
    x = (W - 5 * cw - 4 * gap) / 2
    for k, (name, color, items) in enumerate(STAGES):
        cx = x + k * (cw + gap)
        f.add(
            f'<rect x="{cx}" y="{top}" width="{cw}" height="{ch}" rx="28" '
            f'fill="{color}" fill-opacity=".07" stroke="{color}" '
            f'stroke-width="2.5"/>'
        )
        f.add(f'<circle cx="{cx + 52}" cy="{top + 62}" r="24" fill="{color}"/>')
        f.text(
            cx + 52,
            top + 72,
            str(k + 1),
            28,
            color="#0b0d12",
            weight=800,
            anchor="middle",
        )
        f.text(cx + 88, top + 73, name, 32, weight=800)
        for j, item in enumerate(items):
            iy = top + 170 + j * 110
            f.add(
                f'<rect x="{cx + 24}" y="{iy - 46}" width="{cw - 48}" '
                f'height="76" rx="16" fill="{SURFACE}" stroke="{EDGE}" '
                f'stroke-width="1.5"/>'
            )
            f.add(f'<circle cx="{cx + 50}" cy="{iy - 8}" r="7" fill="{color}"/>')
            f.text(cx + 70, iy, item, 22, weight=500)
        if k < 4:
            ax = cx + cw + 8
            f.add(
                f'<line x1="{ax}" y1="{top + ch / 2}" x2="{ax + gap - 18}" '
                f'y2="{top + ch / 2}" stroke="{MUTED}" stroke-width="4" '
                f'marker-end="url(#arrow-g)"/>'
            )
    f.text(
        W / 2,
        top + ch + 70,
        "One fair harness: the same problems, LLM and " "budget for every method.",
        30,
        color=MUTED,
        anchor="middle",
    )
    return f


def framework_detailed_frame():
    f = Frame()
    lw = f.leaf(150, 95, 70)
    f.text(150 + lw + 24, 150, "The BLADE platform", 60, weight=800)
    png = (ROOT / "docs" / "media" / "blade-platform-dark.png").read_bytes()
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    w = 1700
    h = w * 2000 / 4800
    image(f, (W - w) / 2, 250, w, h, uri, r=24)
    return f


# ----------------------------------------------------------- AAD loop ---
def node(f, cx, cy, w, h, name, sub, color):
    f.add(
        f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" '
        f'rx="26" fill="{SURFACE}" stroke="{color}" stroke-width="3"/>'
    )
    f.add(
        f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" '
        f'rx="26" fill="{color}" fill-opacity=".08"/>'
    )
    f.text(cx, cy - 6, name, 34, weight=800, anchor="middle")
    f.text(cx, cy + 34, sub, 22, color=MUTED, anchor="middle")


def aad_loop_frame():
    f = Frame()
    arrow_defs(f)
    title(
        f,
        "LLM-driven algorithm design",
        "The LLM writes algorithms; scores decide which ones it improves next.",
        y=140,
        size=56,
    )
    # start: task prompt -> LLM writes code -> population (top row)
    node(f, 330, 300, 360, 130, "Task prompt", "problem, interface, example", SOFT)
    node(f, 790, 300, 340, 130, "LLM writes code", "initial algorithms", VIOLET)
    for x1, x2, color in [(512, 616, SOFT), (962, 1046, VIOLET)]:
        f.add(
            f'<line x1="{x1}" y1="300" x2="{x2}" y2="300" stroke="{color}" '
            f'stroke-width="5" marker-end="url(#arrow-{MARK[color]})"/>'
        )
    # the loop
    cx, cy = 1230, 600
    arcs = [
        ("M1412,300 Q1590,300 1590,530", AMBER),
        ("M1590,668 Q1590,900 1414,900", MINT),
        ("M1046,900 Q870,900 870,670", PINK),
        ("M870,532 C870,440 1110,470 1110,370", VIOLET),
    ]
    for d, color in arcs:
        f.add(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="5" '
            f'marker-end="url(#arrow-{MARK[color]})"/>'
        )
    node(f, 1230, 300, 360, 130, "Population", "candidate algorithms", AMBER)
    node(f, 1590, 600, 330, 130, "Scoring", "run on the problem", MINT)
    node(f, 1230, 900, 360, 130, "Selection", "keep the best", PINK)
    node(f, 870, 600, 330, 130, "LLM mutation", "improve or combine", VIOLET)
    # centre: three code cards, the best highlighted
    for k, (dx, color) in enumerate([(-80, SOFT), (0, AMBER), (80, SOFT)]):
        x, y = cx + dx - 50, cy - 70
        f.add(
            f'<rect x="{x}" y="{y + (0 if k == 1 else 20)}" width="100" '
            f'height="120" rx="12" fill="{SURFACE}" stroke="{color}" '
            f'stroke-width="{3 if k == 1 else 1.5}"/>'
        )
        for j in range(5):
            lw = [70, 50, 60, 40, 55][(j + k) % 5]
            f.add(
                f'<rect x="{x + 14}" y="{y + (0 if k == 1 else 20) + 20 + j * 18}" '
                f'width="{lw}" height="7" rx="3" fill="{color}" '
                f'opacity="{.8 if k == 1 else .4}"/>'
            )
    f.text(
        cx,
        cy + 120,
        "repeat until the budget is used",
        24,
        color=MUTED,
        anchor="middle",
    )
    return f


# -------------------------------------------------------------- extras ---
EPISODES = [
    "BLADE in 5 minutes",
    "Experiments in YAML",
    "Local and hosted LLMs",
    "Your own problem",
    "Your own method",
    "Analysing results",
    "BLADE with a coding agent",
]


def episode_frame(k, name):
    f = Frame()
    M.plot_frame(f, 1.0, opacity=0.08)
    lw = f.leaf(150, 370, 110)
    f.text(150, 560, f"EPISODE {k}", 36, color="url(#grad)", weight=700, spacing=8)
    f.text(150, 680, name, 110, weight=800, spacing=-1)
    f.text(150, 760, "BLADE video tutorials", 34, color=MUTED)
    return f


METHODS = [
    ("LLaMEA", VIOLET),
    ("EoH", VIOLET),
    ("ReEvo", VIOLET),
    ("FunSearch", PINK),
    ("ShinkaEvolve", PINK),
    ("LHNS", AMBER),
    ("MCTS-AHD", AMBER),
    ("MoEH", AMBER),
    ("Random search", MINT),
]


def methods_frame():
    f = Frame()
    title(
        f,
        "Nine methods, one harness",
        "Built-in LLM-driven algorithm design methods, plus your own.",
        y=200,
        size=72,
    )
    cw, chh, gap = 470, 150, 40
    x0 = (W - 3 * cw - 2 * gap) / 2
    for k, (name, color) in enumerate(METHODS):
        i, j = divmod(k, 3)
        x, y = x0 + j * (cw + gap), 360 + i * (chh + gap)
        f.add(
            f'<rect x="{x}" y="{y}" width="{cw}" height="{chh}" rx="28" '
            f'fill="{color}" fill-opacity=".08" stroke="{color}" '
            f'stroke-width="2.5"/>'
        )
        f.text(x + cw / 2, y + chh / 2 + 16, name, 46, weight=800, anchor="middle")
    f.text(
        W / 2,
        980,
        "+ your own Method class",
        34,
        color=MUTED,
        anchor="middle",
        family=MONO,
    )
    return f


def lower_third_svg(
    name="Niki van Stein", role="Associate professor · Leiden University"
):
    x, y, w, h = 120, 830, 900, 150
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}"><defs>'
        f'<linearGradient id="grad" x1="0" x2="1"><stop offset="0" '
        f'stop-color="{VIOLET}"/><stop offset=".55" stop-color="{PINK}"/>'
        f'<stop offset="1" stop-color="{AMBER}"/></linearGradient></defs>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" '
        f'fill="#0b0d12" fill-opacity=".82"/>'
        f'<rect x="{x}" y="{y}" width="10" height="{h}" rx="5" fill="url(#grad)"/>'
        f'<text x="{x + 50}" y="{y + 70}" font-family="{FONT}" font-size="54" '
        f'font-weight="800" fill="#ffffff">{esc(name)}</text>'
        f'<text x="{x + 50}" y="{y + 118}" font-family="{FONT}" font-size="30" '
        f'font-weight="600" fill="url(#grad)">{esc(role)}</text></svg>'
    )


# ------------------------------------------------- isolated evaluation ---
GENERATED = [
    ("class AdaptiveDE:", PINK),
    ("    def __init__(self, budget, dim):", "#e6e6f0"),
    ("        self.budget = budget", "#e6e6f0"),
    ("    def __call__(self, func):", "#e6e6f0"),
    ("        pop = np.random.uniform(", "#e6e6f0"),
    ("            -5, 5, (20, self.dim))", "#e6e6f0"),
    ("        ...", MUTED),
]


def isolation_frame():
    f = Frame()
    arrow_defs(f)
    title(
        f,
        "Every algorithm runs in isolation",
        "Generated code is untrusted: it can crash or hang.",
    )
    M.editor(
        f, 120, 330, 560, 420, title="generated by the LLM", lines=GENERATED, size=22
    )
    # the sandbox
    bx, by, bw, bh = 800, 300, 520, 480
    f.add(
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="32" '
        f'fill="{VIOLET}" fill-opacity=".07" stroke="{VIOLET}" '
        f'stroke-width="3" stroke-dasharray="14 10"/>'
    )
    f.text(bx + bw / 2, by + 70, "Sandbox", 40, weight=800, anchor="middle")
    for k, (label, color) in enumerate(
        [
            ("separate process", SOFT),
            ("own virtual environment", MINT),
            ("time limit", AMBER),
            ("output and errors captured", PINK),
        ]
    ):
        yy = by + 150 + k * 78
        f.add(
            f'<rect x="{bx + 40}" y="{yy - 40}" width="{bw - 80}" height="60" '
            f'rx="16" fill="{SURFACE}" stroke="{EDGE}" stroke-width="1.5"/>'
        )
        f.add(f'<circle cx="{bx + 72}" cy="{yy - 10}" r="9" fill="{color}"/>')
        f.text(bx + 98, yy - 1, label, 26, weight=500)
    f.add(
        f'<line x1="690" y1="540" x2="790" y2="540" stroke="{MUTED}" '
        f'stroke-width="5" marker-end="url(#arrow-g)"/>'
    )
    # outcomes
    outcomes = [
        ("score", "fitness 0.87", MINT),
        ("error", "traceback becomes feedback", PINK),
        ("timeout", "stopped, reported as feedback", AMBER),
    ]
    for k, (name, sub, color) in enumerate(outcomes):
        oy = 330 + k * 150
        f.add(
            f'<path d="M{bx + bw + 10},540 C{bx + bw + 60},540 '
            f'{bx + bw + 40},{oy + 50} {bx + bw + 92},{oy + 50}" fill="none" '
            f'stroke="{color}" stroke-width="4" '
            f'marker-end="url(#arrow-{MARK[color]})"/>'
        )
        card(f, 1420, oy, 400, 100, stroke=color, r=20)
        f.text(1446, oy + 44, name, 30, color=color, weight=800)
        f.text(1446, oy + 80, sub, 22, color=MUTED)
    f.text(
        W / 2,
        900,
        "Errors go back to the LLM; a broken benchmark (harness error) "
        "stops the run early.",
        28,
        color=MUTED,
        anchor="middle",
    )
    return f


# ---------------------------------------------------------------- LLMs ---
ICONS = ROOT / "docs" / "media" / "figure" / "icons"


def icon_uri(name: str) -> str:
    data = (ICONS / f"{name}.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


def llms_frame():
    f = Frame()
    title(
        f,
        "Any LLM, hosted or local",
        "Every method gets the same model through one interface.",
        y=170,
        size=66,
    )
    groups = [
        (
            "Hosted APIs",
            VIOLET,
            [
                ("openai", "OpenAI"),
                ("claude-color", "Claude"),
                ("gemini-color", "Gemini"),
                ("deepseek-color", "DeepSeek"),
            ],
        ),
        (
            "Local, on your own GPUs",
            AMBER,
            [
                ("qwen-color", "Qwen"),
                ("ollama", "Ollama"),
                ("vllm-color", "vLLM"),
                ("lmstudio", "LM Studio"),
            ],
        ),
    ]
    gw, gh, gap = 780, 430, 60
    x0 = (W - 2 * gw - gap) / 2
    for g, (name, color, logos) in enumerate(groups):
        gx = x0 + g * (gw + gap)
        f.add(
            f'<rect x="{gx}" y="330" width="{gw}" height="{gh}" rx="32" '
            f'fill="{color}" fill-opacity=".07" stroke="{color}" '
            f'stroke-width="2.5"/>'
        )
        f.text(gx + 44, 400, name, 36, weight=800)
        for k, (icon, label) in enumerate(logos):
            cx = gx + gw / 8 * (2 * k + 1)
            f.add(
                f'<circle cx="{cx}" cy="540" r="96" fill="{color}" '
                f'opacity=".16" filter="url(#blur8)"/>'
                f'<circle cx="{cx}" cy="540" r="74" fill="#ffffff"/>'
            )
            image(f, cx - 46, 494, 92, 92, icon_uri(icon))
            f.text(cx, 680, label, 28, weight=600, anchor="middle")
    pill(f, x0, 830, "provider: multi", color=PINK, size=26)
    f.text(
        x0 + 330,
        866,
        "spread the calls of one experiment over several servers or models",
        28,
        color=MUTED,
    )
    return f


# ----------------------------------------------------- YAML to terminal ---
YAML_EP2 = [
    ("experiment: {output_dir: results/compare, runs: 5}", MINT),
    ("llms:", SOFT),
    ("  qwen: {provider: ollama, model: qwen2.5-coder}", AMBER),
    ("  gemini: {provider: gemini, model: gemini-2.0-flash}", AMBER),
    ("methods:", SOFT),
    ("  - {type: llamea, llm: [qwen, gemini]}", PINK),
    ("  - {type: shinka_evolve, llm: qwen}", PINK),
    ("  - {type: random_search, llm: qwen}", PINK),
    ("problems:", SOFT),
    ("  - {suite: bbob_groups, params: {dim: 5}}", "#e6e6f0"),
    ("  - {type: circle_packing}", "#e6e6f0"),
]

TERMINAL_EP2 = [
    ("$ iohblade validate experiment.yml", "#e6e6f0"),
    ("Methods (4): LLaMEA-qwen, LLaMEA-gemini,", MUTED),
    ("             Shinka-qwen, RS-qwen", MUTED),
    ("Problems (6): BBOB-g1-5d ... BBOB-g5-5d,", MUTED),
    ("              circle_packing_n40", MUTED),
    ("Runs: 4 methods x 6 problems x 5 seeds = 120", "#e6e6f0"),
    ("Configuration is valid.", MINT),
    ("$ iohblade check experiment.yml", "#e6e6f0"),
    ("  ... ready to run.", MINT),
    ("$ iohblade run experiment.yml", AMBER),
]


def yaml_terminal_frame():
    f = Frame()
    arrow_defs(f)
    title(
        f,
        "From one YAML file to 120 runs",
        "Describe the experiment, validate it, check it, run it.",
        y=150,
    )
    M.editor(f, 90, 300, 860, 600, title="experiment.yml", lines=YAML_EP2, size=25)
    M.editor(f, 1010, 300, 820, 600, title="terminal", lines=TERMINAL_EP2, size=24)
    f.add(
        f'<line x1="956" y1="600" x2="1002" y2="600" stroke="{MUTED}" '
        f'stroke-width="5" marker-end="url(#arrow-g)"/>'
    )
    return f


# ------------------------------------------------------------ up next ---
NEXT = [
    "Compare several methods and LLMs in one file.",
    "Connect hosted APIs or models on your own GPUs.",
    "Turn your own optimisation problem into a benchmark.",
    "Plug in your own LLM-driven search method.",
    "Convergence, code evolution and the web app.",
    "Let your coding agent set up the benchmark.",
]


def next_frame(k):
    """Card shown at the end of episode k, announcing episode k + 1."""
    f = Frame()
    M.plot_frame(f, 1.0, opacity=0.06)
    f.text(150, 400, "UP NEXT", 36, color="url(#grad)", weight=700, spacing=8)
    f.text(150, 480, f"Episode {k + 1}", 44, color=MUTED, weight=600)
    f.text(150, 600, EPISODES[k], 104, weight=800, spacing=-1)
    f.text(150, 680, NEXT[k - 1], 38, color=MUTED)
    brand(f)
    return f


# --------------------------------------------------------------- main ---
def frames():
    yield "00-blank", Frame()
    for k, (slug, *spec) in enumerate(BENCHMARKS, 1):
        yield f"bench-{k:02d}-{slug}", bench_frame(*spec)
    yield "ceg", ceg_frame()
    yield "convergence", convergence_frame()
    yield "framework", framework_frame()
    yield "framework-detailed", framework_detailed_frame()
    yield "aad-loop", aad_loop_frame()
    yield "methods", methods_frame()
    yield "isolation", isolation_frame()
    yield "llms", llms_frame()
    yield "yaml-terminal", yaml_terminal_frame()
    for k, name in enumerate(EPISODES, 1):
        yield f"episode-{k}", episode_frame(k, name)
    for k in range(1, len(EPISODES)):
        yield f"next-{k}", next_frame(k)


def main() -> None:
    out = HERE / "frames"
    out.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    for name, frame in frames():
        if only and not any(name.startswith(o) for o in only):
            continue
        (out / f"{name}.svg").write_text(frame.svg(), encoding="utf-8")
        print("wrote", name)
    if not only or "lower" in only:
        (out / "lower-third.svg").write_text(lower_third_svg(), encoding="utf-8")
        print("wrote lower-third")


if __name__ == "__main__":
    main()
