"""More banner directions for BLADE (issue #235), independent of the old
framework figure's palette.

* ``d`` "Code": near-black, faded columns of generated-algorithm code and a
  violet-to-amber convergence curve.
* ``e`` "Population": warm off-white, a faint landscape and a population of
  candidates that converges on the optimum over generations.
* ``f`` "Genealogy": deep green, an evolution tree of generated algorithms
  (nodes coloured by fitness), echoing the veins of the leaf mark.

Each as a README banner (1600x400) and a social preview (1280x640).
Run from the repository root:  uv run python docs/media/banner/make_banner_v2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FONT = (
    "Inter"
    if any("Inter" in f.name for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)
MONO = "DejaVu Sans Mono"

TITLE = "BLADE"
TAGLINE = "Benchmarking LLM-driven Automated Design and Evolution"
SUBLINE = "of iterative optimisation heuristics"

STYLES = {
    "d": dict(bg="#0b0d12", text="#ffffff", sub="#b8a9ff", dark=True),
    "e": dict(bg="#f7f5f0", text="#1b1f24", sub="#d9573b", dark=False),
    "f": dict(bg="#0d2b24", text="#ffffff", sub="#8ee6c0", dark=True),
    # F's genealogy tree in D's colours (chosen direction, #235).
    "g": dict(bg="#0b0d12", text="#ffffff", sub="#b8a9ff", dark=True),
}

CODE = [
    "class AdaptiveDE:",
    "    def __init__(self, budget, dim):",
    "        self.F, self.CR = 0.5, 0.9",
    "    def __call__(self, func):",
    "        pop = rng.uniform(lb, ub, (n, dim))",
    "        sigma *= np.exp(cs / ds * (ps - 1))",
    "        x = np.clip(m + sigma * z, lb, ub)",
    "        if f < self.f_opt:",
    "            self.f_opt, self.x_opt = f, x",
    "        C = (1 - c1) * C + c1 * np.outer(p, p)",
    "        trial = np.where(mask, mutant, x)",
    "        archive.append(best)",
    "        T *= 0.95  # cooling",
    "        idx = np.argsort(fitness)[:mu]",
    "        w = np.log(mu + .5) - np.log(np.arange(1, mu+1))",
    "        return self.f_opt, self.x_opt",
]


def leaf(dark: bool) -> np.ndarray:
    path = ROOT / ("logo-dark.png" if dark else "logo.png")
    image = Image.open(path).convert("RGBA")
    return np.asarray(image.crop((0, 0, image.width, 452)))


def landscape(n=300):
    x = np.linspace(-3.2, 3.2, n)
    X, Y = np.meshgrid(x, x)
    Z = 0.35 * (X**2 + Y**2) - 1.2 * np.cos(2.2 * X) * np.cos(2.0 * Y)
    return X, Y, Z + 0.6 * np.sin(1.3 * X + 0.4)


def fade(fig, left, width, color):
    grad = np.zeros((2, 256, 4))
    grad[..., :3] = to_rgb(color)
    grad[..., 3] = np.linspace(1, 0, 256) ** 1.5
    ax = fig.add_axes([left, 0, width, 1], zorder=5)
    ax.imshow(grad, aspect="auto", extent=(0, 1, 0, 1))
    ax.axis("off")


def art_code(fig, left, style, wide):
    ax = fig.add_axes([left, 0, 1 - left, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(7)
    cols = 2
    rows = 12 if wide else 22
    size = 8.5 if wide else 10
    chars = 38 if wide else 27  # keeps the columns apart
    for c in range(cols):
        offset = rng.integers(len(CODE))
        for r in range(rows):
            line = CODE[(offset + r) % len(CODE)][:chars]
            ax.text(
                0.02 + c * (0.98 / cols),
                1 - (r + 0.5) / rows,
                line,
                family=MONO,
                fontsize=size,
                color="#8f86c9",
                alpha=float(rng.uniform(0.12, 0.42)),
                va="center",
            )
    # Violet-to-amber convergence curve.
    k = 60
    x = np.linspace(0.05, 0.95, k)
    rng = np.random.default_rng(3)
    y = np.maximum.accumulate(
        np.clip(1 - np.exp(-3.2 * x) + rng.normal(0, 0.02, k), 0, 1)
    )
    y = 0.12 + 0.72 * y / y.max()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list("va", ["#7c5cff", "#ff7ab6", "#ffb547"])
    lc = LineCollection(segments, cmap=cmap, linewidth=4, zorder=7)
    lc.set_array(np.linspace(0, 1, k))
    ax.add_collection(lc)
    ax.scatter(x[::6], y[::6], s=30, color="white", zorder=8, edgecolor="none")


def art_population(fig, left, style, wide):
    ax = fig.add_axes([left, 0, 1 - left, 1])
    ax.axis("off")
    X, Y, Z = landscape()
    ax.contourf(X, Y, Z, levels=14, cmap="Greys", alpha=0.18)
    ax.contour(X, Y, Z, levels=14, colors="#1b1f24", linewidths=0.4, alpha=0.18)
    rng = np.random.default_rng(5)
    optimum = np.array([0.0, 0.0])
    cmap = LinearSegmentedColormap.from_list("pop", ["#f3c2b4", "#d9573b", "#7a2416"])
    gens = 7
    for g in range(gens):
        spread = 2.6 * (1 - g / gens) ** 1.6 + 0.12
        pts = optimum + rng.normal(0, spread, (18, 2))
        pts = np.clip(pts, -3.1, 3.1)
        ax.scatter(
            pts[:, 0], pts[:, 1], s=34 - 2 * g, color=cmap(g / (gens - 1)),
            alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3,
        )
    ax.scatter([0], [0], marker="*", s=380, color="#1b1f24", zorder=4,
               edgecolor="white", linewidth=1.0)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)


def art_genealogy(fig, left, style, wide, palette=None):
    ax = fig.add_axes([left, 0, 1 - left, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(11)
    # Grow a tree from the bottom: each node has a parent in the previous
    # generation and a fitness that tends to improve.
    gens = 9 if wide else 11
    nodes = [(0.5, 0.08, 0.1, None)]  # x, y, fitness, parent index
    layer = [0]
    for g in range(1, gens):
        new_layer = []
        for _ in range(int(rng.integers(3, 6))):
            parent = int(rng.choice(layer))
            px, py, pf, _ = nodes[parent]
            x = float(np.clip(px + rng.normal(0, 0.08), 0.04, 0.96))
            y = 0.08 + 0.84 * g / (gens - 1)
            f = float(np.clip(pf + rng.normal(0.08, 0.1), 0, 1))
            nodes.append((x, y, f, parent))
            new_layer.append(len(nodes) - 1)
        layer = new_layer
    best = max(range(len(nodes)), key=lambda i: nodes[i][2])
    path = set()
    i = best
    while i is not None:
        path.add(i)
        i = nodes[i][3]
    palette = palette or dict(
        nodes=["#2f6b58", "#8ee6c0", "#f5d76e"], edge="#8ee6c0", best=None,
        best_color="#f5d76e", bg="#0d2b24",
    )
    cmap = LinearSegmentedColormap.from_list("fit", palette["nodes"])
    best_cmap = (
        LinearSegmentedColormap.from_list("best", palette["best"])
        if palette["best"]
        else None
    )
    for i, (x, y, f, parent) in enumerate(nodes):
        if parent is None:
            continue
        px, py, _, _ = nodes[parent]
        on_path = i in path and parent in path
        if on_path and best_cmap is not None:
            color = best_cmap(y)  # violet at the root, amber at the best
        else:
            color = palette["best_color"] if on_path else palette["edge"]
        ax.plot([px, x], [py, y], color=color,
                lw=2.8 if on_path else 0.9, alpha=0.95 if on_path else 0.35,
                zorder=2 if not on_path else 4, solid_capstyle="round")
    xs, ys, fs, _ = zip(*nodes)
    ax.scatter(xs, ys, s=[40 + 90 * f for f in fs], c=[cmap(f) for f in fs],
               edgecolor=palette["bg"], linewidth=1.0, zorder=5)


def art_genealogy_code_colours(fig, left, style, wide):
    art_genealogy(
        fig, left, style, wide,
        palette=dict(
            nodes=["#3b2f7a", "#7c5cff", "#ff7ab6", "#ffb547"],
            edge="#8f86c9",
            best=["#7c5cff", "#ff7ab6", "#ffb547"],
            best_color="#ffb547",
            bg="#0b0d12",
        ),
    )


ART = {
    "d": art_code,
    "e": art_population,
    "f": art_genealogy,
    "g": art_genealogy_code_colours,
}


def draw(concept, width, height, path, scale=1):
    """Draw ``concept`` at ``width`` x ``height`` (times ``scale`` pixels)."""
    style = STYLES[concept]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi * scale)
    fig.patch.set_facecolor(style["bg"])
    wide = width / height > 3
    left = 0.64 if wide else 0.62
    ART[concept](fig, left, style, wide)
    fade(fig, left, 0.12, style["bg"])

    mark = leaf(style["dark"])
    mark_h = 0.62 if wide else 0.40
    mark_w = mark_h * height / width * mark.shape[1] / mark.shape[0]
    mx = 0.035 if wide else 0.06
    my = (1 - mark_h) / 2 if wide else 0.43
    mark_ax = fig.add_axes([mx, my, mark_w, mark_h], zorder=9)
    mark_ax.imshow(mark)
    mark_ax.axis("off")

    ax = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.patch.set_alpha(0)
    x = mx + mark_w + (0.025 if wide else 0.04)
    if wide:
        y = 0.47
        ax.text(x, y, TITLE, fontsize=64, fontweight="bold", color=style["text"],
                family=FONT, va="bottom")
        ax.text(x, y - 0.04, TAGLINE, fontsize=17, color=style["text"],
                family=FONT, va="top")
        ax.text(x, y - 0.15, SUBLINE, fontsize=17, color=style["sub"],
                family=FONT, va="top")
    else:
        y = 0.66
        ax.text(x, y, TITLE, fontsize=78, fontweight="bold", color=style["text"],
                family=FONT, va="bottom")
        for i, line in enumerate(
            ["Benchmarking LLM-driven", "Automated Design and Evolution"]
        ):
            ax.text(x, y - 0.03 - 0.075 * i, line, fontsize=22,
                    color=style["text"], family=FONT, va="top")
        ax.text(x, y - 0.18, SUBLINE, fontsize=22, color=style["sub"],
                family=FONT, va="top")
        ax.text(0.06, 0.09, "github.com/XAI-liacs/BLADE  ·  pip install iohblade",
                fontsize=16, color=style["sub"], family=FONT)
    fig.savefig(path, dpi=dpi * scale, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    for concept in STYLES:
        draw(concept, 1600, 400, OUT / f"banner-{concept}-readme.png")
        draw(concept, 1280, 640, OUT / f"banner-{concept}-social.png")
    # The chosen banner (#235): README banner at 2x, GitHub social preview.
    draw("g", 1600, 400, ROOT / "docs" / "media" / "blade-banner.png", scale=2)
    draw("g", 1280, 640, ROOT / "docs" / "media" / "blade-social-preview.png")


if __name__ == "__main__":
    main()
