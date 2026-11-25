"""Draft banners for BLADE (issue #235): logo + optimisation landscape.

Writes three concepts, each as a README banner (1600x400) and a GitHub
social preview (1280x640), to ``docs/media/banner/``:

* ``a``: deep navy, white leaf mark, contour lines of a multimodal landscape
  in teal and a green convergence curve with generation markers.
* ``b``: light variant of ``a`` (navy text on off-white), for light pages.
* ``c``: navy-to-teal gradient with a line-art 3D surface and the curve.

Run from the repository root:  uv run python docs/media/banner/make_banner.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

NAVY = "#0e2a3f"
DEEP = "#123a55"
TEAL = "#1a6a8c"
LIGHT_BLUE = "#8fc4e6"
GREEN = "#3aa35b"
LIGHT_GREEN = "#9fd6a5"
OFF_WHITE = "#f6f8f7"

TITLE = "BLADE"
TAGLINE = "Benchmarking LLM-driven Automated Design and Evolution"
SUBLINE = "of iterative optimisation heuristics"

FONT = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"


def leaf(dark_background: bool) -> np.ndarray:
    """The leaf mark of the logo (without the wordmark), as RGBA."""
    path = ROOT / ("logo-dark.png" if dark_background else "logo.png")
    image = Image.open(path).convert("RGBA")
    return np.asarray(image.crop((0, 0, image.width, 452)))


def landscape(n: int = 400):
    """A smooth multimodal landscape (Rastrigin-like with a global trend)."""
    x = np.linspace(-3.2, 3.2, n)
    y = np.linspace(-3.2, 3.2, n)
    X, Y = np.meshgrid(x, y)
    Z = (
        0.35 * (X**2 + Y**2)
        - 1.2 * np.cos(2.2 * X) * np.cos(2.0 * Y)
        + 0.6 * np.sin(1.3 * X + 0.4)
    )
    return X, Y, Z


def convergence(k: int = 22, seed: int = 3):
    """A best-so-far curve over generations (rising, with plateaus)."""
    rng = np.random.default_rng(seed)
    steps = np.maximum(rng.exponential(0.08, k) * (rng.random(k) < 0.55), 0)
    best = np.clip(np.cumsum(steps) + 0.08, 0, 0.95)
    return np.linspace(0, 1, k), best / best.max() * 0.85 + 0.05


def text_block(ax, x, y, color, sub_color, wide):
    """Wordmark, tagline and subline, left-aligned at (x, y)."""
    if wide:
        ax.text(x, y, TITLE, fontsize=64, fontweight="bold", color=color,
                family=FONT, va="bottom", transform=ax.transAxes)
        ax.text(x, y - 0.04, TAGLINE, fontsize=17, color=color, family=FONT,
                va="top", transform=ax.transAxes)
        ax.text(x, y - 0.15, SUBLINE, fontsize=17, color=sub_color,
                family=FONT, va="top", transform=ax.transAxes)
    else:
        ax.text(x, y, TITLE, fontsize=78, fontweight="bold", color=color,
                family=FONT, va="bottom", transform=ax.transAxes)
        lines = ["Benchmarking LLM-driven", "Automated Design and Evolution"]
        for i, line in enumerate(lines):
            ax.text(x, y - 0.03 - 0.075 * i, line, fontsize=22, color=color,
                    family=FONT, va="top", transform=ax.transAxes)
        ax.text(x, y - 0.03 - 0.075 * 2, SUBLINE, fontsize=22,
                color=sub_color, family=FONT, va="top",
                transform=ax.transAxes)


def fade(fig, left, width, color):
    """A horizontal fade from ``color`` (opaque) to transparent."""
    from matplotlib.colors import to_rgb

    rgb = to_rgb(color[:3] if isinstance(color, tuple) else color)
    grad = np.zeros((2, 256, 4))
    grad[..., :3] = rgb
    grad[..., 3] = np.linspace(1, 0, 256) ** 1.5
    ax = fig.add_axes([left, 0, width, 1])
    ax.imshow(grad, aspect="auto", extent=(0, 1, 0, 1), zorder=5)
    ax.axis("off")


def draw(concept: str, width: int, height: int, path: Path) -> None:
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    wide = width / height > 3
    dark = concept in ("a", "c")
    background = NAVY if dark else OFF_WHITE

    # Background.
    if concept == "c":
        cmap = LinearSegmentedColormap.from_list("bg", [NAVY, NAVY, DEEP, TEAL])
        grad = np.linspace(0, 1, 256)[None, :]
        ax.imshow(grad, extent=(0, 1, 0, 1), aspect="auto", cmap=cmap, zorder=0)
    else:
        fig.patch.set_facecolor(background)

    # Landscape on the right, faded into the background.
    X, Y, Z = landscape()
    left = 0.64 if wide else 0.62
    inset = fig.add_axes([left, 0.0, 1 - left, 1.0])
    inset.axis("off")
    inset.set_facecolor("none")
    if concept in ("a", "b"):
        line = LIGHT_BLUE if dark else TEAL
        inset.contour(X, Y, Z, levels=18, colors=line, linewidths=0.9,
                      alpha=0.55 if dark else 0.45)
        inset.set_xlim(-3.2, 3.2)
        inset.set_ylim(-3.2, 3.2)
    else:
        rows = range(0, Z.shape[0], 14)
        for i, row in enumerate(rows):
            zs = Z[row, ::2]
            xs = np.linspace(0, 1, zs.size)
            inset.plot(xs, 0.06 + i * 0.031 + 0.035 * zs, color=LIGHT_BLUE,
                       lw=0.8, alpha=0.2 + 0.5 * i / len(rows))
        inset.set_xlim(0, 1)
        inset.set_ylim(0, 1.0)
    if concept == "c":
        cmap = LinearSegmentedColormap.from_list("bg", [NAVY, NAVY, DEEP, TEAL])
        fade(fig, left, 0.12, cmap(left))
    else:
        fade(fig, left, 0.12, background)

    # Convergence curve with generation markers, over the landscape.
    curve = fig.add_axes([left + 0.07, 0.14, 1 - left - 0.11, 0.72])
    curve.axis("off")
    curve.set_facecolor("none")
    gx, gy = convergence()
    curve.step(gx, gy, where="post", color=GREEN, lw=3.2 if dark else 2.8,
               zorder=6)
    curve.scatter(gx, gy, s=26, color=LIGHT_GREEN if dark else GREEN,
                  edgecolor=background if concept != "c" else DEEP,
                  linewidth=1.0, zorder=7)
    curve.set_xlim(-0.02, 1.02)
    curve.set_ylim(0, 1)

    # Leaf mark and text on the left.
    mark = leaf(dark)
    mark_h = 0.62 if wide else 0.40
    mark_w = mark_h * height / width * mark.shape[1] / mark.shape[0]
    mx = 0.035 if wide else 0.06
    my = (1 - mark_h) / 2 if wide else 0.43
    mark_ax = fig.add_axes([mx, my, mark_w, mark_h])
    mark_ax.imshow(mark)
    mark_ax.axis("off")
    text_color = "white" if dark else NAVY
    sub_color = LIGHT_BLUE if dark else TEAL
    # Text on its own top layer, above the landscape and its fade.
    ax = fig.add_axes([0, 0, 1, 1], zorder=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.patch.set_alpha(0)
    if wide:
        text_block(ax, mx + mark_w + 0.025, 0.47, text_color, sub_color, True)
    else:
        text_block(ax, mx + mark_w + 0.04, 0.66, text_color, sub_color, False)
        ax.text(0.06, 0.09,
                "github.com/XAI-liacs/BLADE  ·  pip install iohblade",
                fontsize=16, color=sub_color, family=FONT,
                transform=ax.transAxes)

    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    for concept in ("a", "b", "c"):
        draw(concept, 1600, 400, OUT / f"banner-{concept}-readme.png")
        draw(concept, 1280, 640, OUT / f"banner-{concept}-social.png")


if __name__ == "__main__":
    main()
