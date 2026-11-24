"""LinkedIn cards for the BLADE campaign (issue #235).

Dark cards in the banner's style, each in two formats:

* ``portrait``: 1080x1350 (4:5), the largest image LinkedIn shows in the feed;
* ``landscape``: 1200x627, the link-share format.

Writes SVGs to ``docs/media/campaign/``; render PNGs with
``node docs/media/campaign/render.js``.

Run from the repository root:
    uv run python docs/media/campaign/make_cards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figure"))

import make_platform_figure as F  # noqa: E402

F.set_theme("dark")

WHITE = "#ffffff"
SOFT = "#b8a9ff"  # the banner's subline colour
GRADIENT = ("#7c5cff", "#ff7ab6", "#ffb547")
FOOTER = "github.com/XAI-liacs/BLADE  ·  pip install iohblade"


def start(width: int, height: int) -> None:
    F.parts.clear()
    F.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
          f'height="{height}" viewBox="0 0 {width} {height}">')
    F.add(
        "<defs>"
        '<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{GRADIENT[0]}"/>'
        f'<stop offset=".5" stop-color="{GRADIENT[1]}"/>'
        f'<stop offset="1" stop-color="{GRADIENT[2]}"/>'
        "</linearGradient>"
        '<radialGradient id="glow" cx=".85" cy=".1" r=".7">'
        '<stop offset="0" stop-color="#7c5cff" stop-opacity=".22"/>'
        '<stop offset="1" stop-color="#7c5cff" stop-opacity="0"/>'
        "</radialGradient>"
        "</defs>"
    )
    F.add(f'<rect width="{width}" height="{height}" fill="{F.BG}"/>')
    F.add(f'<rect width="{width}" height="{height}" fill="url(#glow)"/>')


def finish() -> str:
    F.add("</svg>")
    return "\n".join(F.parts)


def brand(x: int, y: int, size: float = 1.0) -> None:
    """Leaf mark and wordmark, top-left at (x, y)."""
    h = 54 * size
    F.image(x, y, h * 0.9, h, F.leaf_uri())
    F.text(x + h * 0.9 + 14, y + h * 0.78, "BLADE", 34 * size, WHITE, 800)


def pill(x: int, y: int, label: str, anchor: str = "start") -> None:
    w = 0.58 * 17 * len(label) + 36
    if anchor == "end":
        x -= w
    F.add(f'<rect x="{x}" y="{y}" width="{w}" height="40" rx="20" '
          f'fill="none" stroke="url(#accent)" stroke-width="2"/>')
    F.text(x + w / 2, y + 27, label, 17, WHITE, 600, "middle")


def headline(x: int, y: int, lines: list[str], size: int, gap: float = 1.15,
             accent_last: bool = True) -> float:
    for i, line in enumerate(lines):
        fill = "url(#accent)" if accent_last and i == len(lines) - 1 else WHITE
        F.add(f'<text x="{x}" y="{y + i * size * gap}" font-family="{F.FONT}" '
              f'font-size="{size}" font-weight="800" fill="{fill}" '
              f'letter-spacing="-1">{F.esc(line)}</text>')
    return y + (len(lines) - 1) * size * gap


def footer(width: int, height: int, index: str | None, pad: int = 64) -> None:
    F.add(f'<line x1="{pad}" y1="{height - 92}" x2="{width - pad}" '
          f'y2="{height - 92}" stroke="#262a36" stroke-width="1.5"/>')
    F.text(pad, height - 48, FOOTER, 19, SOFT, 500)
    if index:
        F.text(width - pad, height - 48, index, 19, F.MUTED, 600, "end")


def code_card(x, y, w, lines, title="experiment.yml", size=19, gap=31):
    h = 70 + gap * len(lines)
    F.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
          f'fill="#05060a" stroke="#2a2e3a" stroke-width="1.5"/>')
    for k, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        F.add(f'<circle cx="{x + 28 + k * 22}" cy="{y + 26}" r="7" fill="{c}"/>')
    F.text(x + 110, y + 32, title, 16, "#8f86c9", family=F.MONO)
    for k, (line, color) in enumerate(lines):
        F.text(x + 28, y + 76 + k * gap, line, size, color, family=F.MONO)
    return y + h


# Card 1: release announcement -------------------------------------------------
FEATURES = [
    ("YAML experiments", "init · validate · check · run"),
    ("Ten search methods", "LLaMEA, EoH, ReEvo, FunSearch, ShinkaEvolve, ..."),
    ("Default benchmark suites", "BBOB groups, MA-BBOB, AlphaEvolve maths, ..."),
    ("Any LLM, hosted or local", "OpenAI, Claude, Gemini, Ollama, vLLM, ..."),
    ("Built for coding agents", "iohblade skill"),
]


def release(fmt: str) -> str:
    if fmt == "portrait":
        W, H = 1080, 1350
        start(W, H)
        brand(64, 60)
        pill(W - 64, 66, "v1.2.0 released", "end")
        y = headline(64, 230, ["Benchmark LLMs that", "design optimisation",
                               "algorithms."], 70)
        F.text(64, y + 62, "BLADE 1.2 · a benchmark suite for LLM-driven "
               "algorithm design", 24, F.MUTED)
        F.mini_tree(700, 600, 320, 480)
        for k, (title, sub) in enumerate(FEATURES):
            yk = 600 + k * 120
            F.add(f'<circle cx="82" cy="{yk - 9}" r="8" fill="{GRADIENT[k % 3]}"/>')
            F.text(108, yk, title, 30, WHITE, 700)
            F.text(108, yk + 38, sub, 20, F.MUTED)
        footer(W, H, None)
    else:
        W, H = 1200, 627
        start(W, H)
        brand(56, 44, 0.85)
        pill(W - 56, 48, "v1.2.0 released", "end")
        y = headline(56, 190, ["Benchmark LLMs that", "design optimisation",
                               "algorithms."], 52)
        F.mini_tree(640, 140, 300, 250)
        for k, (title, _) in enumerate(FEATURES[:4]):
            yk = y + 64 + k * 36
            F.add(f'<circle cx="66" cy="{yk - 7}" r="6" fill="{GRADIENT[k % 3]}"/>')
            F.text(84, yk, title, 22, WHITE, 600)
        footer(W, H, None, pad=56)
    return finish()


# Card 2: YAML experiments ----------------------------------------------------
YAML = [
    ("llms:", "#b8a9ff"),
    ("  qwen: {provider: ollama, model: qwen3}", "#ffb547"),
    ("methods:", "#b8a9ff"),
    ("  - type: llamea", "#ff7ab6"),
    ("  - type: shinka_evolve", "#ff7ab6"),
    ("problems:", "#b8a9ff"),
    ("  - suite: continuous_bbo", "#8ee6c0"),
]
SHELL = [
    ("$ iohblade init my-exp", "#e6e6f0"),
    ("$ iohblade validate my-exp/experiment.yml", "#e6e6f0"),
    ("  ✓ configuration is valid", "#8ee6c0"),
    ("$ iohblade check my-exp/experiment.yml", "#e6e6f0"),
    ("  ✓ ready to run (no LLM calls used)", "#8ee6c0"),
    ("$ iohblade run my-exp/experiment.yml", "#e6e6f0"),
]


def yaml_card(fmt: str) -> str:
    if fmt == "portrait":
        W, H = 1080, 1350
        start(W, H)
        brand(64, 60)
        pill(W - 64, 66, "New in 1.2", "end")
        y = headline(64, 230, ["From idea to benchmark", "in four commands."], 66)
        F.text(64, y + 58, "Describe LLMs, methods and problems in one YAML "
               "file.", 24, F.MUTED)
        end = code_card(64, y + 110, W - 128, YAML, size=21, gap=34)
        end = code_card(64, end + 30, W - 128, SHELL, title="terminal",
                        size=20, gap=33)
        F.text(64, end + 72, "Built-in problems, methods and LLMs by name;",
               24, WHITE, 600)
        F.text(64, end + 106, "your own as a Python file.", 24, WHITE, 600)
        footer(W, H, "2 / 7")
    else:
        W, H = 1200, 627
        start(W, H)
        brand(56, 44, 0.85)
        pill(W - 56, 48, "New in 1.2", "end")
        y = headline(56, 196, ["From idea to", "benchmark in", "four commands."],
                     52)
        F.text(56, y + 50, "One YAML file. Errors with line numbers.", 20,
               F.MUTED)
        F.text(56, y + 80, "Smoke tests without LLM calls.", 20, F.MUTED)
        code_card(560, 120, 584, SHELL, title="terminal", size=16, gap=27)
        footer(W, H, "2 / 7", pad=56)
    return finish()


def main() -> None:
    for name, fn in (("01-release", release), ("02-yaml", yaml_card)):
        for fmt in ("portrait", "landscape"):
            path = HERE / f"card-{name}-{fmt}.svg"
            path.write_text(fn(fmt))
            print("wrote", path.name)


if __name__ == "__main__":
    main()
