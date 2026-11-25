"""Mockups of a new BLADE platform figure, in the banner's palette.

Three layouts of the same content:

1. ``pipeline``: left to right, Define -> Components -> Run -> Record ->
   Analyse.
2. ``loop``: the generation loop (method, LLM, isolated evaluation) in the
   centre, with definition, components, execution and analysis around it.
3. ``stack``: horizontal layers, from interfaces at the top to analysis at
   the bottom.

Run from the repository root:
    uv run python docs/media/figure/make_platform_mockups.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FONT = (
    "Inter"
    if any("Inter" in f.name for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)
MONO = "DejaVu Sans Mono"

INK = "#14161c"
MUTED = "#5b5f6b"
LINE = "#d9dbe3"
BG = "#ffffff"
C = {  # accent, tint, text-safe accent
    "violet": ("#7c5cff", "#f0ecff", "#5a3de0"),
    "pink": ("#ff7ab6", "#ffecf4", "#c83c7f"),
    "amber": ("#ffb547", "#fff4e2", "#b8730a"),
    "grey": ("#8a8f9c", "#f3f4f7", "#4a4f5c"),
}

# Content -------------------------------------------------------------------
DEFINE = [
    "YAML experiment",
    "iohblade init · validate · check · run",
    "Python API",
    "Coding-agent skill",
]
PROBLEMS = [
    "BBOB function groups · MA-BBOB",
    "AlphaEvolve math (17)",
    "Steiner · TSP · CVRP · colouring",
    "Photonics · AutoML · kernel tuning",
    "Your own problem",
]
SUITES = ["quick", "continuous_bbo", "alphaevolve_math", "combinatorial", "full"]
METHODS = [
    "LLaMEA · EoH · ReEvo",
    "FunSearch · ShinkaEvolve",
    "LHNS · MCTS-AHD · MoEH",
    "Random search · your own",
]
LLMS = [
    "OpenAI · Claude · Gemini · DeepSeek",
    "Ollama · vLLM · LM Studio · MLX",
    "multi: several LLMs or servers",
]
RUN = [
    "Seeds · parallel runs",
    "Isolated, cached eval envs",
    "Exact budgets · timeouts",
    "HPO with SMAC",
]
RECORD = [
    "File · MLflow · Trackio",
    "Configs, sources & hashes",
    "Prompts, code & feedback",
]
ANALYSE = [
    "Web app",
    "Convergence & test scores",
    "Behaviour metrics",
    "Code evolution trees",
    "Human baselines",
    "Leaderboard (coming)",
]


def leaf(size_px: int) -> np.ndarray:
    image = Image.open(ROOT / "logo.png").convert("RGBA")
    return np.asarray(image.crop((0, 0, image.width, 452)))


def new_figure(width=2400, height=1150):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, color="grey", fill=True, lw=1.6, radius=18):
    accent, tint, _ = C[color]
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=tint if fill else "none", edgecolor=accent, linewidth=lw,
    )
    ax.add_patch(patch)


def title(ax, x, y, text, color="grey", size=19, number=None):
    _, _, strong = C[color]
    if number is not None:
        ax.add_patch(plt.Circle((x + 14, y - 12), 15, color=C[color][0]))
        ax.text(x + 14, y - 13, str(number), ha="center", va="center",
                fontsize=13, color="white", fontweight="bold", family=FONT)
        x += 40
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=strong,
            family=FONT, va="top")


def items(ax, x, y, lines, size=13.5, gap=30, color=INK, mono=False):
    for i, line in enumerate(lines):
        ax.text(x, y - i * gap, line, fontsize=size, color=color,
                family=MONO if mono else FONT, va="top")


def chips(ax, x, y, labels, color="grey", size=12.5, max_w=None, gap=10):
    """Rounded labels in rows starting at (x, y), wrapping at ``max_w``."""
    accent, tint, strong = C[color]
    cx, cy = x, y
    for label in labels:
        w = 0.6 * size * 100 / 72 * len(label) + 26
        if max_w is not None and cx + w > x + max_w:
            cx, cy = x, cy - 42
        ax.add_patch(FancyBboxPatch(
            (cx, cy - 30), w, 30, boxstyle="round,pad=0,rounding_size=15",
            facecolor="white", edgecolor=accent, linewidth=1.2,
        ))
        ax.text(cx + w / 2, cy - 15, label, ha="center", va="center",
                fontsize=size, color=strong, family=MONO)
        cx += w + gap
    return cy - 30


def arrow(ax, p, q, color=MUTED, lw=2.2, rad=0.0, style="-|>", ms=22):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4,
    ))


def header(ax, fig, width, height, subtitle):
    mark = leaf(60)
    h = 70
    w = h * mark.shape[1] / mark.shape[0]
    mark_ax = fig.add_axes([40 / width, (height - 40 - h) / height,
                            w / width, h / height])
    mark_ax.imshow(mark)
    mark_ax.axis("off")
    ax.text(40 + w + 18, height - 42, "BLADE", fontsize=34, fontweight="bold",
            color=INK, family=FONT, va="top")
    ax.text(40 + w + 20, height - 104, subtitle, fontsize=15, color=MUTED,
            family=FONT, va="top")


# 1. Pipeline -----------------------------------------------------------------
def pipeline(path):
    W, H = 2400, 900
    fig, ax = new_figure(W, H)
    header(ax, fig, W, H,
           "Benchmarking LLM-driven automated algorithm design: "
           "from experiment definition to analysis")
    top, bottom = 740, 40
    cols = [  # x, width
        (40, 400), (490, 860), (1400, 290), (1740, 280), (2070, 290),
    ]
    names = ["Define", "Components", "Run", "Record", "Analyse"]
    colors = ["grey", "violet", "amber", "grey", "pink"]
    for i, ((x, w), name, color) in enumerate(zip(cols, names, colors)):
        box(ax, x, bottom, w, top - bottom, color, fill=(color != "violet"))
        title(ax, x + 22, top - 22, name, color, number=i + 1)
        if i < len(cols) - 1:
            nx = cols[i + 1][0]
            arrow(ax, (x + w + 4, (top + bottom) / 2),
                  (nx - 4, (top + bottom) / 2), lw=3)
    define = ["YAML experiment", "iohblade CLI:",
              "  init · validate · check · run", "Python API",
              "Coding-agent skill"]
    items(ax, 62, top - 90, define, size=14, gap=36)
    ax.text(62, top - 300, "experiment.yml", fontsize=13, color=MUTED,
            family=MONO, va="top")
    code = ["llms:", "  qwen: {provider: openai}", "methods:",
            "  - type: shinka_evolve", "problems:", "  - suite: full"]
    items(ax, 62, top - 330, code, size=12, gap=24, color=INK, mono=True)

    # Components: four sub-boxes.
    x0, w0 = cols[1]
    sub_w = (w0 - 3 * 22) / 2
    sub = [
        ("Problems", PROBLEMS, "amber", (x0 + 22, 375, sub_w, 290)),
        ("Methods", METHODS, "violet", (x0 + 44 + sub_w, 375, sub_w, 290)),
        ("LLMs", LLMS, "pink", (x0 + 22, 80, sub_w, 270)),
        ("Default suites", [], "amber", (x0 + 44 + sub_w, 80, sub_w, 270)),
    ]
    for name, lines, color, (x, y, w, h) in sub:
        box(ax, x, y, w, h, color)
        title(ax, x + 20, y + h - 18, name, color, size=17)
        items(ax, x + 20, y + h - 66, lines, size=13.5, gap=36)
    sx, sy, sw, sh = sub[3][3]
    chips(ax, sx + 20, sy + sh - 66, SUITES, "amber", max_w=sw - 40,
          size=11.5)
    ax.text(sx + 20, sy + 80, "dim · budget_factor · use_best\n"
            "set once for the whole suite", fontsize=13, color=MUTED,
            family=FONT, va="top")
    ax.text(x0 + w0 / 2, 52, "built-in by registry key, or your own class",
            ha="center", fontsize=13, color=MUTED, family=FONT)

    for (x, w), lines in zip(cols[2:], [RUN, RECORD, ANALYSE]):
        items(ax, x + 22, top - 90, lines, size=14, gap=44)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


# 2. Loop ---------------------------------------------------------------------
def loop(path):
    W, H = 2400, 1300
    fig, ax = new_figure(W, H)
    header(ax, fig, W, H,
           "Each generation: a method prompts an LLM, the generated algorithm "
           "is evaluated in isolation, the feedback steers the next prompt")
    cx, cy, r = 1200, 640, 270
    nodes = {
        "method": (cx, cy + r, "Method", "violet"),
        "llm": (cx + r * 0.95, cy - r * 0.62, "LLM", "pink"),
        "eval": (cx - r * 0.95, cy - r * 0.62, "Isolated evaluation", "amber"),
    }
    for key, (x, y, name, color) in nodes.items():
        w = 330 if key == "eval" else 240
        box(ax, x - w / 2, y - 45, w, 90, color, radius=45, lw=2.4)
        ax.text(x, y, name, ha="center", va="center", fontsize=20,
                fontweight="bold", color=C[color][2], family=FONT)
    m, l, e = nodes["method"], nodes["llm"], nodes["eval"]
    arrow(ax, (m[0] + 130, m[1] - 20), (l[0] + 10, l[1] + 50), rad=-0.3,
          color=C["violet"][0], lw=3)
    arrow(ax, (l[0] - 130, l[1] - 30), (e[0] + 170, e[1] - 30), rad=-0.3,
          color=C["pink"][0], lw=3)
    arrow(ax, (e[0] - 10, e[1] + 50), (m[0] - 130, m[1] - 20), rad=-0.3,
          color=C["amber"][0], lw=3)
    ax.text(cx + 250, cy + 150, "prompt with parents\n& feedback",
            fontsize=14, color=C["violet"][2], family=FONT, ha="left")
    ax.text(cx, cy - r - 20, "algorithm code", fontsize=14,
            color=C["pink"][2], family=FONT, ha="center", va="top")
    ax.text(cx - 250, cy + 150, "fitness, errors\n& feedback", fontsize=14,
            color=C["amber"][2], family=FONT, ha="right")
    ax.text(cx, cy + 20, "× budget", fontsize=26, fontweight="bold",
            color=INK, family=FONT, ha="center", va="center")
    ax.text(cx, cy - 25, "generations per run", fontsize=14, color=MUTED,
            family=FONT, ha="center", va="center")

    # Satellites.
    def panel(x, y, w, h, name, lines, color, size=13.5, gap=34):
        box(ax, x, y, w, h, color)
        title(ax, x + 20, y + h - 18, name, color, size=17)
        items(ax, x + 20, y + h - 66, lines, size=size, gap=gap)

    panel(40, 820, 470, 250, "Define", DEFINE, "grey")
    panel(840, 1000 - 40, 720, 200, "Methods", [
        "LLaMEA · EoH · ReEvo · FunSearch · ShinkaEvolve",
        "LHNS · MCTS-AHD · MoEH · random search · your own"], "violet")
    panel(1830, 520, 530, 280, "LLMs", LLMS + ["   (local or hosted)"], "pink")
    panel(40, 300, 540, 380, "Problems & suites", PROBLEMS[:4], "amber")
    chips(ax, 60, 400, SUITES, "amber", max_w=500, size=11.5)
    panel(640, 40, 560, 230, "Run", ["Seeds · parallel runs · resume",
                                     "Cached eval envs · timeouts",
                                     "Exact budgets · HPO with SMAC"], "amber")
    panel(1240, 40, 520, 230, "Record", RECORD, "grey")
    panel(1830, 40, 530, 440, "Analyse", ANALYSE, "pink", gap=56)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


# 3. Stack --------------------------------------------------------------------
def stack(path):
    W, H = 2400, 1130
    fig, ax = new_figure(W, H)
    header(ax, fig, W, H, "Layers of the platform, from interfaces to analysis")
    notes = [
        "how you describe and launch an experiment",
        "built-in benchmarks, or your own Problem class",
        "named sets of problems with fixed settings",
        "LLM-driven algorithm design methods",
        "hosted APIs or local servers, alone or combined",
        "how generated code is run safely and fairly",
        "everything needed to reproduce a run",
        "compare methods and inspect generated algorithms",
    ]
    layers = [
        ("Interfaces", "grey", ["YAML experiments", "iohblade CLI",
                                "Python API", "coding-agent skill",
                                "templates: problem / method"]),
        ("Problems", "amber", ["BBOB groups", "MA-BBOB", "AlphaEvolve math",
                               "Steiner", "TSP", "CVRP", "graph colouring",
                               "photonics", "AutoML", "kernel tuning",
                               "your own"]),
        ("Suites", "amber", SUITES + ["dim", "budget_factor", "use_best"]),
        ("Methods", "violet", ["LLaMEA", "EoH", "ReEvo", "FunSearch",
                               "ShinkaEvolve", "LHNS", "MCTS-AHD", "MoEH",
                               "random search", "your own"]),
        ("LLMs", "pink", ["OpenAI", "Claude", "Gemini", "DeepSeek", "Ollama",
                          "vLLM / SGLang", "LM Studio", "MLX", "multi"]),
        ("Execution", "amber", ["seeds", "parallel runs", "resume",
                                "isolated cached envs", "timeouts",
                                "exact budgets", "harness checks",
                                "HPO (SMAC)"]),
        ("Results", "grey", ["file logger", "MLflow", "Trackio",
                             "configs & hashes", "prompts & code"]),
        ("Analysis", "pink", ["web app", "convergence", "test scores",
                              "behaviour metrics", "code evolution",
                              "human baselines", "leaderboard (coming)"]),
    ]
    top = H - 140
    row_h = 118
    for i, (name, color, labels) in enumerate(layers):
        y = top - (i + 1) * row_h
        box(ax, 40, y + 8, W - 80, row_h - 16, color, radius=16)
        ax.text(70, y + row_h / 2, name, fontsize=19, fontweight="bold",
                color=C[color][2], family=FONT, va="center")
        chips(ax, 300, y + row_h / 2 + 17, labels, color, size=13.5,
              max_w=W - 860)
        ax.text(W - 70, y + row_h / 2, notes[i], fontsize=14, color=MUTED,
                family=FONT, va="center", ha="right", style="italic")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def main():
    pipeline(OUT / "mockup-1-pipeline.png")
    loop(OUT / "mockup-2-loop.png")
    stack(OUT / "mockup-3-stack.png")


if __name__ == "__main__":
    main()
