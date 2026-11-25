"""LinkedIn cards for the BLADE campaign (issue #235).

Seven dark portrait cards (1080x1350, 4:5, the largest image LinkedIn shows
in the feed) in the banner's style. The post texts are in ``posts.md``.

Writes ``docs/media/campaign/card-NN-<name>.svg``; render the PNGs with
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

W, H = 1080, 1350
PAD = 64
TOTAL = 7
WHITE = "#ffffff"
SOFT = "#b8a9ff"  # the banner's subline colour
VIOLET, PINK, AMBER, MINT = "#7c5cff", "#ff7ab6", "#ffb547", "#8ee6c0"
FOOTER = "github.com/XAI-liacs/BLADE  ·  pip install iohblade"


# Frame -----------------------------------------------------------------------
def start() -> None:
    F.parts.clear()
    F.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
          f'viewBox="0 0 {W} {H}">')
    F.add(
        "<defs>"
        '<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{VIOLET}"/>'
        f'<stop offset=".5" stop-color="{PINK}"/>'
        f'<stop offset="1" stop-color="{AMBER}"/>'
        "</linearGradient>"
        '<radialGradient id="glow" cx=".85" cy=".1" r=".7">'
        f'<stop offset="0" stop-color="{VIOLET}" stop-opacity=".22"/>'
        f'<stop offset="1" stop-color="{VIOLET}" stop-opacity="0"/>'
        "</radialGradient>"
        "</defs>"
    )
    F.add(f'<rect width="{W}" height="{H}" fill="{F.BG}"/>')
    F.add(f'<rect width="{W}" height="{H}" fill="url(#glow)"/>')


def finish() -> str:
    F.add("</svg>")
    return "\n".join(F.parts)


def brand() -> None:
    F.image(PAD, 60, 49, 54, F.leaf_uri())
    F.text(PAD + 63, 102, "BLADE", 34, WHITE, 800)


def pill(label: str) -> None:
    w = 0.58 * 17 * len(label) + 36
    x = W - PAD - w
    F.add(f'<rect x="{x}" y="66" width="{w}" height="40" rx="20" fill="none" '
          f'stroke="url(#accent)" stroke-width="2"/>')
    F.text(x + w / 2, 93, label, 17, WHITE, 600, "middle")


def headline(lines: list[str], size: int = 66, y: int = 230) -> float:
    """Big title; the last line is filled with the accent gradient."""
    for i, line in enumerate(lines):
        fill = "url(#accent)" if i == len(lines) - 1 else WHITE
        F.add(f'<text x="{PAD}" y="{y + i * size * 1.15}" '
              f'font-family="{F.FONT}" font-size="{size}" font-weight="800" '
              f'fill="{fill}" letter-spacing="-1">{F.esc(line)}</text>')
    return y + (len(lines) - 1) * size * 1.15


def subline(y: float, s: str) -> None:
    F.text(PAD, y, s, 24, F.MUTED)


def footer(index: int) -> None:
    F.add(f'<line x1="{PAD}" y1="{H - 92}" x2="{W - PAD}" y2="{H - 92}" '
          f'stroke="#262a36" stroke-width="1.5"/>')
    F.text(PAD, H - 48, FOOTER, 19, SOFT, 500)
    F.text(W - PAD, H - 48, f"{index} / {TOTAL}", 19, F.MUTED, 600, "end")


def card_box(x, y, w, h, stroke="#2a2e3a", fill="#12151d"):
    F.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
          f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def code_card(x, y, w, lines, title="experiment.yml", size=21, gap=34):
    h = 70 + gap * len(lines)
    card_box(x, y, w, h, fill="#05060a")
    for k, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        F.add(f'<circle cx="{x + 28 + k * 22}" cy="{y + 26}" r="7" fill="{c}"/>')
    F.text(x + 110, y + 32, title, 16, "#8f86c9", family=F.MONO)
    for k, (line, color) in enumerate(lines):
        F.text(x + 28, y + 76 + k * gap, line, size, color, family=F.MONO)
    return y + h


def bullet(y, title, sub, color):
    F.add(f'<circle cx="{PAD + 18}" cy="{y - 9}" r="8" fill="{color}"/>')
    F.text(PAD + 44, y, title, 30, WHITE, 700)
    if sub:
        F.text(PAD + 44, y + 38, sub, 20, F.MUTED)


# 1 Release -------------------------------------------------------------------
def card_release() -> str:
    start()
    brand()
    pill("v1.2.0 released")
    y = headline(["Benchmark LLMs that", "design optimisation", "algorithms."], 70)
    subline(y + 62, "BLADE 1.2 · a benchmark suite for LLM-driven algorithm "
            "design")
    F.mini_tree(700, 600, 320, 480)
    items = [
        ("YAML experiments", "init · validate · check · run", VIOLET),
        ("Nine methods + your own", "LLaMEA, EoH, ReEvo, FunSearch, "
         "ShinkaEvolve, ...", PINK),
        ("Default benchmark suites", "BBOB groups, MA-BBOB, AlphaEvolve "
         "maths, ...", AMBER),
        ("Any LLM, hosted or local", "OpenAI, Claude, Gemini, Ollama, vLLM, ...",
         VIOLET),
        ("Built for coding agents", "iohblade skill", PINK),
    ]
    for k, (title, sub, color) in enumerate(items):
        bullet(600 + k * 120, title, sub, color)
    footer(1)
    return finish()


# 2 YAML ----------------------------------------------------------------------
def card_yaml() -> str:
    start()
    brand()
    pill("New in 1.2")
    y = headline(["From idea to benchmark", "in four commands."])
    subline(y + 58, "Describe LLMs, methods and problems in one YAML file.")
    yaml = [
        ("llms:", SOFT),
        ("  qwen: {provider: ollama, model: qwen3}", AMBER),
        ("methods:", SOFT),
        ("  - type: llamea", PINK),
        ("  - type: shinka_evolve", PINK),
        ("problems:", SOFT),
        ("  - suite: continuous_bbo", MINT),
    ]
    shell = [
        ("$ iohblade init my-exp", "#e6e6f0"),
        ("$ iohblade validate my-exp/experiment.yml", "#e6e6f0"),
        ("  ✓ configuration is valid", MINT),
        ("$ iohblade check my-exp/experiment.yml", "#e6e6f0"),
        ("  ✓ ready to run (no LLM calls used)", MINT),
        ("$ iohblade run my-exp/experiment.yml", "#e6e6f0"),
    ]
    end = code_card(PAD, y + 110, W - 2 * PAD, yaml)
    end = code_card(PAD, end + 30, W - 2 * PAD, shell, "terminal", 20, 33)
    F.text(PAD, end + 72, "Built-in problems, methods and LLMs by name;", 24,
           WHITE, 600)
    F.text(PAD, end + 106, "your own as a Python file.", 24, WHITE, 600)
    footer(2)
    return finish()


# 3 Suites --------------------------------------------------------------------
def card_suites() -> str:
    start()
    brand()
    pill("Default suites")
    y = headline(["Benchmarks with fixed", "settings, ready to run."])
    subline(y + 58, "Name a suite in your experiment; instances are fixed.")
    tiles = [
        (F.thumb_landscape, "bbob", "5 BBOB function groups",
         "10-D · train iid 1-5 · test iid 6-15"),
        (F.thumb_landscape, "ma_bbob", "MA-BBOB",
         "10-D · 25 train · 50 test instances"),
        (F.thumb_packing, "alphaevolve_math", "17 AlphaEvolve problems",
         "packing, Heilbronn, auto-correlation, ..."),
        (F.thumb_route, "combinatorial", "Steiner, TSP, CVRP, colouring",
         "6 combinatorial problems"),
    ]
    ty = y + 110
    tw, th, gap = (W - 2 * PAD - 24) / 2, 300, 24
    for k, (thumb, key, title, sub) in enumerate(tiles):
        tx = PAD + (k % 2) * (tw + gap)
        yy = ty + (k // 2) * (th + gap)
        card_box(tx, yy, tw, th)
        thumb(tx + 24, yy + 24, 104)
        F.chip(tx + 24, yy + 150, key, "amber", 15)
        F.text(tx + 24, yy + 224, title, 22, WHITE, 700)
        F.text(tx + 24, yy + 258, sub, 17, F.MUTED)
    yy = ty + 2 * (th + gap) + 20
    F.chip(PAD, yy, "quick", "amber", 16)
    F.text(PAD + 110, yy + 22, "a small set to try a setup", 20, F.MUTED)
    F.chip(PAD, yy + 50, "full", "amber", 16)
    F.text(PAD + 110, yy + 72, "everything; dim and budget_factor set the BBOB parts",
           20, F.MUTED)
    footer(3)
    return finish()


# 4 Methods -------------------------------------------------------------------
def card_methods() -> str:
    start()
    brand()
    pill("Methods")
    y = headline(["Nine methods.", "One fair harness."], 74)
    subline(y + 62, "Compare LLM-driven algorithm design methods on equal terms.")
    F.mini_tree(640, 470, 380, 400)
    methods = ["LLaMEA", "EoH", "ReEvo", "FunSearch", "ShinkaEvolve", "LHNS",
               "MCTS-AHD", "MoEH", "Random search"]
    for k, name in enumerate(methods):
        F.chip(PAD, 470 + k * 46, name, "violet", 17)
    rules = [
        ("Exactly the same budget", "every method gets the same number of "
         "evaluations"),
        ("Same LLM, same problems", "and the same isolated evaluation"),
        ("FunSearch and ShinkaEvolve", "reimplemented to run with any LLM"),
    ]
    for k, (title, sub) in enumerate(rules):
        bullet(950 + k * 92, title, sub, [VIOLET, PINK, AMBER][k])
    footer(4)
    return finish()


# 5 Local GPUs ----------------------------------------------------------------
def card_local() -> str:
    start()
    brand()
    pill("Local LLMs")
    y = headline(["Run it on your", "own GPUs."], 74)
    subline(y + 62, "Open models through Ollama, vLLM or LM Studio.")
    logos = [("qwen-color", "Qwen"), ("deepseek-color", "DeepSeek"),
             ("ollama", "Ollama"), ("vllm-color", "vLLM"),
             ("lmstudio", "LM Studio")]
    gx = (W - 2 * PAD) / len(logos)
    for k, (name, label) in enumerate(logos):
        cx = PAD + gx * (k + 0.5)
        F.add(f'<circle cx="{cx}" cy="{y + 160}" r="48" fill="#ffffff"/>')
        F.image(cx - 30, y + 130, 60, 60, F.icon_uri(name))
        F.text(cx, y + 240, label, 19, WHITE, 600, "middle")
    yaml = [
        ("experiment: {n_jobs: 32}", MINT),
        ("llms:", SOFT),
        ("  gpu0:", "#e6e6f0"),
        ("    provider: openai", AMBER),
        ("    model: Qwen/Qwen3-32B", AMBER),
        ('    params: {base_url: "http://localhost:8000/v1"}', AMBER),
        ("  # gpu1 to gpu3: the same on ports 8001-8003", F.MUTED),
        ("  qwen: {provider: multi, llms: [gpu0, gpu1, gpu2, gpu3]}", PINK),
    ]
    end = code_card(PAD, y + 290, W - 2 * PAD, yaml, size=19, gap=32)
    F.text(PAD, end + 62, "One vLLM server per GPU, combined with", 24, WHITE, 600)
    F.text(PAD, end + 96, "provider: multi. Runs share the servers.", 24, WHITE,
           600)
    footer(5)
    return finish()


# 6 Analysis ------------------------------------------------------------------
def card_analysis() -> str:
    start()
    brand()
    pill("Analysis")
    y = headline(["See what the LLM", "actually designed."])
    subline(y + 58, "Results, code and lineage of every generated algorithm.")
    top = y + 110
    tiles = [
        (F.mini_convergence, "Convergence & test scores"),
        (F.mini_boxes, "Compare methods across seeds"),
        (F.mini_tree, "Code evolution: who came from whom"),
        (F.mini_podium, "Leaderboard (coming soon)"),
    ]
    tw, th, gap = (W - 2 * PAD - 24) / 2, 320, 24
    for k, (fn, label) in enumerate(tiles):
        tx = PAD + (k % 2) * (tw + gap)
        yy = top + (k // 2) * (th + gap)
        card_box(tx, yy, tw, th)
        fn(tx + 40, yy + 34, tw - 80, 200)
        F.text(tx + tw / 2, yy + th - 32, label, 19, WHITE, 600, "middle")
    yy = top + 2 * (th + gap) + 40
    F.text(PAD, yy, "$ iohblade-webapp", 24, MINT, 600, family=F.MONO)
    F.text(PAD + 330, yy, "browse runs, prompts and code", 22, F.MUTED)
    footer(6)
    return finish()


# 7 Coding agent --------------------------------------------------------------
def bubble(x, y, w, lines, user: bool):
    h = 36 + 34 * len(lines)
    fill = "#1d1836" if user else "#12151d"
    stroke = VIOLET if user else "#2a2e3a"
    F.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" '
          f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    for k, (line, color, mono) in enumerate(lines):
        F.text(x + 26, y + 46 + k * 34, line, 19 if mono else 21, color,
               500 if not mono else 400, family=F.MONO if mono else F.FONT)
    return y + h


def card_agent() -> str:
    start()
    brand()
    pill("Coding agents")
    y = headline(["Set up a benchmark by", "asking your agent."])
    subline(y + 58, "iohblade skill teaches coding agents to use BLADE.")
    yy = y + 110
    F.text(PAD, yy, "$ iohblade skill", 24, MINT, 600, family=F.MONO)
    yy = bubble(PAD + 160, yy + 36, W - 2 * PAD - 160, [
        ("Benchmark my scheduling heuristic problem", WHITE, False),
        ("with LLaMEA and ShinkaEvolve on a local", WHITE, False),
        ("Qwen model, 5 runs each.", WHITE, False),
    ], user=True)
    yy = bubble(PAD, yy + 24, W - 2 * PAD - 120, [
        ("I'll set this up with BLADE:", WHITE, False),
        ("✓ iohblade init --template custom-problem", MINT, True),
        ("✓ wrote problem.py and experiment.yml", MINT, True),
        ("✓ iohblade validate: configuration is valid", MINT, True),
        ("✓ iohblade check: ready to run", MINT, True),
        ("Start the run with: iohblade run ...", WHITE, False),
    ], user=False)
    F.text(PAD, yy + 70, "Works with Claude Code and other agents", 24, WHITE,
           600)
    F.text(PAD, yy + 104, "that read skills.", 24, WHITE, 600)
    footer(7)
    return finish()


CARDS = [
    ("01-release", card_release),
    ("02-yaml", card_yaml),
    ("03-suites", card_suites),
    ("04-methods", card_methods),
    ("05-local-llms", card_local),
    ("06-analysis", card_analysis),
    ("07-coding-agents", card_agent),
]


def main() -> None:
    for name, fn in CARDS:
        path = HERE / f"card-{name}.svg"
        path.write_text(fn())
        print("wrote", path.name)


if __name__ == "__main__":
    main()
