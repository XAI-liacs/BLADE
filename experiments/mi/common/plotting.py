"""Plot style shared by the MI notebooks (matplotlib).

Categorical colours come in a fixed order from a CVD-validated palette, so a
given position/series always has the same colour across experiments. Neutral
gray is reserved for references (controls, baselines, chance).
"""

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NEUTRAL = "#8a8a85"
INK = "#2b2b29"
POSITION_COLORS = {
    "last": SERIES[0],
    "prompt_mean": SERIES[1],
    "gen_mean": SERIES[2],
}
POSITION_LABELS = {
    "last": "last prompt token",
    "prompt_mean": "prompt mean",
    "gen_mean": "generated-code mean",
}


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
