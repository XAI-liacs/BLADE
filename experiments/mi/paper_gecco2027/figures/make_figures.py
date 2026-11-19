"""Figures for the GECCO 2027 draft, from the committed exp05 results.

* ``dial.pdf``: slope per unit dose of the exploration dial in the
  single-generation screens (measured exploration %, and the share of
  population-based code), per problem, 95% bootstrap CIs.
* ``budget200.pdf``: best-so-far AOCC of the steered LLaMEA arms on
  MA-BBOB, mean over 16 seeds, 200 evaluations.

Usage (from paper_gecco2027): python figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "exp05_steering/results"

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)
INK = "#333333"
MUTED = "#8a8a8a"
# Okabe-Ito (colour-blind safe), fixed order per arm
ARM_STYLE = {
    "control": ("#000000", "-", "control"),
    "adaptive": ("#0072B2", "-", "adaptive dial"),
    "explore": ("#009E73", "-", "dial explore"),
    "exploit": ("#D55E00", "-", "dial exploit"),
    "prompt_explore": ("#009E73", ":", "prompt explore"),
    "prompt_exploit": ("#D55E00", ":", "prompt exploit"),
}


def screen_slopes():
    rows = []
    d = pd.read_csv(RES / "dial/slopes.csv")
    for direction, label in [
        ("b_exploration_pct_resid@L27", "MA-BBOB"),
        ("random27", "MA-BBOB, random"),
    ]:
        x = d[d.direction == direction].iloc[0]
        rows.append(
            {
                "problem": label,
                "metric": "exploration_pct",
                "slope": x.slope_exploration_pct,
                "lo": x.slope_exploration_pct_lo,
                "hi": x.slope_exploration_pct_hi,
            }
        )
    for tag, label in [
        ("transfer-bragg", "Bragg"),
        ("transfer-packing", "Packing"),
        ("transfer-bbob-f20", "BBOB f20"),
        ("transfer-bbob-f2", "BBOB f2"),
        ("transfer-bbob-f9", "BBOB f9"),
    ]:
        s = pd.read_csv(RES / tag / "slopes.csv").query("phase == 'pooled'")
        for m in ("exploration_pct", "population"):
            x = s[s.metric == m]
            if len(x):
                x = x.iloc[0]
                rows.append(
                    {
                        "problem": label,
                        "metric": m,
                        "slope": x.slope,
                        "lo": x.lo,
                        "hi": x.hi,
                    }
                )
    return pd.DataFrame(rows)


def fig_dial():
    s = screen_slopes()
    order = [
        "MA-BBOB",
        "MA-BBOB, random",
        "Bragg",
        "Packing",
        "BBOB f20",
        "BBOB f2",
        "BBOB f9",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.9), sharey=True)
    panels = [
        ("exploration_pct", "exploration (%)"),
        ("population", "population code (share)"),
    ]
    for ax, (m, title) in zip(axes, panels):
        for i, p in enumerate(order):
            x = s[(s.problem == p) & (s.metric == m)]
            if not len(x):
                ax.text(
                    0.66,
                    i,
                    "not measured",
                    va="center",
                    ha="center",
                    color=MUTED,
                    fontsize=5.5,
                    transform=ax.get_yaxis_transform(),
                )
                continue
            x = x.iloc[0]
            sig = x.lo > 0 or x.hi < 0
            c = MUTED if "random" in p else "#0072B2"
            ax.plot([x.lo, x.hi], [i, i], color=c, lw=1.2, solid_capstyle="round")
            ax.plot(x.slope, i, "o", ms=3.5, mfc=c if sig else "white", mec=c, mew=1.0)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("slope per unit dose")
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(HERE / "dial.pdf")


def fig_budget200():
    rows = json.loads((RES / "dashboard/curves.json").read_text())["rows"]
    c = pd.DataFrame(rows).query("problem == 'MA-BBOB (budget 200)'")
    fig, ax = plt.subplots(figsize=(3.33, 2.25))
    for arm, (col, ls, label) in ARM_STYLE.items():
        g = c[c.arm == arm].sort_values("evaluation")
        ax.plot(g.evaluation, g["mean"], color=col, ls=ls, lw=1.3, label=label)
        if arm in ("control", "exploit"):
            ax.fill_between(g.evaluation, g.q25, g.q75, color=col, alpha=0.10, lw=0)
    ax.axvline(100, color=MUTED, lw=0.6, ls="--")
    ax.text(102, 0.13, "budget 100", color=MUTED, fontsize=5.5)
    ax.set_xlim(1, 200)
    ax.set_ylim(0.1, 0.8)
    ax.set_xlabel("evaluated algorithms")
    ax.set_ylabel("best-so-far AOCC")
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=3,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        handlelength=2.0,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "budget200.pdf")


if __name__ == "__main__":
    fig_dial()
    fig_budget200()
