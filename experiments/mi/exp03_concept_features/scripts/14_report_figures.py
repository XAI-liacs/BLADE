"""Step 14: figures for the experiment 3 report.

* ``keyword_masking.png``: per concept, the best feature's validation
  AUROC on the raw code and with the keyword lines masked, against what
  the other concepts' labels plus code length predict (``other_labels``).
* ``failure_types.png``: per error type, out-of-fold AUROC of the code
  text alone and of text + SAE code (step 11).

Reads ``results/<tag>/keyword_masking.csv`` and ``failure_types.csv``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features

from common.paths import results_dir
from common.plotting import INK, NEUTRAL, SERIES, setup
from exp_config import EXPERIMENT


def masking_figure(df: pd.DataFrame, path: Path):
    best = df.loc[df.groupby("concept").auroc_line.idxmax()].copy()
    raw = df.groupby("concept").auroc_none.max()
    best["raw"] = best.concept.map(raw)
    best = best.sort_values("auroc_line")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = range(len(best))
    for i, r in enumerate(best.itertuples()):
        ax.plot([r.auroc_other_labels, r.auroc_line], [i, i], color="#d6d5d0", lw=2)
    ax.scatter(
        best.auroc_other_labels,
        y,
        color=NEUTRAL,
        s=42,
        zorder=3,
        label="other labels + length (no SAE)",
    )
    ax.scatter(
        best.raw,
        y,
        color=SERIES[0],
        s=42,
        zorder=3,
        marker="o",
        facecolors="none",
        linewidths=2,
        label="best feature, raw code",
    )
    ax.scatter(
        best.auroc_line,
        y,
        color=SERIES[1],
        s=42,
        zorder=4,
        label="best feature, keyword lines masked",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [
            f"{c.replace('_', ' ')}  (L{l} f{f})"
            for c, l, f in zip(best.concept, best.layer, best.feature)
        ]
    )
    ax.set_xlabel("validation AUROC for the concept label")
    ax.set_xlim(0.5, 1.01)
    ax.grid(axis="y", visible=False)
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    ax.set_title(
        "Concept features with the keyword lines masked",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def failure_type_figure(df: pd.DataFrame, path: Path):
    df = df.rename(columns={"auroc_T+S": "auroc_TS"}).sort_values("delta")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = range(len(df))
    for i, r in enumerate(df.itertuples()):
        ax.plot([r.auroc_T, r.auroc_TS], [i, i], color="#d6d5d0", lw=2)
    ax.scatter(df.auroc_T, y, color=NEUTRAL, s=42, zorder=3, label="code text")
    ax.scatter(
        df.auroc_TS, y, color=SERIES[0], s=42, zorder=4, label="text + SAE code (L21)"
    )
    for i, r in enumerate(df.itertuples()):
        ax.annotate(
            f"+{r.delta:.2f}",
            (r.auroc_TS, i),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{t} (n = {n})" for t, n in zip(df.type, df.n_type)])
    ax.set_xlabel("out-of-fold AUROC, error type vs working code")
    ax.set_xlim(0.55, 1.07)
    ax.grid(axis="y", visible=False)
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    ax.set_title(
        "What the SAE code adds to the text, by error type",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    args = ap.parse_args()
    setup()
    out = results_dir(EXPERIMENT, args.tag)
    masking_figure(
        pd.read_csv(out / "keyword_masking.csv"), out / "keyword_masking.png"
    )
    failure_type_figure(
        pd.read_csv(out / "failure_types.csv"), out / "failure_types.png"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
