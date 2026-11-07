"""Compare experiments 1 and 2 across Qwen3-8B, Qwen3-14B and Gemma-3-12B-it.

Reads the result files written by the experiment scripts (nothing is
recomputed except the paired pre-screen test) and writes to
``cross_model/results/``:

* ``layer_curves.png``: probe score per layer, on relative depth
  (layer / number of layers), for the three main targets;
* ``summary.csv`` / ``summary.md``: one row per finding and model.

Models whose results are missing (for example counterfactuals still
running) are left out of the affected rows.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parent
sys.path.insert(0, str(MI_ROOT))

from common.plotting import INK, NEUTRAL, SERIES, setup

EXP1 = MI_ROOT / "exp01_probing" / "results"
EXP2 = MI_ROOT / "exp02_feedback_attribution" / "results"
OUT = HERE / "results"

# name -> (exp1/exp2 tag suffix, number of layers, counterfactual tag)
MODELS = {
    "Qwen3-8B": ("", 36, "cf"),
    "Qwen3-14B": ("-qwen3-14b", 40, "cf-qwen3-14b"),
    "Gemma-3-12B": ("-gemma-3-12b", 48, "cf-gemma-3-12b"),
}
COLORS = dict(zip(MODELS, SERIES))

# target -> (position, metric, label)
CURVES = {
    "child_failed": ("gen_mean", "auroc", "child fails (AUROC)"),
    "over_budget": ("gen_mean", "auroc", "child over budget (AUROC)"),
    "parent_fitness": ("prompt_mean", "spearman", "parent fitness (Spearman)"),
}

# (stratum, variant, metric) rows of the counterfactual summary
CF_ROWS = [
    ("error", "no_error", "failed"),
    ("error", "no_error", "same_error"),
    ("error", "no_error", "error_line_kept"),
    ("error", "error_generic", "failed"),
    ("scored", "no_feedback", "failed"),
    ("scored", "score_low", "pf"),
    ("scored", "score_high", "pf"),
    ("scored", "score_low", "failed"),
    ("scored", "score_low", "similarity"),
    ("scored", "steer_up", "pf"),
    ("scored", "steer_random", "pf"),
    ("scored", "steer_up", "similarity"),
    ("scored", "steer_down", "similarity"),
    ("scored", "steer_random", "similarity"),
    ("scored", "steer_up", "failed"),
    ("scored", "steer_down", "failed"),
    ("scored", "steer_random", "failed"),
    ("scored", "budget_msg", "over_budget"),
]
# (stratum, metric) of the unmodified prompt, from cf_means.csv
CF_BASE = [
    ("error", "failed"),
    ("error", "same_error"),
    ("scored", "failed"),
    ("scored", "over_budget"),
    ("scored", "similarity"),
]


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def fmt_ci(v, lo, hi, digits: int = 3) -> str:
    return f"{v:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def layer_curves():
    setup()
    fig, axes = plt.subplots(1, len(CURVES), figsize=(12, 3.4), sharex=True)
    for ax, (target, (pos, metric, label)) in zip(axes, CURVES.items()):
        for name, (suffix, n_layers, _) in MODELS.items():
            s = read(EXP1 / f"main{suffix}" / "probe_scores.csv")
            if s is None:
                continue
            s = s[(s.target == target) & (s.position == pos)].sort_values("layer")
            x = s.layer / (n_layers - 1)
            ax.plot(x, s[metric], color=COLORS[name], label=name)
            ax.plot(
                x,
                s[f"control_{metric}"],
                color=COLORS[name],
                lw=1,
                ls=":",
            )
        if metric == "auroc":
            ax.axhline(0.5, color=NEUTRAL, lw=1, ls="--")
        ax.set_title(f"{label}\n{pos.replace('_', ' ')}", color=INK)
        ax.set_xlabel("relative depth (layer / last layer)")
    axes[0].set_ylabel("cross-validated score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS))
    fig.text(
        0.99,
        0.01,
        "dotted: shuffled-label control",
        ha="right",
        color=NEUTRAL,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "layer_curves.png", dpi=150)
    plt.close(fig)


def probe_rows():
    rows = []
    for name, (suffix, n_layers, _) in MODELS.items():
        s = read(EXP1 / f"main{suffix}" / "probe_scores.csv")
        if s is None:
            continue
        base = read(EXP1 / f"main{suffix}" / "baselines.csv")
        for target, (pos, metric, label) in CURVES.items():
            t = s[s.target == target]
            best = t.loc[t[metric].idxmax()]
            rows.append(
                {
                    "section": "exp1 best probe",
                    "finding": label,
                    "model": name,
                    "value": f"{best[metric]:.3f} (L{int(best.layer)}, "
                    f"depth {best.layer / (n_layers - 1):.2f}, {best.position})",
                }
            )
            b = base[base.target == target]
            bb = b.loc[b[metric].idxmax()]
            rows.append(
                {
                    "section": "exp1 best probe",
                    "finding": f"{label}: best baseline",
                    "model": name,
                    "value": f"{bb[metric]:.3f} ({bb.baseline})",
                }
            )
            if metric == "auroc":
                rows.append(
                    {
                        "section": "exp1 best probe",
                        "finding": f"{label}: n, base rate",
                        "model": name,
                        "value": f"{int(best.n)}, {best.base_rate:.3f}",
                    }
                )
        inc = read(EXP1 / f"main{suffix}" / "followup" / "incremental.csv")
        if inc is not None:
            for target in ("child_failed", "over_budget"):
                r = inc[
                    (inc.target == target)
                    & (inc.position == "gen_mean")
                    & (inc.model == "M+T+P - M+T")
                ]
                if len(r):
                    r = r.iloc[0]
                    rows.append(
                        {
                            "section": "exp1 incremental value",
                            "finding": f"{target}: M+T+P - M+T (gen mean)",
                            "model": name,
                            "value": fmt_ci(r.score, r.ci_low, r.ci_high),
                        }
                    )
    return rows


def prescreen_rows():
    rows = []
    for name, (suffix, _, _) in MODELS.items():
        d = EXP1 / ("prescreen" + suffix)
        m = read(d / "screen_metrics.csv")
        runs = read(d / "runs.csv")
        if m is None or runs is None:
            continue
        a = m[m.condition == "all"].iloc[0]
        for col in ("failure_rate", "auroc", "failures_caught", "working_lost"):
            rows.append(
                {
                    "section": "exp1 held-out pre-screen",
                    "finding": col,
                    "model": name,
                    "value": f"{a[col]:.3f}",
                }
            )
        rows.append(
            {
                "section": "exp1 held-out pre-screen",
                "finding": "precision",
                "model": name,
                "value": f"{a['precision']:.2f}",
            }
        )
        p = runs.pivot(index="seed", columns="condition", values="best")
        _, pval = wilcoxon(p.screened, p.shadow)
        rows.append(
            {
                "section": "exp1 held-out pre-screen",
                "finding": "best fitness, screened vs shadow (median)",
                "model": name,
                "value": f"{p.screened.median():.3f} vs {p.shadow.median():.3f} "
                f"(Wilcoxon p={pval:.2f}, n={len(p)})",
            }
        )
    return rows


def span_rows():
    rows = []
    for name, (suffix, _, _) in MODELS.items():
        d = EXP2 / f"main{suffix}"
        drop = read(d / "span_drop.csv")
        grad = read(d / "span_gradients.csv")
        if drop is None:
            continue
        drop = drop[drop.operator == "all"].set_index("span")
        for span in ("parent_code", "format", "feedback"):
            if span in drop.index:
                r = drop.loc[span]
                rows.append(
                    {
                        "section": "exp2 span deletion (nats/token)",
                        "finding": span,
                        "model": name,
                        "value": fmt_ci(r["mean"], r.ci_low, r.ci_high),
                    }
                )
        if grad is not None:
            g = grad[grad.operator == "all"].set_index("span")
            if "parent_code" in g.index:
                rows.append(
                    {
                        "section": "exp2 span deletion (nats/token)",
                        "finding": "grad x input share, parent code",
                        "model": name,
                        "value": f"{g.loc['parent_code', 'grad_share']:.2f}",
                    }
                )
        sc = read(d / "score_counterfactual.csv")
        if sc is not None:
            for _, r in sc[sc.operator == "all"].iterrows():
                rows.append(
                    {
                        "section": "exp2 span deletion (nats/token)",
                        "finding": f"{r.variant} (teacher-forced drop)",
                        "model": name,
                        "value": fmt_ci(r["mean"], r.ci_low, r.ci_high, 4),
                    }
                )
    return rows


def cf_rows():
    rows = []
    for name, (_, _, tag) in MODELS.items():
        e = read(EXP2 / tag / "cf_effects.csv")
        if e is None:
            continue
        means = read(EXP2 / tag / "cf_means.csv")
        full = means[means.variant == "full"].set_index("stratum")
        for stratum, metric in CF_BASE:
            rows.append(
                {
                    "section": "exp2 counterfactual generation (variant - full)",
                    "finding": f"{stratum}: full prompt, {metric} (base)",
                    "model": name,
                    "value": f"{full.loc[stratum, metric]:.3f}",
                }
            )
        e = e.set_index(["stratum", "variant", "metric"])
        for key in CF_ROWS:
            if key in e.index:
                r = e.loc[key]
                rows.append(
                    {
                        "section": "exp2 counterfactual generation (variant - full)",
                        "finding": f"{key[0]}: {key[1]} -> {key[2]}",
                        "model": name,
                        "value": fmt_ci(r.effect, r.ci_low, r.ci_high),
                    }
                )
    return rows


def to_markdown(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if pd.isna(v) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    layer_curves()
    rows = probe_rows() + prescreen_rows() + span_rows() + cf_rows()
    long = pd.DataFrame(rows)
    wide = long.pivot_table(
        index=["section", "finding"],
        columns="model",
        values="value",
        aggfunc="first",
        sort=False,
    )[[m for m in MODELS if m in long.model.unique()]]
    wide.to_csv(OUT / "summary.csv")
    lines = []
    for section, block in wide.groupby(level=0, sort=False):
        lines += [f"### {section}", "", to_markdown(block.droplevel(0)), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
