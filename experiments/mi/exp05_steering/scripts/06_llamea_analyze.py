"""Step 6: search outcome and code diversity of the LLaMEA arms.

Per generated algorithm (``data/raw/<tag>/*/run-*/log.jsonl``, in the order
LLaMEA generated them): operator, lever, failure, fitness, experiment 3's
concept labels and, for each diversity metric (``common.diversity``:
``codebleu``, ``ast``, ``concepts``),

* ``novelty``: distance to the nearest algorithm generated earlier in the
  same run;
* ``step``: distance to its parent (how far the operator moved).

Per run: best fitness (AOCC), failure rate, mean pairwise distance of all
generated algorithms and of the working ones, entropy and number of
distinct concept sets (algorithm families), and the diversity of the final
population (the ``n_parents`` best). Arms are compared by the mean over
seeds with 95% bootstrap CIs; operators by pooling their children over
runs (CIs over runs).

Writes ``results/<tag>/solutions.csv``, ``runs.csv``, ``arms.csv``,
``operators.csv``, ``summary.md`` and ``diversity.png``.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import importlib

from common import diversity as dv
from common.paths import data_dir, results_dir
from common.runs import fitness_scalar, read_jsonl
from exp_config import EXPERIMENT, LLAMEA_CAMPAIGNS

prep = importlib.import_module("01_prepare")
labels_mod = prep.exp03_module("03_concept_labels")

N_PARENTS = 4
BEHAVIOUR = ["exploration_pct", "intensification"]  # traced campaigns only
PACKING = ["feasible", "strict_score", "violation"]  # packing campaigns only
NEAR_FEASIBLE = 1e-6  # max violation counted as near-feasible


def load_runs(tag: str) -> list[pd.DataFrame]:
    runs = []
    for log in sorted(data_dir(EXPERIMENT, "raw", tag).glob("*/run-*/log.jsonl")):
        rows = read_jsonl(log)
        if not rows:
            continue
        name = log.parent.name  # run-<method>-<model>-<problem>-<seed>-<i>
        arm = name.split("-")[2] if name.startswith("run-LLaMEA-") else "?"
        seed = int(re.search(r"-(\d+)(?:-\d+)?$", name).group(1))
        df = pd.DataFrame(
            {
                "arm": arm,
                "seed": seed,
                "run": f"{arm}-{seed}",
                "idx": range(len(rows)),
                "id": [r["id"] for r in rows],
                "parent": [(r.get("parent_ids") or [None])[0] for r in rows],
                "generation": [r.get("generation") for r in rows],
                "operator": [(r.get("metadata") or {}).get("operator") for r in rows],
                "steer": [(r.get("metadata") or {}).get("steer") for r in rows],
                "steer_dose": [
                    (r.get("metadata") or {}).get("steer_dose") for r in rows
                ],
                **{
                    f"b_{m}": [
                        ((r.get("metadata") or {}).get("behaviour") or {}).get(m)
                        for r in rows
                    ]
                    for m in BEHAVIOUR
                },
                **{
                    m: [(r.get("metadata") or {}).get(m) for r in rows] for m in PACKING
                },
                "code": [r.get("code") or "" for r in rows],
                "fitness": [fitness_scalar(r.get("fitness")) for r in rows],
                "error": [r.get("error") or "" for r in rows],
            }
        )
        df["failed"] = ~np.isfinite(df.fitness) | (df.error != "")
        for m in BEHAVIOUR:
            df[f"b_{m}"] = pd.to_numeric(df[f"b_{m}"], errors="coerce")
        df["aocc"] = np.where(df.failed, 0.0, df.fitness.fillna(0.0))
        # Packing: strict feasibility of the raw packing (before repair).
        df["feasible"] = df.feasible.map(lambda v: v is True)
        df["strict_score"] = pd.to_numeric(df.strict_score, errors="coerce")
        df["violation"] = pd.to_numeric(df.violation, errors="coerce")
        runs.append(df)
    return runs


def run_measures(df: pd.DataFrame, pool) -> tuple[pd.DataFrame, dict]:
    """Per-solution novelty and step, and per-run summary, for one run."""
    codes = df.code.tolist()
    sets = []
    for c in codes:
        lab = labels_mod.labels(c) if c else {}
        sets.append(frozenset(k for k, v in lab.items() if v))
    df = df.copy()
    df["concepts"] = ["|".join(sorted(s)) for s in sets]
    pos = {i: k for k, i in enumerate(df.id)}
    par = [pos.get(p) for p in df.parent]
    mats = {
        "codebleu": dv.similarity_matrix(codes, "codebleu", pool),
        "ast": dv.similarity_matrix(codes, "ast", pool),
        "concepts": dv.concept_similarity_matrix(sets),
    }
    ok = np.where(~df.failed.to_numpy())[0]
    best = np.argsort(-df.aocc.to_numpy(), kind="stable")[:N_PARENTS]
    summary = {
        "arm": df.arm.iloc[0],
        "seed": df.seed.iloc[0],
        "n": len(df),
        "best_aocc": float(df.aocc.max()),
        "mean_aocc": float(df.aocc.mean()),
        "failed": float(df.failed.mean()),
    }
    for m, S in mats.items():
        df[f"novelty_{m}"] = dv.novelty(S)
        df[f"step_{m}"] = [
            1 - S[i, p] if p is not None else np.nan for i, p in enumerate(par)
        ]
        summary[f"div_{m}"] = dv.mean_pairwise_distance(S)
        summary[f"div_ok_{m}"] = dv.mean_pairwise_distance(S, ok)
        summary[f"div_final_{m}"] = dv.mean_pairwise_distance(S, best)
    for m in BEHAVIOUR:
        b = df[f"b_{m}"].astype(float)
        summary[m] = float(b.mean()) if b.notna().any() else np.nan
        summary[f"{m}_final"] = (
            float(b.iloc[best].mean()) if b.iloc[best].notna().any() else np.nan
        )
    if df.violation.notna().any():  # packing: feasibility of working ones
        w = df.iloc[ok]
        summary["feasible"] = float(w.feasible.mean()) if len(w) else np.nan
        summary["near_feasible"] = (
            float((w.violation <= NEAR_FEASIBLE).mean()) if len(w) else np.nan
        )
        st = df.strict_score.where(df.feasible)
        summary["best_strict"] = float(st.max()) if st.notna().any() else 0.0
    ent, n_fam = dv.family_entropy(sets)
    summary.update(family_entropy=ent, n_families=n_fam)
    return df.drop(columns="code"), summary


def boot_ci(x, n_boot, rng):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    b = [rng.choice(x, len(x)).mean() for _ in range(n_boot)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def by_arm(runs: pd.DataFrame, n_boot, rng) -> pd.DataFrame:
    cols = [c for c in runs.columns if c not in ("arm", "seed", "n")]
    rows = []
    for arm, g in runs.groupby("arm"):
        row = {"arm": arm, "runs": len(g)}
        for c in cols:
            lo, hi = boot_ci(g[c], n_boot, rng)
            row.update({c: g[c].mean(), f"{c}_lo": lo, f"{c}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def by_operator(sol: pd.DataFrame, n_boot, rng) -> pd.DataFrame:
    """Children per (arm, operator), means with CIs over runs."""
    fit = sol.set_index("id").aocc  # parents include the initial population
    sol = sol[sol.operator != "init"].copy()
    sol["parent_aocc"] = sol.parent.map(fit)
    sol["improved"] = (sol.aocc > sol.parent_aocc).astype(float)
    cols = [
        "failed",
        "aocc",
        "improved",
        "novelty_codebleu",
        "novelty_ast",
        "novelty_concepts",
        "step_codebleu",
        "step_ast",
        "step_concepts",
    ]
    rows = []
    for (arm, op), g in sol.groupby(["arm", "operator"]):
        per_run = g.groupby("run")[cols].mean()
        row = {"arm": arm, "operator": op, "n": len(g)}
        for c in cols:
            lo, hi = boot_ci(per_run[c], n_boot, rng)
            row.update({c: g[c].mean(), f"{c}_lo": lo, f"{c}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def plot(sol: pd.DataFrame, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import INK, SERIES, setup

    setup()
    arms = sorted(sol.arm.unique())
    traced = sol["b_exploration_pct"].notna().any()
    n = 4 if traced else 3
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4))
    for k, arm in enumerate(arms):
        g = sol[sol.arm == arm]
        c = SERIES[k % len(SERIES)]
        best = g.groupby("run").aocc.cummax()
        curve = best.groupby(g.idx).mean()
        axes[0].plot(curve.index + 1, curve.values, color=c, label=arm)
        panels = [(axes[1], "novelty_codebleu"), (axes[2], "novelty_concepts")]
        if traced:
            panels.append((axes[3], "b_exploration_pct"))
        for ax, m in panels:
            w = g.groupby("idx")[m].mean().rolling(12, min_periods=6).mean()
            ax.plot(w.index + 1, w.values, color=c, label=arm)
    titles = (
        "best fitness so far",
        "novelty (1 - CodeBLEU to nearest earlier)",
        "novelty (concept-set Jaccard)",
        "exploration % of working algorithms",
    )
    for ax, t in zip(axes, titles):
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("algorithms generated")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=len(arms),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def with_ci(df, cols):
    """Formats ``c`` as 'mean (lo to hi)'."""
    out = df[[c for c in df.columns if c in ("arm", "operator", "runs", "n")]].copy()
    for c in cols:
        out[c] = [
            f"{m:.3f} ({lo:.3f} to {hi:.3f})"
            for m, lo, hi in zip(df[c], df[f"{c}_lo"], df[f"{c}_hi"])
        ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="llamea", choices=list(LLAMEA_CAMPAIGNS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = results_dir(EXPERIMENT, args.tag)
    out.mkdir(parents=True, exist_ok=True)

    sols, summaries = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for df in load_runs(args.tag):
            s, r = run_measures(df, pool)
            sols.append(s)
            summaries.append(r)
            print(f"{r['arm']}-{r['seed']}: n={r['n']} best={r['best_aocc']:.3f}")
    sol = pd.concat(sols, ignore_index=True)
    runs = pd.DataFrame(summaries).sort_values(["arm", "seed"])
    sol.to_csv(out / "solutions.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    arms = by_arm(runs, args.n_boot, rng)
    arms.to_csv(out / "arms.csv", index=False)
    ops = by_operator(sol, args.n_boot, rng)
    ops.to_csv(out / "operators.csv", index=False)

    arm_cols = [
        "best_aocc",
        "failed",
        "div_codebleu",
        "div_ast",
        "div_concepts",
        "div_final_codebleu",
        "family_entropy",
        "n_families",
    ]
    arm_cols += [c for c in ("feasible", "near_feasible", "best_strict") if c in runs]
    arm_cols += [
        c
        for m in BEHAVIOUR
        for c in (m, f"{m}_final")
        if c in runs and runs[c].notna().any()
    ]
    op_cols = [
        "failed",
        "aocc",
        "improved",
        "novelty_codebleu",
        "novelty_concepts",
        "step_codebleu",
    ]
    lines = [
        f"# LLaMEA campaign `{args.tag}`",
        "",
        f"{runs.arm.nunique()} arms, {len(runs)} runs, {len(sol)} algorithms."
        " Means over runs with 95% bootstrap CIs.",
        "",
        "## Arms",
        "",
        to_markdown(with_ci(arms, arm_cols)),
        "",
        "## Operators (children pooled over runs, CIs over runs)",
        "",
        to_markdown(with_ci(ops, op_cols)),
        "",
    ]
    steered = sol[sol.steer.notna()]
    if len(steered):
        lev = steered.groupby("steer")[
            ["failed", "aocc", "novelty_codebleu", "novelty_concepts"]
        ].mean()
        lev.insert(0, "n", steered.groupby("steer").size())
        lines += ["## Per lever", "", to_markdown(lev.reset_index()), ""]
    (out / "summary.md").write_text("\n".join(lines))
    plot(sol, out / "diversity.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
