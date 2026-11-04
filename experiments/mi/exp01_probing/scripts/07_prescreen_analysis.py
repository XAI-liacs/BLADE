"""Step 7: analyse the held-out pre-screen campaign.

Run after ``02_build_dataset.py --tag prescreen``. Two questions:

1. **Does the frozen probe generalise to new seeds?** Scored on the true
   outcome of every algorithm (both conditions; LLaMEA's view does not
   change the label). Reports AUROC and, at the frozen threshold, the
   fraction flagged, failures caught, working algorithms lost and
   top-decile algorithms lost (top decile of true fitness within the run).
2. **What does acting on it do to the search?** Paired by seed: best fitness
   that LLaMEA could select (``shadow`` sees every result, ``screened`` loses
   the flagged ones) and the true failure rate of what it generates.

Writes ``results/<tag>/``: ``screen_metrics.csv``, ``runs.csv`` and
``best_so_far.csv``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT


def screen_metrics(df: pd.DataFrame) -> dict:
    y, flag = df["child_failed"].to_numpy(), df["prescreen_flagged"].to_numpy()
    fit = df["child_fitness"]
    top = fit >= df.groupby("run")["child_fitness"].transform(lambda f: f.quantile(0.9))
    working = y == 0
    return {
        "n": len(df),
        "failure_rate": y.mean(),
        "auroc": roc_auc_score(y, df["prescreen_score"]) if 0 < y.sum() < len(y) else np.nan,
        "flagged": flag.mean(),
        "failures_caught": flag[y == 1].mean(),
        "working_lost": flag[working].mean(),
        "top_decile_lost": flag[top.to_numpy() & working].mean(),
        "precision": y[flag == 1].mean() if flag.any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="prescreen")
    args = ap.parse_args()

    proc = data_dir(EXPERIMENT, "processed", args.tag)
    meta = pd.read_csv(proc / "meta.csv", dtype={"record_id": str, "run": str})
    meta = meta.sort_values(["run", "query_idx"])
    cfg = json.loads((data_dir(EXPERIMENT, "raw", args.tag) / "config.json").read_text())
    out = results_dir(EXPERIMENT, args.tag)
    print(f"probe: {cfg['probe']}")

    rows = []
    for cond, df in [("all", meta), *meta.groupby("condition")]:
        rows.append({"condition": cond, **screen_metrics(df)})
    for op, df in meta.groupby("operator"):
        rows.append({"condition": f"all / {op}", **screen_metrics(df)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "screen_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    # What LLaMEA could select: the true fitness unless it was rejected.
    visible = meta["child_fitness"].where(meta["visible_failed"] == 0)
    meta = meta.assign(visible_fitness=visible)
    meta["eval_idx"] = meta.groupby("run").cumcount() + 1
    meta["best_so_far"] = meta.groupby("run")["visible_fitness"].cummax()
    meta[["run", "seed", "condition", "eval_idx", "best_so_far"]].to_csv(
        out / "best_so_far.csv", index=False
    )

    runs = (
        meta.groupby(["seed", "condition"])
        .agg(
            n=("record_id", "size"),
            best=("visible_fitness", "max"),
            true_best=("child_fitness", "max"),
            failure_rate=("child_failed", "mean"),
            flagged=("prescreen_flagged", "mean"),
        )
        .reset_index()
    )
    runs.to_csv(out / "runs.csv", index=False)
    wide = runs.pivot(index="seed", columns="condition")
    print("\nper seed:\n" + wide[["n", "best", "failure_rate"]].round(3).to_string())
    paired = wide.dropna(subset=[("best", "shadow"), ("best", "screened")])
    if len(paired) >= 5:
        for col in ("best", "failure_rate"):
            a, b = paired[(col, "shadow")], paired[(col, "screened")]
            p = wilcoxon(a, b).pvalue if (a != b).any() else 1.0
            print(
                f"{col}: shadow median {a.median():.3f}, screened median "
                f"{b.median():.3f}, paired Wilcoxon p={p:.3f} (n={len(paired)})"
            )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
