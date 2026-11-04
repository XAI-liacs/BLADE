"""Step 2: summarise span attribution per operator and parent state.

Reads ``data/processed/<tag>/span_attribution.csv`` and
``span_gradients.csv`` from ``01_span_attribution.py`` and writes to
``results/<tag>/``:

* ``span_drop.csv``: per operator and span, the mean drop in response
  log-probability (nats/token) when the span is deleted, with a 95%
  bootstrap CI over LLaMEA runs; also for the code part of the response;
* ``score_counterfactual.csv``: the same for replacing the parent's score
  by 0.010 / 0.900;
* ``span_gradients.csv``: mean gradient x input share and share per token;
* ``span_drop_by_parent.csv``: drops split by whether the parent failed
  (its feedback is then an error message instead of a score).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT


def boot_ci(values: pd.Series, groups: pd.Series, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    runs = groups.unique()
    by_run = {r: values[groups == r].to_numpy() for r in runs}
    means = [
        np.concatenate([by_run[r] for r in rng.choice(runs, len(runs))]).mean()
        for _ in range(n_boot)
    ]
    return np.percentile(means, [2.5, 97.5])


def drops(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df["variant"] == "full"].set_index("record_id")
    other = df[df["variant"] != "full"].copy()
    for col in ("logprob", "logprob_code"):
        other[f"drop_{col.replace('logprob', 'all').replace('all_code', 'code')}"] = (
            other["record_id"].map(full[col]) - other[col]
        )
    return other


def summarise(d: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(by):
        lo, hi = boot_ci(g[col], g["run"])
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, key)),
                "n": len(g),
                "mean": g[col].mean(),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    proc = data_dir(EXPERIMENT, "processed", args.tag)
    out = results_dir(EXPERIMENT, args.tag)

    d = drops(pd.read_csv(proc / "span_attribution.csv", dtype={"record_id": str}))
    spans = d[d["variant"].str.startswith("-")].assign(
        span=lambda x: x["variant"].str[1:]
    )
    all_ops = spans.assign(operator="all")
    both = pd.concat([spans, all_ops])
    res = summarise(both, ["operator", "span"], "drop_all").merge(
        summarise(both, ["operator", "span"], "drop_code")[
            ["operator", "span", "mean", "ci_low", "ci_high"]
        ],
        on=["operator", "span"],
        suffixes=("", "_code"),
    )
    res.to_csv(out / "span_drop.csv", index=False)
    print("drop in response log-prob when a span is deleted (nats/token):")
    print(
        res.pivot(index="span", columns="operator", values="mean").round(3).to_string()
    )

    by_parent = summarise(spans, ["parent_failed", "span"], "drop_all")
    by_parent.to_csv(out / "span_drop_by_parent.csv", index=False)
    print("\nby parent state (all operators):")
    print(
        by_parent.pivot(index="span", columns="parent_failed", values="mean")
        .round(3)
        .to_string()
    )

    cf = d[d["variant"].isin(["score_low", "score_high"])]
    cf = pd.concat([cf, cf.assign(operator="all")])
    cfs = summarise(cf, ["operator", "variant"], "drop_all")
    cfs.to_csv(out / "score_counterfactual.csv", index=False)
    print("\nscore counterfactual (nats/token):")
    print(cfs.round(4).to_string(index=False))

    g = pd.read_csv(proc / "span_gradients.csv", dtype={"record_id": str})
    g["density"] = g["grad_share"] / g["n_tokens"].clip(lower=1)
    gs = (
        pd.concat([g, g.assign(operator="all")])
        .groupby(["operator", "span"])[["grad_share", "density", "n_tokens"]]
        .mean()
        .reset_index()
    )
    gs.to_csv(out / "span_gradients.csv", index=False)
    print("\ngradient x input share:")
    print(
        gs.pivot(index="span", columns="operator", values="grad_share")
        .round(3)
        .to_string()
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
