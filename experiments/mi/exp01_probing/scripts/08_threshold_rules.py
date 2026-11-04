"""Step 8: compare pre-screen decision rules on held-out runs.

The fixed threshold of ``04_export_probes.py`` flags more than its target
rate early in a run and under the ``new`` operator, because failures really
are more common there. This script fits four rules on the out-of-fold
scores of a past campaign (``--fit-tag``, default ``main``), all targeting
the same flag rate:

* ``fixed``: one global threshold (what ``prescreen`` used);
* ``per_operator``: one threshold per operator, ``rate`` of each operator;
* ``risk``: calibrated P(fail | score, operator), one probability threshold;
* ``running``: quantile of the last 24 scores of the same run.

It then replays every run of the pre-screen campaign (``--eval-tag``) in
order, applies each rule online and scores it against the true outcomes.
The ``shadow`` runs are the clean test set: their trajectory never depended
on a flag.

Writes ``results/<fit-tag>/probes/rules.json`` (fitted rules, used by
``05_prescreen_generate.py``) and ``results/<eval-tag>/rules_*.csv``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import data_dir, results_dir
from common.prescreen import RULES, RunningRule
from exp_config import EXPERIMENT

PHASES = [-1, 11, 30, 60, 1000]
PHASE_LABELS = ["0-11 (init)", "12-30", "31-60", "61-100"]


def fit_rules(fit_tag: str, rate: float) -> dict:
    oof = pd.read_csv(
        results_dir(EXPERIMENT, fit_tag) / "oof_child_failed_gen_mean.csv"
    )
    s, ops, y = oof["pred"].to_numpy(), oof["operator"].to_numpy(), oof["child_failed"]
    rules = {}
    for name, cls in RULES.items():
        rule = RunningRule(window=24) if cls is RunningRule else cls()
        if name == "risk":
            rule.fit(s, ops, rate, labels=y.to_numpy())
        else:
            rule.fit(s, ops, rate)
        rules[name] = rule
    return rules


def replay(meta: pd.DataFrame, rules: dict) -> pd.DataFrame:
    meta = meta.sort_values(["run", "query_idx"]).copy()
    for name, rule in rules.items():
        flags = []
        for _, run in meta.groupby("run", sort=False):
            history = []
            for score, op in zip(run["prescreen_score"], run["operator"]):
                flags.append(bool(rule.flag(score, op, history)))
                history.append(score)
        meta[f"flag_{name}"] = flags
    return meta


def metrics(df: pd.DataFrame, flag: str) -> dict:
    y, f = df["child_failed"].to_numpy(), df[flag].to_numpy()
    working = y == 0
    q90 = df.groupby("run")["child_fitness"].transform(lambda v: v.quantile(0.9))
    top = (df["child_fitness"] >= q90).to_numpy() & working
    return {
        "n": len(df),
        "failure_rate": y.mean(),
        "flagged": f.mean(),
        "failures_caught": f[y == 1].mean() if (y == 1).any() else np.nan,
        "working_lost": f[working].mean(),
        "top_decile_lost": f[top].mean() if top.any() else np.nan,
        "precision": y[f].mean() if f.any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-tag", default="main")
    ap.add_argument("--eval-tag", default="prescreen")
    ap.add_argument("--rate", type=float, default=0.2)
    args = ap.parse_args()

    rules = fit_rules(args.fit_tag, args.rate)
    probe_dir = results_dir(EXPERIMENT, args.fit_tag) / "probes"
    (probe_dir / "rules.json").write_text(
        json.dumps({k: r.to_dict() for k, r in rules.items()}, indent=2) + "\n"
    )

    meta = pd.read_csv(
        data_dir(EXPERIMENT, "processed", args.eval_tag) / "meta.csv",
        dtype={"record_id": str, "run": str},
    )
    meta = replay(meta, rules)
    meta["phase"] = pd.cut(meta["query_idx"], PHASES, labels=PHASE_LABELS)
    out = results_dir(EXPERIMENT, args.eval_tag)

    rows, by_group = [], []
    for subset, df in [("shadow", meta[meta["condition"] == "shadow"]), ("all", meta)]:
        for name in rules:
            rows.append({"subset": subset, "rule": name, **metrics(df, f"flag_{name}")})
            for col in ("phase", "operator"):
                for g, d in df.groupby(col, observed=True):
                    by_group.append(
                        {
                            "subset": subset,
                            "rule": name,
                            "by": col,
                            "group": g,
                            **metrics(d, f"flag_{name}"),
                        }
                    )
    summary = pd.DataFrame(rows)
    groups = pd.DataFrame(by_group)
    # How far does the flag rate drift from the target across phases?
    drift = (
        groups[groups["by"] == "phase"]
        .assign(dev=lambda g: (g["flagged"] - args.rate).abs())
        .groupby(["subset", "rule"])["dev"]
        .max()
        .rename("max_phase_deviation")
    )
    summary = summary.merge(drift, on=["subset", "rule"])
    summary.to_csv(out / "rules_summary.csv", index=False)
    groups.to_csv(out / "rules_by_group.csv", index=False)
    meta[
        [
            "run",
            "query_idx",
            "condition",
            "operator",
            "phase",
            "child_failed",
            "child_fitness",
            "prescreen_score",
        ]
        + [f"flag_{n}" for n in rules]
    ].to_csv(out / "rules_flags.csv", index=False)

    shadow = meta[meta["condition"] == "shadow"]
    print(
        f"probe AUROC on shadow runs: "
        f"{roc_auc_score(shadow['child_failed'], shadow['prescreen_score']):.3f}"
    )
    print(summary[summary["subset"] == "shadow"].round(3).to_string(index=False))
    print("\nflag rate by phase (shadow):")
    print(
        groups[(groups["subset"] == "shadow") & (groups["by"] == "phase")]
        .pivot(index="group", columns="rule", values="flagged")
        .round(3)
        .to_string()
    )
    print(f"\nwrote {probe_dir / 'rules.json'} and {out}/rules_*.csv")


if __name__ == "__main__":
    main()
