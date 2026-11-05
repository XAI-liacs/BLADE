"""Step 5: effect of each prompt counterfactual on what the model writes.

Reads ``data/raw/<tag>/shard-*.jsonl`` (``04_generate_counterfactuals.py``)
and ``data/processed/<tag>/cf_queries.jsonl``. Per sample:

* ``failed``: the child crashed, timed out or had no code;
* ``score``: fitness, with failures counted as 0;
* ``over_budget``: a working child overspent on at least one instance;
* ``similarity``: ``difflib`` ratio between the child's and the parent's
  code; ``same_name``: the class name is kept;
* ``prefix``: length of the common prefix with the ``full`` response of the
  same query and replicate (same sampling seed), as a fraction of the
  ``full`` response; ``identical``: the two responses are equal. An edit
  the model ignores gives ``prefix`` = 1;
* ``error_line_kept``: (error stratum) the line named in the parent's error
  is still in the child's code;
* ``same_error``: (error stratum) the child fails with the parent's error
  type;
* probe readouts ``pf`` (parent fitness), ``cf`` (child failure), ``ob``
  (over budget) from experiment 1, at the source campaign's probe layers
  (Qwen3-8B: 15, 21, 9; older shards name them ``pf_L15`` etc.).

Effects are ``variant - full``, averaged over replicates within a query, then
over queries, with 95% bootstrap CIs over LLaMEA runs. Writes to
``results/<tag>/``: ``cf_samples.csv``, ``cf_means.csv`` and
``cf_effects.csv``.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT

METRICS = [
    "failed",
    "score",
    "over_budget",
    "similarity",
    "same_name",
    "prefix",
    "identical",
    "error_line_kept",
    "same_error",
    "pf",
    "cf",
    "ob",
    "n_gen_tokens",
]


def error_type(text: str | None) -> str | None:
    m = re.search(r"\n?(\w+(?:Error|Exception|Warning))\b", text or "")
    return m.group(1) if m else None


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load_samples(tag: str) -> pd.DataFrame:
    proc = data_dir(EXPERIMENT, "processed", tag)
    queries = {
        q["record_id"]: q
        for q in (json.loads(l) for l in open(proc / "cf_queries.jsonl"))
    }
    rows = []
    for path in sorted(data_dir(EXPERIMENT, "raw", tag).glob("shard-*.jsonl")):
        rows += [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    # Shards written before the readouts were renamed (Qwen3-8B, tag "cf")
    df = df.rename(columns={"pf_L15": "pf", "cf_L21": "cf", "ob_L9": "ob"})
    df = df[~df["infra"].fillna(False).astype(bool)]
    df = df.drop_duplicates(["record_id", "variant", "k"], keep="last")

    def per_row(r):
        q = queries[r["record_id"]]
        code = r.get("code") if isinstance(r.get("code"), str) else ""
        parent = q["parent_code"] or ""
        full_text = q["variants"]["full"]["text"]
        m = re.search(r"On line: (.+)", full_text.split("## Feedback:")[-1])
        err_line = m.group(1).strip() if m else None
        p_err = error_type(full_text.split("## Feedback:")[-1])
        over = r.get("over_budget_instances")
        return pd.Series(
            {
                "parent_fitness": q["parent_fitness"],
                "score": 0.0 if r["failed"] else r["fitness"],
                "over_budget": (
                    np.nan if r["failed"] or over is None else float(over > 0)
                ),
                "similarity": (
                    difflib.SequenceMatcher(None, parent, code, autojunk=False).ratio()
                    if code
                    else np.nan
                ),
                "same_name": (
                    float(r.get("name") == q["parent_name"]) if code else np.nan
                ),
                "error_line_kept": (
                    float(err_line in code) if err_line and code else np.nan
                ),
                "same_error": (
                    float(error_type(r.get("error")) == p_err)
                    if q["stratum"] == "error" and r["failed"] and p_err
                    else (0.0 if q["stratum"] == "error" and p_err else np.nan)
                ),
            }
        )

    df["failed"] = df["failed"].astype(float)
    df = pd.concat([df, df.apply(per_row, axis=1)], axis=1)
    full = df[df["variant"] == "full"].set_index(["record_id", "k"])["response"]
    key = list(zip(df["record_id"], df["k"]))
    ref = [full.get(kk) for kk in key]
    df["prefix"] = [
        (
            common_prefix(a, b) / max(len(b), 1)
            if isinstance(a, str) and isinstance(b, str)
            else np.nan
        )
        for a, b in zip(df["response"], ref)
    ]
    df["identical"] = [
        float(a == b) if isinstance(a, str) and isinstance(b, str) else np.nan
        for a, b in zip(df["response"], ref)
    ]
    return df


def boot_ci(per_query: pd.DataFrame, col: str, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = per_query.dropna(subset=[col])
    runs = d["run"].unique()
    groups = {r: d.loc[d["run"] == r, col].to_numpy() for r in runs}
    means = []
    for _ in range(n_boot):
        vals = np.concatenate([groups[r] for r in rng.choice(runs, len(runs))])
        means.append(vals.mean())
    return np.percentile(means, [2.5, 97.5])


def effects(df: pd.DataFrame) -> pd.DataFrame:
    q = (
        df.groupby(["stratum", "record_id", "run", "variant"])[METRICS]
        .mean()
        .reset_index()
    )
    rows = []
    for stratum, g in q.groupby("stratum"):
        full = g[g["variant"] == "full"].set_index("record_id")
        for variant, gv in g.groupby("variant"):
            if variant == "full":
                continue
            gv = gv.set_index("record_id")
            for m in METRICS:
                if m in ("prefix", "identical"):
                    d = gv[[m, "run"]].rename(columns={m: "diff"})
                    base = np.nan
                else:
                    d = (gv[m] - full[m].reindex(gv.index)).to_frame("diff")
                    d["run"] = gv["run"]
                    base = full[m].reindex(gv.index).mean()
                d = d.dropna()
                if len(d) < 5:
                    continue
                lo, hi = boot_ci(d, "diff")
                rows.append(
                    {
                        "stratum": stratum,
                        "variant": variant,
                        "metric": m,
                        "n_queries": len(d),
                        "full": base,
                        "variant_mean": gv[m].mean(),
                        "effect": d["diff"].mean(),
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="cf")
    args = ap.parse_args()
    out = results_dir(EXPERIMENT, args.tag)

    df = load_samples(args.tag)
    print(f"{len(df)} samples, {df['record_id'].nunique()} queries")
    df.drop(columns=["response", "code", "description"], errors="ignore").to_csv(
        out / "cf_samples.csv", index=False
    )
    means = df.groupby(["stratum", "variant"])[METRICS].mean()
    means["n"] = df.groupby(["stratum", "variant"]).size()
    means.to_csv(out / "cf_means.csv")
    eff = effects(df)
    eff.to_csv(out / "cf_effects.csv", index=False)

    pd.set_option("display.width", 200)
    print(means.round(3).to_string())
    show = ["failed", "score", "over_budget", "similarity", "prefix", "pf"]
    print("\neffects (variant - full), 95% CI over runs:")
    e = eff[eff["metric"].isin(show + ["error_line_kept", "same_error"])]
    print(
        e.assign(
            ci=lambda x: x.apply(
                lambda r: f"{r.effect:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                axis=1,
            )
        )
        .pivot(index=["stratum", "variant"], columns="metric", values="ci")
        .to_string()
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
