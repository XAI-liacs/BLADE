"""Paired-by-seed arm differences vs control and concept shares.

Usage (from exp05_steering): python scripts/07_paired.py <tag> [--with
<tag2> ...]. Reads results/<tag>/runs.csv and solutions.csv written by
06_llamea_analyze.py; --with merges the runs and solutions of other tags
(e.g. the control of llamea-dial for the stage 1 arms). Every arm other
than control is paired with control by seed.
"""

import argparse
import collections

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--with", dest="others", nargs="*", default=[])
args = ap.parse_args()
tags = [args.tag] + args.others
runs = pd.concat([pd.read_csv(f"results/{t}/runs.csv") for t in tags])
s = pd.concat([pd.read_csv(f"results/{t}/solutions.csv") for t in tags])
runs = runs.drop_duplicates(["arm", "seed"])
cols = [
    c
    for c in [
        "best_aocc",
        "failed",
        "n_families",
        "div_concepts",
        "feasible",
        "near_feasible",
        "best_strict",
        "exploration_pct",
        "intensification",
        "exploration_pct_final",
    ]
    if c in runs
]
rng = np.random.default_rng(0)
ctrl = runs[runs.arm == "control"].set_index("seed")
print("paired diff vs control (mean, 95% bootstrap CI over seeds)")
for arm in [a for a in runs.arm.unique() if a != "control"]:
    a = runs[runs.arm == arm].set_index("seed").reindex(ctrl.index)
    out = []
    for c in cols:
        x = (a[c] - ctrl[c]).dropna().to_numpy()
        if len(x) == 0:
            continue
        b = [rng.choice(x, len(x)).mean() for _ in range(4000)]
        out.append(
            f"{c}={x.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]"
        )
    print(f"{arm} (n={len(x)})", "; ".join(out))
for p in ["", "prompt_"]:
    if {p + "explore", p + "exploit"} <= set(runs.arm):
        x = (
            runs[runs.arm == p + "explore"].set_index("seed")[cols]
            - runs[runs.arm == p + "exploit"].set_index("seed")[cols]
        )
        print(p + "explore-exploit", x.mean().round(3).to_dict())
# concept shares among working algorithms
w = s[~s.failed].copy()
w["concepts"] = w.concepts.fillna("")
keys = [
    k
    for k, _ in collections.Counter(
        x for v in w.concepts for x in v.split("|") if x
    ).most_common(12)
]
tab = pd.DataFrame(
    {
        k: w.groupby("arm").concepts.apply(
            lambda v: v.str.split("|").apply(lambda l: k in l).mean()
        )
        for k in keys
    }
)
print(tab.round(2).to_string())
# operator new: steered only on init/new? exploration behaviour per operator
if s.b_exploration_pct.notna().any():
    print(w.groupby(["arm", "operator"]).b_exploration_pct.mean().unstack().round(1))
    print(w.groupby(["arm"]).b_exploration_pct.mean().round(1))
