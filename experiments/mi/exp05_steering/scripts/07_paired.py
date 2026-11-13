"""Paired-by-seed arm differences vs control and concept shares.

Usage (from exp05_steering): python scripts/07_paired.py <tag>. Reads
results/<tag>/runs.csv and solutions.csv written by 06_llamea_analyze.py.
"""

import collections
import sys

import numpy as np
import pandas as pd

tag = sys.argv[1]
d = f"results/{tag}"
runs = pd.read_csv(f"{d}/runs.csv")
s = pd.read_csv(f"{d}/solutions.csv")
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
for arm in ["explore", "exploit", "anneal", "randomdir"]:
    a = runs[runs.arm == arm].set_index("seed")
    out = []
    for c in cols:
        x = (a[c] - ctrl[c]).dropna().to_numpy()
        b = [rng.choice(x, len(x)).mean() for _ in range(4000)]
        out.append(
            f"{c}={x.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]"
        )
    print(arm, "; ".join(out))
x = (
    runs[runs.arm == "explore"].set_index("seed")[cols]
    - runs[runs.arm == "exploit"].set_index("seed")[cols]
)
print("explore-exploit", x.mean().round(3).to_dict())
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
