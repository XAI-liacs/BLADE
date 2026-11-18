"""Budget 200 vs 100: does the gap between arms change with more budget?

Reads ``results/<tag>/solutions.csv`` (06_llamea_analyze.py) of the
budget-200 campaigns. In the warm-started ones the first 100 evaluations
are the budget-100 runs; in the ones run from scratch at 200 (anneal200,
randomdir200) "@100" is the first 100 evaluations of the 200 run (for
anneal that is mid-schedule, dose near 0). Per run, best AOCC and auc_best
(07_paired.py: mean of the best-so-far curve, failures count as the run's
worst) at budget 100 and 200, the gain from 100 to 200, and paired
differences (95% bootstrap CI over seeds) vs control, explore - exploit,
prompt_explore - prompt_exploit and adaptive_cal - adaptive. Writes
``results/budget200/per_run.csv`` and ``paired.csv``. Usage (from
exp05_steering):
python scripts/13_budget200.py llamea-ext200 llamea-ext200-adaptive \
    llamea-ext200-rest llamea-ext200-seeds llamea-anneal200 \
    llamea-randomdir200
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

s = pd.concat([pd.read_csv(f"results/{t}/solutions.csv") for t in sys.argv[1:]])
s = s.drop_duplicates(["arm", "seed", "idx"])


def stats(g, budget):
    f = g[g.idx < budget].sort_values("idx").aocc.astype(float)
    f = f.where(np.isfinite(f))
    f = f.fillna(f.min())
    return pd.Series({"best": f.max(), "auc": f.cummax().mean(), "n": len(f)})


rows = []
for (arm, seed), g in s.groupby(["arm", "seed"]):
    for b in (100, 200):
        rows.append({"arm": arm, "seed": seed, "budget": b, **stats(g, b)})
r = pd.DataFrame(rows)
assert (r.n == r.budget).all(), r[r.n != r.budget]
w = r.pivot_table(index=["arm", "seed"], columns="budget", values=["best", "auc"])
w.columns = [f"{m}{b}" for m, b in w.columns]
w["best_gain"] = w.best200 - w.best100
w["auc_gain"] = w.auc200 - w.auc100
print("mean over seeds")
m = w.groupby("arm").mean().round(3)
m["seeds"] = w.groupby("arm").size()
print(m.to_string())
rng = np.random.default_rng(0)


def ci(x):
    b = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return (
        f"{x.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"
    )


print("\ngain 100 -> 200 per arm")
for arm, g in w.groupby("arm"):
    print(
        arm,
        "best",
        ci(g.best_gain.to_numpy()),
        "auc",
        ci(g.auc_gain.to_numpy()),
        f"improved on {(g.best_gain > 0).sum()}/{len(g)} seeds",
    )
OUT = Path("results/budget200")
OUT.mkdir(parents=True, exist_ok=True)
w.round(4).to_csv(OUT / "per_run.csv")
arms = list(w.index.levels[0])
pairs = [(a, "control") for a in arms if a != "control"]
pairs += [
    (a, b)
    for a, b in [
        ("explore", "exploit"),
        ("prompt_explore", "prompt_exploit"),
        ("adaptive_cal", "adaptive"),
    ]
    if a in arms and b in arms
]
prow = []
print("\npaired differences (mean [95% CI], seeds, wins)")
for a, b in pairs:
    x, y = w.loc[a], w.loc[b]
    seeds = x.index.intersection(y.index)
    for c in ["best100", "best200", "auc100", "auc200", "best_gain", "auc_gain"]:
        d = (x.loc[seeds, c] - y.loc[seeds, c]).to_numpy()
        b_ = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        lo, hi = np.percentile(b_, [2.5, 97.5])
        prow.append(
            {
                "arm": a,
                "versus": b,
                "metric": c,
                "seeds": len(d),
                "diff": d.mean(),
                "lo": lo,
                "hi": hi,
                "wins": int((d > 0).sum()),
            }
        )
        print(
            f"{a} - {b} {c} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
            f" n={len(d)} wins={(d > 0).sum()}"
        )
pd.DataFrame(prow).round(4).to_csv(OUT / "paired.csv", index=False)
