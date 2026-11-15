"""Budget 200 vs 100: does the gap between arms change with more budget?

Reads ``results/<tag>/solutions.csv`` (06_llamea_analyze.py) of the
warm-started budget-200 campaigns (the first 100 evaluations are the
budget-100 runs). Per run, best AOCC and auc_best (07_paired.py: mean of
the best-so-far curve, failures count as the run's worst) at budget 100
and 200, the gain from 100 to 200, and paired differences vs control (95%
bootstrap CI over seeds). Usage (from exp05_steering):
python scripts/13_budget200.py llamea-ext200 llamea-ext200-adaptive
"""

import sys

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
print(w.groupby("arm").mean().round(3).to_string())
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
ctrl = w.loc["control"]
print("\npaired diff vs control")
for arm in [a for a in w.index.levels[0] if a != "control"]:
    a = w.loc[arm].reindex(ctrl.index)
    for c in ["best100", "best200", "auc100", "auc200", "best_gain"]:
        print(arm, c, ci((a[c] - ctrl[c]).dropna().to_numpy()))
    print(arm, "wins best@200 on", int((a.best200 > ctrl.best200).sum()), "/", len(a))
